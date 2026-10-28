"""기존 MediaPipe 추출 결과를 영상 전체 프레임에 맞춰 캐시한다."""

from __future__ import annotations

from pathlib import Path
import zipfile
import hashlib
import json

import numpy as np


def _read_cache(cache_path: Path, video_path: Path, mtime_ns: int, selection: str = "first") -> dict | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as saved:
            if str(saved["video"].item()) != video_path.as_posix():
                return None
            if int(saved["mtime_ns"].item()) != mtime_ns:
                return None
            saved_selection = str(saved["selection"].item()) if "selection" in saved else "first"
            if saved_selection != selection:
                return None
            data = {
                "has_fingers": bool(saved["has_fingers"].item()) if "has_fingers" in saved else True,
                "lms": saved["lms"].copy(),
                "valid": saved["valid"].copy(),
                "fps": float(saved["fps"].item()),
                "w": int(saved["w"].item()),
                "h": int(saved["h"].item()),
                "total": int(saved["total"].item()),
                "video": video_path.as_posix(),
                "mtime_ns": mtime_ns,
                "raw_lms": saved["raw_lms"].copy() if "raw_lms" in saved else saved["lms"].copy(),
            }
        if data["lms"].shape != (data["total"], 33, 4):
            return None
        if data["raw_lms"].shape != data["lms"].shape:
            return None
        if data["valid"].shape != (data["total"],):
            return None
        if data["lms"].dtype != np.float32 or data["valid"].dtype != np.bool_:
            return None
        if data["fps"] <= 0 or data["w"] <= 0 or data["h"] <= 0:
            return None
        if not np.isnan(data["lms"][~data["valid"]]).all():
            return None
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None
    data["cache_hit"] = True
    return data


def extract_pose(
    video_path: Path,
    cache_path: Path,
    exercise: str,
    *,
    no_cache: bool = False,
    plate_rows: list | None = None,
    pose_cfg=None,
    rep_mask=None,
) -> dict:
    """extract_sequence의 검출·스무딩을 재사용하고 경로·mtime으로 캐시를 확인한다."""
    video_path = Path(video_path).resolve()
    cache_path = Path(cache_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    from feedback import plate
    from core.config import DEFAULT
    from dataclasses import asdict
    pose_cfg = pose_cfg or DEFAULT.pose
    mtime_ns = video_path.stat().st_mtime_ns
    selection = "first" if plate_rows is None else "plate-wrist-v2:" + hashlib.sha256(
        json.dumps([exercise, [(r['seq'], r.get('plate')) for r in plate_rows]],
                   sort_keys=True).encode()).hexdigest()
    if plate_rows is not None:
        selection += ":" + json.dumps(asdict(pose_cfg), sort_keys=True)
        if rep_mask is not None:
            selection += ":rep:" + hashlib.sha256(np.asarray(rep_mask, dtype=bool).tobytes()).hexdigest()
    if not no_cache:
        cached = _read_cache(cache_path, video_path, mtime_ns, selection)
        if cached is not None:
            return cached

    from export_overlay import video_info
    from offline_compare import extract_sequence

    info = video_info(video_path)
    sequence = (extract_sequence(video_path, exercise) if plate_rows is None
                else (extract_plate_sequence(video_path, exercise, plate_rows)
                      if pose_cfg.backend == 'mediapipe' else
                      extract_backend_sequence(video_path, exercise, plate_rows, pose_cfg,
                                                    **({"rep_mask": rep_mask} if rep_mask is not None else {}))))
    lms = np.full((len(sequence), 33, 4), np.nan, dtype=np.float32)
    valid = np.array([frame is not None for frame in sequence], dtype=bool)
    for frame_index, frame in enumerate(sequence):
        if frame is not None:
            lms[frame_index] = np.asarray(frame, dtype=np.float32)
    data = {
        "has_fingers": plate_rows is None or pose_cfg.backend == "mediapipe",
        "lms": lms,
        "valid": valid,
        "fps": float(info["fps"]),
        "w": int(info["w"]),
        "h": int(info["h"]),
        "total": len(sequence),
        "video": video_path.as_posix(),
        "mtime_ns": mtime_ns,
        "selection": selection,
        "raw_lms": np.asarray(getattr(sequence, "raw_lms", lms), dtype=np.float32),
    }
    if hasattr(sequence, 'timings'):
        data['inference_timing'] = sequence.timings
        data['score_distribution'] = sequence.score_distribution
        data['tracking'] = sequence.tracking
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{k:v for k,v in data.items() if k not in ('inference_timing', 'score_distribution', 'tracking')})
    data["cache_hit"] = False
    return data


def choose_plate_candidate(candidates, row, previous_center, exercise, size, *, cfg=None, in_rep=False):
    """select_lifter consumes pixel x/y; feedback smoothing consumes normalized x/y."""
    from feedback import plate  # establish the plate package import path
    from core.recog.pose import select_lifter, person_center

    from core.config import DEFAULT
    cfg = cfg or DEFAULT.pose
    pixel = [np.asarray(candidate, dtype=float).copy() for candidate in candidates]
    for candidate in pixel:
        candidate[:, :2] *= np.asarray(size)
    box = row.get("plate", {}).get("box") if row.get("plate", {}).get("used") else None
    index = select_lifter(pixel, box, previous_center, exercise, cfg=cfg, image_size=size, in_rep=in_rep)
    if index is None:
        return None, previous_center
    center = person_center(pixel[index])
    return candidates[index], center if center is not None else previous_center


class _PlateSequence(list):
    """Keep selected raw landmarks for quality checks before EMA smoothing."""


def extract_plate_sequence(video_path, exercise, rows):
    import cv2
    from dataclasses import replace
    from core.config import DEFAULT
    from unified_feedback_v4 import (create_landmarker, MODEL_PATH, extract_all_landmarks,
                                     LandmarkSmoother, valid_lms)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    smoother, sequence, previous = LandmarkSmoother(), _PlateSequence(), None
    sequence.raw_lms = []
    try:
        with create_landmarker(MODEL_PATH) as landmarker:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                seq = len(sequence)
                if seq >= len(rows):
                    raise ValueError("포즈 프레임이 원판 rows보다 많습니다")
                candidates = extract_all_landmarks(landmarker, image, int(seq / fps * 1000))
                raw, previous = choose_plate_candidate(candidates, rows[seq], previous, exercise,
                                                        (image.shape[1], image.shape[0]),
                                                        cfg=replace(DEFAULT.pose, backend="mediapipe"))
                sequence.append(smoother.update(raw) if valid_lms(raw) else None)
                sequence.raw_lms.append(np.asarray(raw, dtype=np.float32).copy() if valid_lms(raw)
                                        else np.full((33, 4), np.nan, dtype=np.float32))
    finally:
        capture.release()
    if len(sequence) != len(rows):
        raise ValueError(f"원판 rows({len(rows)})와 포즈 프레임({len(sequence)})이 다릅니다")
    return sequence


def extract_backend_sequence(video_path, exercise, rows, cfg, *, rep_mask=None):
    """Route RTMPose through the unchanged feedback EMA and MP33 contract."""
    import cv2
    from core.recog.pose import PoseEstimator
    from unified_feedback_v4 import LandmarkSmoother, valid_lms

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    smoother, sequence, previous = LandmarkSmoother(), _PlateSequence(), None
    sequence.raw_lms = []
    if rep_mask is None:
        rep_mask = np.zeros(len(rows), dtype=bool)
    if len(rep_mask) != len(rows):
        raise ValueError('Rep mask and plate rows must share the frame clock')
    try:
        with PoseEstimator(cfg=cfg) as estimator:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                seq = len(sequence)
                if seq >= len(rows):
                    raise ValueError('Pose has more frames than plate rows')
                h, w = image.shape[:2]
                candidates = estimator(image, seq/fps)
                for candidate in candidates:
                    candidate[:, :2] /= [w, h]
                raw, previous = choose_plate_candidate(candidates, rows[seq], previous,
                                                        exercise, (w, h), cfg=cfg, in_rep=bool(rep_mask[seq]))
                if hasattr(estimator, "observe"):
                    pixels = None if raw is None else np.asarray(raw).copy()
                    if pixels is not None:
                        pixels[:, :2] *= [w, h]
                    box = rows[seq].get("plate", {}).get("box") if rows[seq].get("plate", {}).get("used") else None
                    estimator.observe(pixels, box, (w, h))
                if raw is None:
                    previous = None  # continuity concerns consecutive observed frames only
                sequence.append(smoother.update(raw) if valid_lms(raw) else None)
                sequence.raw_lms.append(np.asarray(raw, dtype=np.float32).copy() if valid_lms(raw)
                                        else np.full((33, 4), np.nan, dtype=np.float32))
            sequence.tracking = dict(detection_frames=estimator.detection_frames, lost_frames=estimator.lost_frames)
            sequence.timings = estimator.timings.copy()
            scores = np.asarray(estimator.scores)
            sequence.score_distribution = dict(n=len(scores),
                quantiles=dict(zip(('min','p01','p05','p25','p50','p75','p95','p99','max'),
                                   np.quantile(scores, [0,.01,.05,.25,.5,.75,.95,.99,1]).tolist())) if len(scores) else {})
    finally:
        capture.release()
    if len(sequence) != len(rows):
        raise ValueError('Pose and plate frame counts differ')
    return sequence
