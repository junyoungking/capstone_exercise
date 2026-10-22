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
                "lms": saved["lms"].copy(),
                "valid": saved["valid"].copy(),
                "fps": float(saved["fps"].item()),
                "w": int(saved["w"].item()),
                "h": int(saved["h"].item()),
                "total": int(saved["total"].item()),
                "video": video_path.as_posix(),
                "mtime_ns": mtime_ns,
            }
        if data["lms"].shape != (data["total"], 33, 4):
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
) -> dict:
    """extract_sequence의 검출·스무딩을 재사용하고 경로·mtime으로 캐시를 확인한다."""
    video_path = Path(video_path).resolve()
    cache_path = Path(cache_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    mtime_ns = video_path.stat().st_mtime_ns
    selection = "first" if plate_rows is None else "plate-v1:" + hashlib.sha256(
        json.dumps([exercise, [(r['seq'], r.get('plate')) for r in plate_rows]],
                   sort_keys=True).encode()).hexdigest()
    if not no_cache:
        cached = _read_cache(cache_path, video_path, mtime_ns, selection)
        if cached is not None:
            return cached

    from export_overlay import video_info
    from offline_compare import extract_sequence

    info = video_info(video_path)
    sequence = (extract_sequence(video_path, exercise) if plate_rows is None
                else extract_plate_sequence(video_path, exercise, plate_rows))
    lms = np.full((len(sequence), 33, 4), np.nan, dtype=np.float32)
    valid = np.array([frame is not None for frame in sequence], dtype=bool)
    for frame_index, frame in enumerate(sequence):
        if frame is not None:
            lms[frame_index] = np.asarray(frame, dtype=np.float32)
    data = {
        "lms": lms,
        "valid": valid,
        "fps": float(info["fps"]),
        "w": int(info["w"]),
        "h": int(info["h"]),
        "total": len(sequence),
        "video": video_path.as_posix(),
        "mtime_ns": mtime_ns,
        "selection": selection,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    data["cache_hit"] = False
    return data


def choose_plate_candidate(candidates, row, previous_center, exercise, size):
    """select_lifter consumes pixel x/y; feedback smoothing consumes normalized x/y."""
    from feedback import plate  # establish the plate package import path
    from core.recog.pose import select_lifter, person_center

    pixel = [np.asarray(candidate, dtype=float).copy() for candidate in candidates[:2]]
    for candidate in pixel:
        candidate[:, :2] *= np.asarray(size)
    box = row.get("plate", {}).get("box") if row.get("plate", {}).get("used") else None
    index = select_lifter(pixel, box, previous_center, exercise, image_size=size)
    if index is None:
        return None, previous_center
    center = person_center(pixel[index])
    return candidates[index], center if center is not None else previous_center


def extract_plate_sequence(video_path, exercise, rows):
    import cv2
    from unified_feedback_v4 import (create_landmarker, MODEL_PATH, extract_all_landmarks,
                                     LandmarkSmoother, valid_lms)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    smoother, sequence, previous = LandmarkSmoother(), [], None
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
                                                        (image.shape[1], image.shape[0]))
                sequence.append(smoother.update(raw) if valid_lms(raw) else None)
    finally:
        capture.release()
    if len(sequence) != len(rows):
        raise ValueError(f"원판 rows({len(rows)})와 포즈 프레임({len(sequence)})이 다릅니다")
    return sequence
