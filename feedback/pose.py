"""기존 MediaPipe 추출 결과를 영상 전체 프레임에 맞춰 캐시한다."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np


def _read_cache(cache_path: Path, video_path: Path, mtime_ns: int) -> dict | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as saved:
            if str(saved["video"].item()) != video_path.as_posix():
                return None
            if int(saved["mtime_ns"].item()) != mtime_ns:
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
) -> dict:
    """extract_sequence의 검출·스무딩을 재사용하고 경로·mtime으로 캐시를 확인한다."""
    video_path = Path(video_path).resolve()
    cache_path = Path(cache_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    mtime_ns = video_path.stat().st_mtime_ns
    if not no_cache:
        cached = _read_cache(cache_path, video_path, mtime_ns)
        if cached is not None:
            return cached

    from export_overlay import video_info
    from offline_compare import extract_sequence

    info = video_info(video_path)
    sequence = extract_sequence(video_path, exercise)
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
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    data["cache_hit"] = False
    return data
