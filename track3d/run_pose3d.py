"""Extract H36M-17 3D pose for the report's 3D view.

The module keeps RTMLib, ONNX Runtime, and Torch imports inside CLI-only
functions so the synthetic tests can run in the normal capstone environment.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feedback import config as fb_config


@dataclass(frozen=True)
class Pose3DStats:
    total_frames: int
    fps: float
    width: int
    height: int
    detected_frames: int
    interpolated_frames: int
    null_frames: int
    two_d_seconds: float
    three_d_seconds: float
    total_seconds: float
    bone_cv: list[dict[str, float | int | None]]


def select_largest_box(boxes: np.ndarray) -> np.ndarray | None:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return None
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[1] < 4:
        raise ValueError(f"boxes must have at least 4 columns, got {boxes.shape}")
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    index = int(np.argmax(widths * heights))
    return boxes[index:index + 1]


def contiguous_runs(indices: Iterable[int]) -> list[np.ndarray]:
    values = np.asarray(list(indices), dtype=np.int64)
    if values.size == 0:
        return []
    return [run for run in np.split(values, np.flatnonzero(np.diff(values) > 1) + 1) if run.size]


def interpolate_short_gaps(points: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill missing 2D keypoints for short gaps and keep long gaps as NaN.

    Edge gaps are held at the nearest valid frame when they are short. Long gaps
    stay NaN, which splits 3D inference into separate valid runs.
    """

    result = np.asarray(points, dtype=np.float32).copy()
    if result.ndim != 3 or result.shape[1:] != (17, 2):
        raise ValueError(f"points must be [frames, 17, 2], got {result.shape}")
    missing = ~np.isfinite(result).all(axis=(1, 2))
    interpolated = np.zeros(result.shape[0], dtype=bool)
    valid = np.flatnonzero(~missing)
    if valid.size == 0:
        return result, interpolated, np.ones(result.shape[0], dtype=bool)

    for gap in contiguous_runs(np.flatnonzero(missing)):
        if len(gap) > max_gap:
            continue
        before = valid[valid < gap[0]]
        after = valid[valid > gap[-1]]
        if before.size and after.size:
            left = int(before[-1])
            right = int(after[0])
            for joint in range(17):
                for axis in range(2):
                    result[gap, joint, axis] = np.interp(gap, [left, right], result[[left, right], joint, axis])
        elif before.size:
            result[gap] = result[int(before[-1])]
        elif after.size:
            result[gap] = result[int(after[0])]
        interpolated[gap] = True
    null_frames = ~np.isfinite(result).all(axis=(1, 2))
    return result, interpolated, null_frames


def valid_runs_after_interpolation(points: np.ndarray) -> list[np.ndarray]:
    valid = np.flatnonzero(np.isfinite(points).all(axis=(1, 2)))
    return contiguous_runs(valid)


def normalize_keypoints(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array([1.0, height / width], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) / float(width) * 2.0 - scale


def infer_runs_3d(model, points2d: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    import torch

    output = np.full((points2d.shape[0], 17, 3), np.nan, dtype=np.float32)
    pad = (model.receptive_field() - 1) // 2
    started = perf_counter()
    with torch.inference_mode():
        for run in valid_runs_after_interpolation(points2d):
            normalized = normalize_keypoints(points2d[run], width, height)
            padded = np.pad(normalized, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            prediction = model(torch.from_numpy(padded[None].copy())).cpu().numpy()[0]
            output[run] = prediction.astype(np.float32)
    return output, perf_counter() - started


def bone_length_cv(frames: np.ndarray, parents: Iterable[int] = fb_config.POSE3D_PARENTS) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    parents = tuple(parents)
    for joint, parent in enumerate(parents):
        if parent < 0:
            continue
        vectors = frames[:, joint] - frames[:, parent]
        lengths = np.linalg.norm(vectors, axis=1)
        lengths = lengths[np.isfinite(lengths)]
        if lengths.size == 0:
            mean = None
            std = None
            cv = None
        else:
            mean_value = float(np.mean(lengths))
            std_value = float(np.std(lengths))
            mean = mean_value
            std = std_value
            cv = None if mean_value == 0 else std_value / mean_value
        rows.append({"joint": joint, "parent": parent, "samples": int(lengths.size), "mean": mean, "std": std, "cv": cv})
    return rows


def frames_for_json(frames: np.ndarray) -> list[list[list[float]] | None]:
    rows: list[list[list[float]] | None] = []
    for frame in frames:
        if not np.isfinite(frame).all():
            rows.append(None)
        else:
            rows.append([[round(float(value), 6) for value in joint] for joint in frame])
    return rows


def write_pose3d_js(path: Path, who: str, fps: float, frames: np.ndarray) -> dict:
    payload = {
        "joints": "h36m17",
        "joint_names": list(fb_config.POSE3D_JOINTS),
        "parents": list(fb_config.POSE3D_PARENTS),
        "fps": fps,
        "total": int(frames.shape[0]),
        "frames": frames_for_json(frames),
    }
    wrapper = f"window.POSE3D = window.POSE3D || {{}};\nwindow.POSE3D[{json.dumps(who)}] = "
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(wrapper + json.dumps(payload, ensure_ascii=False, allow_nan=False) + ";\n", encoding="utf-8")
    return payload


def write_stats(path: Path, stats: Pose3DStats, mode: str, models: dict[str, str], checkpoint: Path) -> None:
    packages = {}
    for name in ("rtmlib", "onnxruntime", "torch", "numpy", "opencv-contrib-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "mode": mode,
        "models": models,
        "checkpoint": str(checkpoint),
        "packages": packages,
        "stats": {
            "total_frames": stats.total_frames,
            "fps": stats.fps,
            "width": stats.width,
            "height": stats.height,
            "detected_frames": stats.detected_frames,
            "interpolated_frames": stats.interpolated_frames,
            "null_frames": stats.null_frames,
            "two_d_seconds": stats.two_d_seconds,
            "three_d_seconds": stats.three_d_seconds,
            "total_seconds": stats.total_seconds,
            "seconds_per_frame": stats.total_seconds / stats.total_frames if stats.total_frames else None,
            "bone_cv": stats.bone_cv,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ensure_models(model_dir: Path, mode: str) -> dict[str, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    files = fb_config.POSE3D_MODEL_FILES[mode]
    if mode != "lightweight":
        return {"det": model_dir / files["local_det"], "pose": model_dir / files["local_pose"]}

    cache = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"
    paths = {}
    for key in ("det", "pose"):
        source_name = files[key]
        target = model_dir / files[f"local_{key}"]
        source = cache / source_name
        if not target.exists():
            if not source.exists():
                raise FileNotFoundError(f"RTMLib lightweight model is missing: {source}")
            shutil.copy2(source, target)
        paths[key] = target
    return paths


def load_videopose_model(model_dir: Path):
    import torch

    model_path = fb_config.POSE3D_VENDOR_MODEL
    if not model_path.exists():
        model_path = model_dir / "videopose_model.py"
    spec = importlib.util.spec_from_file_location("track3d_videopose_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import VideoPose3D model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(fb_config.POSE3D_THREADS)
    model = module.TemporalModel(
        17,
        2,
        17,
        list(fb_config.POSE3D_VIDEPOSE_FILTER_WIDTHS),
        causal=False,
        channels=fb_config.POSE3D_VIDEPOSE_CHANNELS,
    )
    checkpoint_path = model_dir / fb_config.POSE3D_CHECKPOINT_NAME
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_pos"], strict=True)
    model.eval()
    return model, checkpoint_path


def bind_onnx_threads(body) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = fb_config.POSE3D_THREADS
    options.inter_op_num_threads = 1
    for tool in (body.det_model, body.pose_model):
        tool.session = ort.InferenceSession(tool.onnx_model, options, providers=["CPUExecutionProvider"])


def read_video_info(video: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, width, height, total


def extract_2d_keypoints_from_video(
    video: Path,
    mode: str,
    model_paths: dict[str, Path],
    expected_frames: int,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    from rtmlib import Body

    body = Body(
        det=str(model_paths["det"]),
        pose=str(model_paths["pose"]),
        det_input_size=fb_config.POSE3D_DET_INPUT_SIZES[mode],
        pose_input_size=fb_config.POSE3D_POSE_INPUT_SIZES[mode],
        mode=mode,
        backend="onnxruntime",
        device="cpu",
        to_openpose=False,
    )
    bind_onnx_threads(body)
    allocated = max(expected_frames, 1)
    points = np.full((allocated, 17, 2), np.nan, dtype=np.float32)
    detected = np.zeros(allocated, dtype=bool)
    started = perf_counter()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index >= points.shape[0]:
                points = np.concatenate([points, np.full((256, 17, 2), np.nan, dtype=np.float32)], axis=0)
                detected = np.concatenate([detected, np.zeros(256, dtype=bool)])
            boxes = select_largest_box(np.asarray(body.det_model(image)))
            if boxes is not None:
                keypoints, _scores = body.pose_model(image, bboxes=boxes)
                keypoints = np.asarray(keypoints, dtype=np.float32)
                if keypoints.shape[0] > 0:
                    points[index] = keypoints[0]
                    detected[index] = True
            if progress_every > 0 and ((index + 1) % progress_every == 0):
                total_text = str(expected_frames) if expected_frames else "?"
                print(f"2D {index + 1}/{total_text} elapsed={perf_counter() - started:.2f}s", flush=True)
            index += 1
    finally:
        cap.release()
    if index == 0:
        raise RuntimeError(f"No frames read from video: {video}")
    if progress_every > 0 and index % progress_every != 0:
        print(f"2D {index}/{index} elapsed={perf_counter() - started:.2f}s", flush=True)
    return points[:index], detected[:index], index, perf_counter() - started


def run(video: Path, who: str, out: Path, mode: str, model_dir: Path, progress_every: int) -> Pose3DStats:
    started = perf_counter()
    model_paths = ensure_models(model_dir, mode)
    fps, width, height, expected_frames = read_video_info(video)
    print(f"video={video} frames={expected_frames} fps={fps:.3f} size={width}x{height}", flush=True)
    points2d_raw, detected, total_frames, two_d_seconds = extract_2d_keypoints_from_video(
        video, mode, model_paths, expected_frames, progress_every
    )
    if expected_frames and total_frames != expected_frames:
        print(f"frame_count_notice expected={expected_frames} read={total_frames}", flush=True)
    points2d, interpolated, nulls = interpolate_short_gaps(points2d_raw, fb_config.POSE3D_MAX_INTERPOLATE_GAP)
    model, checkpoint = load_videopose_model(model_dir)
    frames3d, three_d_seconds = infer_runs_3d(model, points2d, width, height)
    payload = write_pose3d_js(out, who, fps, frames3d)
    bone_cv = bone_length_cv(frames3d)
    total_seconds = perf_counter() - started
    stats = Pose3DStats(
        total_frames=int(payload["total"]),
        fps=fps,
        width=width,
        height=height,
        detected_frames=int(detected.sum()),
        interpolated_frames=int(interpolated.sum()),
        null_frames=int(nulls.sum()),
        two_d_seconds=two_d_seconds,
        three_d_seconds=three_d_seconds,
        total_seconds=total_seconds,
        bone_cv=bone_cv,
    )
    write_stats(out.with_suffix(out.suffix + ".stats.json"), stats, mode, {k: str(v) for k, v in model_paths.items()}, checkpoint)
    print(
        f"done who={who} frames={stats.total_frames} detected={stats.detected_frames} "
        f"interpolated={stats.interpolated_frames} null={stats.null_frames} "
        f"2d={stats.two_d_seconds:.2f}s 3d={stats.three_d_seconds:.2f}s total={stats.total_seconds:.2f}s",
        flush=True,
    )
    for row in bone_cv:
        cv = row["cv"]
        cv_text = "null" if cv is None else f"{cv:.6f}"
        print(f"bone_cv joint={row['joint']} parent={row['parent']} samples={row['samples']} cv={cv_text}", flush=True)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract report-only H36M17 3D pose")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--who", required=True, choices=("user", "ref"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", default=fb_config.POSE3D_RTMLIB_MODE, choices=("lightweight", "balanced"))
    parser.add_argument("--model-dir", default=Path("out/pose3d_models"), type=Path)
    parser.add_argument("--progress-every", default=50, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    run(args.video, args.who, args.out, args.mode, args.model_dir, args.progress_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
