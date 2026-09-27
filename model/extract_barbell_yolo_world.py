"""Extract YOLO-World barbell detections into phase-model feature caches.

The output cache intentionally mirrors the existing pose cache contract:
``kpts`` is shaped ``[T, 1, 3]`` and stores
``[x_center, y_center, confidence]`` for a single pseudo-joint representing the
barbell.  Additional arrays preserve detector evidence so the downstream
experiment can report detection quality without requiring manual labels.

The module is import-safe when ``ultralytics`` is not installed.  The optional
dependency is imported only when extraction is requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

try:  # Import works both as `python -m model.extract_barbell_yolo_world` and direct script.
    from model import train_ablation as ta
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    import train_ablation as ta  # type: ignore


DEFAULT_MODEL = "yolov8s-world.pt"
DEFAULT_CLASSES = ("barbell", "weightlifting bar", "barbell with weight plates", "")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def _load_yolo_world(model_path: str, prompt_classes: Sequence[str]):
    try:
        from ultralytics import YOLO, YOLOWorld
    except ImportError as exc:  # pragma: no cover - dependency optional in tests
        raise RuntimeError("YOLO-World barbell extraction requires `pip install ultralytics`.") from exc

    try:
        model = YOLOWorld(model_path)
    except Exception:
        # Older/newer Ultralytics builds can route world weights through YOLO().
        model = YOLO(model_path)

    if hasattr(model, "set_classes"):
        model.set_classes(list(prompt_classes))
    return model


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _best_box(result: Any) -> tuple[np.ndarray, float, int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return np.full((4,), np.nan, dtype=np.float32), 0.0, -1
    xyxy = _as_numpy(getattr(boxes, "xyxy", None)).astype(np.float32)
    if xyxy.size == 0:
        return np.full((4,), np.nan, dtype=np.float32), 0.0, -1
    xyxy = xyxy.reshape(-1, 4)
    conf = _as_numpy(getattr(boxes, "conf", None)).astype(np.float32).reshape(-1)
    if conf.size != len(xyxy):
        conf = np.ones((len(xyxy),), dtype=np.float32)
    cls = _as_numpy(getattr(boxes, "cls", None)).astype(np.float32).reshape(-1)
    if cls.size != len(xyxy):
        cls = np.full((len(xyxy),), -1, dtype=np.float32)
    best = int(np.argmax(conf))
    return xyxy[best], float(conf[best]), int(cls[best])


def _interpolate_centers(raw_centers: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Fill missing centers for temporal continuity while preserving confidence."""

    centers = np.asarray(raw_centers, dtype=np.float32).copy()
    mask = np.asarray(detected, dtype=bool)
    if len(centers) == 0:
        return centers.reshape(0, 2)
    if not mask.any():
        centers[:, 0] = 0.5
        centers[:, 1] = 0.5
        return centers
    frame_idx = np.arange(len(centers), dtype=np.float32)
    for dim in range(2):
        values = centers[:, dim]
        valid = mask & np.isfinite(values)
        if not valid.any():
            centers[:, dim] = 0.5
            continue
        centers[:, dim] = np.interp(frame_idx, frame_idx[valid], values[valid]).astype(np.float32)
    return centers


def _smooth_centers(centers: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or len(centers) == 0:
        return centers.astype(np.float32, copy=False)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(centers.astype(np.float32), ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    smoothed = np.stack([np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(2)], axis=1)
    return smoothed.astype(np.float32)


def extract_barbell_yolo_world(
    video_path: Path | str,
    out_path: Path | str,
    *,
    model_path: str = DEFAULT_MODEL,
    prompt_classes: Sequence[str] = DEFAULT_CLASSES,
    model: Optional[object] = None,
    device: Optional[str] = None,
    imgsz: int = 640,
    conf: float = 0.05,
    iou: float = 0.7,
    smooth_window: int = 5,
    stream_video: bool = True,
) -> dict[str, Any]:
    """Extract one video's selected barbell box as a one-node temporal signal."""

    import cv2

    video_path = Path(video_path)
    out_path = Path(out_path)
    yolo = model if model is not None else _load_yolo_world(model_path, prompt_classes)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"open fail: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    boxes: list[np.ndarray] = []
    centers: list[list[float]] = []
    confs: list[float] = []
    cls_ids: list[int] = []
    detected: list[bool] = []

    def append_result(result: Any) -> None:
        nonlocal width, height
        if result is not None and getattr(result, "orig_shape", None):
            h, w = result.orig_shape[:2]
            height = int(h or height)
            width = int(w or width)
        box, score, cls_id = _best_box(result)

        if np.isfinite(box).all() and score > 0.0:
            norm_box = np.array(
                [
                    box[0] / max(width, 1),
                    box[1] / max(height, 1),
                    box[2] / max(width, 1),
                    box[3] / max(height, 1),
                ],
                dtype=np.float32,
            )
            cx = float((norm_box[0] + norm_box[2]) / 2.0)
            cy = float((norm_box[1] + norm_box[3]) / 2.0)
            hit = True
        else:
            norm_box = np.full((4,), np.nan, dtype=np.float32)
            cx = cy = float("nan")
            hit = False
            score = 0.0
            cls_id = -1

        boxes.append(norm_box)
        centers.append([cx, cy])
        confs.append(float(score))
        cls_ids.append(int(cls_id))
        detected.append(bool(hit))

    kwargs: dict[str, Any] = {
        "verbose": False,
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
    }
    if device:
        kwargs["device"] = device

    if stream_video and hasattr(yolo, "predict"):
        for result in yolo.predict(source=str(video_path), stream=True, **kwargs):
            append_result(result)
    else:
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = yolo.predict(frame, **kwargs) if hasattr(yolo, "predict") else yolo(frame, **kwargs)
            append_result(results[0] if results else None)
        cap.release()

    raw_centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    detected_arr = np.asarray(detected, dtype=bool)
    detection_conf = np.asarray(confs, dtype=np.float32)
    filled_centers = _interpolate_centers(raw_centers, detected_arr)
    smoothed_centers = _smooth_centers(filled_centers, smooth_window)

    kpts = np.zeros((len(smoothed_centers), 1, 3), dtype=np.float32)
    if len(smoothed_centers):
        kpts[:, 0, :2] = np.clip(smoothed_centers, 0.0, 1.0)
        kpts[:, 0, 2] = detection_conf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        kpts=kpts,
        fps=fps,
        num_frames=len(kpts),
        source=str(video_path),
        width=width,
        height=height,
        model_name=str(model_path),
        prompt_classes=np.asarray(list(prompt_classes), dtype=object),
        conf_thres=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        raw_centers=raw_centers,
        centers=smoothed_centers,
        detected=detected_arr,
        detection_conf=detection_conf,
        cls_ids=np.asarray(cls_ids, dtype=np.int64),
        detection_rate=float(detected_arr.mean()) if len(detected_arr) else 0.0,
        smooth_window=int(smooth_window),
    )
    return {
        "video": str(video_path),
        "out": str(out_path),
        "frames": int(len(kpts)),
        "fps": float(fps),
        "detected_frames": int(detected_arr.sum()),
        "detection_rate": float(detected_arr.mean()) if len(detected_arr) else 0.0,
        "mean_conf": float(detection_conf[detected_arr].mean()) if detected_arr.any() else 0.0,
    }


def iter_labeled_videos(
    *,
    label_dir: Path,
    video_dir: Path,
    out_dir: Path,
    exercise_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterable[Tuple[str, str, Path, Path]]:
    labels = ta.load_labels(label_dir)
    emitted = 0
    for typ, name in sorted(labels.keys()):
        if exercise_type and typ != exercise_type:
            continue
        src = video_dir / typ / name
        dst = ta.pose_out_path(typ, name, pose_backend=ta.POSE_BACKEND_BARBELL, barbell_dir=out_dir)
        if not src.exists():
            continue
        yield typ, name, src, dst
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def write_summary(path: Path, rows: Sequence[dict[str, Any]], *, model: str, prompt_classes: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rates = [float(r.get("detection_rate", 0.0)) for r in rows if not r.get("skipped")]
    payload = {
        "model": model,
        "prompt_classes": list(prompt_classes),
        "videos": len(rows),
        "processed": sum(1 for r in rows if not r.get("skipped")),
        "skipped_existing": sum(1 for r in rows if r.get("skipped")),
        "mean_detection_rate": float(np.mean(rates)) if rates else 0.0,
        "min_detection_rate": float(np.min(rates)) if rates else 0.0,
        "rows": list(rows),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract YOLO-World barbell center trajectories for offline phase experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics YOLO-World model path/name.")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. 0, cpu, cuda:0.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Low detection threshold; best box is selected per frame.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        help="Prompt class to set on YOLO-World. Repeatable; defaults to barbell prompts plus an empty background class.",
    )
    parser.add_argument(
        "--background-class",
        action="store_true",
        help="Append an empty background class to custom prompt lists; the built-in default already includes one.",
    )
    parser.add_argument("--smooth-window", type=int, default=5, help="Odd moving-average window for interpolated centers.")
    parser.add_argument("--no-stream-video", action="store_true", help="Disable Ultralytics video-stream inference fallback to frame calls.")
    parser.add_argument("--type", choices=ta.CLASS_LIST, default=None, help="Optional exercise type filter.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of videos to process.")
    parser.add_argument("--video-dir", type=Path, default=ta.VIDEO_DIR)
    parser.add_argument("--label-dir", type=Path, default=ta.LABEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=ta.BARBELL_DIR)
    parser.add_argument("--summary", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--fail-log", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt_classes = list(args.classes or DEFAULT_CLASSES)
    if args.background_class and "" not in prompt_classes:
        prompt_classes.append("")

    fail_log = args.fail_log or (ta.WORK_DIR / "barbell_yolo_world_fail.txt")
    summary_path = args.summary or (args.out_dir / "barbell_detection_summary.json")
    yolo = _load_yolo_world(args.model, prompt_classes)
    targets = list(
        iter_labeled_videos(
            label_dir=args.label_dir,
            video_dir=args.video_dir,
            out_dir=args.out_dir,
            exercise_type=args.type,
            limit=args.limit,
        )
    )
    rows: list[dict[str, Any]] = []
    fail_rows: list[str] = []
    ok_count = 0
    for typ, name, src, dst in tqdm(targets, desc="barbell-yolo-world"):
        if dst.exists() and not args.overwrite:
            rows.append({"type": typ, "name": name, "video": str(src), "out": str(dst), "skipped": True})
            ok_count += 1
            continue
        try:
            row = extract_barbell_yolo_world(
                src,
                dst,
                model_path=args.model,
                prompt_classes=prompt_classes,
                model=yolo,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                smooth_window=args.smooth_window,
                stream_video=not args.no_stream_video,
            )
            row.update({"type": typ, "name": name, "skipped": False})
            rows.append(row)
            ok_count += 1
        except Exception as exc:  # pragma: no cover - video/model environment dependent
            fail_rows.append(f"{typ},{name},{src},{exc}")
            rows.append({"type": typ, "name": name, "video": str(src), "out": str(dst), "error": str(exc)})

    if fail_rows:
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(fail_rows) + "\n", encoding="utf-8")
    write_summary(summary_path, rows, model=args.model, prompt_classes=prompt_classes)
    print(
        "Barbell extraction complete: "
        f"success={ok_count} failed={len(fail_rows)} summary={summary_path} "
        f"fail_log={fail_log if fail_rows else 'n/a'}"
    )
    return 1 if fail_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
