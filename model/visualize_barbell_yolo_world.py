"""Render cached YOLO-World barbell boxes back onto source videos.

The extractor stores one selected normalized box per frame in each ``.npz``
cache.  This script finds the best/worst videos from a detection summary,
optionally per exercise, draws those cached boxes directly on the frames, and
writes review videos.

Example:
    python model/visualize_barbell_yolo_world.py \
      --summary phase_experiments/barbell_yolo_world/full/barbell_detection_summary.json

    python model/visualize_barbell_yolo_world.py \
      --selection per-exercise \
      --summary phase_experiments/barbell_yolo_world/full/barbell_detection_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


DEFAULT_SUMMARY = Path("phase_experiments/barbell_yolo_world/full/barbell_detection_summary.json")
DEFAULT_OUT_DIR = Path("phase_experiments/barbell_yolo_world/visualizations")


@dataclass(frozen=True)
class VideoVisualization:
    selection: str
    exercise: str
    name: str
    video: str
    cache: str
    output: str
    frames: int
    detected_frames: int
    missing_frames: int
    visibility: float
    mean_conf: float
    min_conf: float
    max_conf: float
    visible_segments: int
    longest_missing_run: int


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


def _safe_stem(value: str) -> str:
    value = Path(value).stem
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")
    return value or "video"


def _resolve_path(raw: str | Path, *, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return base / path


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _eligible_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row.get("error")
        and row.get("out")
        and row.get("video")
        and row.get("frames")
        and "detection_rate" in row
    ]


def select_rows(rows: Sequence[dict[str, Any]], selection: str) -> list[tuple[str, dict[str, Any]]]:
    """Return tagged rows to visualize.

    Selection criterion:
    - ``visibility`` is the cached detection rate, i.e.
      detected barbell frames / total frames.
    - Worst = lowest visibility; ties use lower mean confidence.
    - Best = highest visibility; ties use higher mean confidence.

    ``best`` breaks ties by higher mean confidence, because many videos have
    perfect detection coverage.
    """

    candidates = _eligible_rows(rows)
    if not candidates:
        raise RuntimeError("No visualizable rows found in detection summary.")

    worst = min(candidates, key=lambda row: (float(row["detection_rate"]), float(row.get("mean_conf", 0.0))))
    best = max(candidates, key=lambda row: (float(row["detection_rate"]), float(row.get("mean_conf", 0.0))))

    if selection == "worst":
        return [("worst", worst)]
    if selection == "best":
        return [("best", best)]
    if selection == "best-worst":
        if best is worst:
            return [("best_worst", best)]
        return [("worst", worst), ("best", best)]
    if selection == "per-exercise":
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            grouped.setdefault(str(row.get("type", "unknown")), []).append(row)

        selected: list[tuple[str, dict[str, Any]]] = []
        for exercise in sorted(grouped):
            group_rows = grouped[exercise]
            group_worst = min(
                group_rows,
                key=lambda row: (float(row["detection_rate"]), float(row.get("mean_conf", 0.0))),
            )
            group_best = max(
                group_rows,
                key=lambda row: (float(row["detection_rate"]), float(row.get("mean_conf", 0.0))),
            )
            if group_best is group_worst:
                selected.append(("best_worst", group_best))
            else:
                selected.extend([("worst", group_worst), ("best", group_best)])
        return selected
    raise ValueError(f"Unsupported selection: {selection}")


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Barbell cache not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _as_bool_vector(value: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=bool).reshape(-1)
    if len(arr) >= length:
        return arr[:length]
    out = np.zeros((length,), dtype=bool)
    out[: len(arr)] = arr
    return out


def _as_float_matrix(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    arr = arr.reshape(-1, shape[1]) if arr.size else np.empty((0, shape[1]), dtype=np.float32)
    if len(arr) >= shape[0]:
        return arr[: shape[0]]
    out = np.full(shape, np.nan, dtype=np.float32)
    out[: len(arr)] = arr
    return out


def _count_true_segments(mask: np.ndarray) -> int:
    if len(mask) == 0:
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(starts.sum())


def _longest_false_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def _draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.58,
    thickness: int = 1,
) -> None:
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_center_cross(frame: np.ndarray, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = point
    cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2, line_type=cv2.LINE_AA)


def render_overlay(
    *,
    row: dict[str, Any],
    tag: str,
    cache_path: Path,
    video_path: Path,
    out_dir: Path,
    overwrite: bool = False,
    codec: str = "mp4v",
) -> VideoVisualization:
    cache = _load_cache(cache_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or int(row.get("frames", 0) or 0))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or np.asarray(cache.get("fps", 30.0)).item() or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or np.asarray(cache.get("width", 0)).item() or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or np.asarray(cache.get("height", 0)).item() or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {video_path}: {width}x{height}")

    boxes = _as_float_matrix(cache.get("boxes", np.empty((0, 4), dtype=np.float32)), (frame_count, 4))
    centers = _as_float_matrix(cache.get("centers", np.empty((0, 2), dtype=np.float32)), (frame_count, 2))
    detected = _as_bool_vector(cache.get("detected", np.zeros((0,), dtype=bool)), frame_count)
    conf = np.asarray(cache.get("detection_conf", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if len(conf) < frame_count:
        padded = np.zeros((frame_count,), dtype=np.float32)
        padded[: len(conf)] = conf
        conf = padded
    else:
        conf = conf[:frame_count]

    visibility = float(detected.mean()) if frame_count else 0.0
    detected_frames = int(detected.sum())
    missing_frames = int(frame_count - detected_frames)
    visible_conf = conf[detected]
    mean_conf = float(visible_conf.mean()) if len(visible_conf) else 0.0
    min_conf = float(visible_conf.min()) if len(visible_conf) else 0.0
    max_conf = float(visible_conf.max()) if len(visible_conf) else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{tag}_{row.get('type', 'unknown')}_{_safe_stem(str(row.get('name', video_path.name)))}_vis{visibility:.3f}.mp4"
    if output.exists() and not overwrite:
        cap.release()
        return VideoVisualization(
            selection=tag,
            exercise=str(row.get("type", "")),
            name=str(row.get("name", video_path.name)),
            video=str(video_path),
            cache=str(cache_path),
            output=str(output),
            frames=frame_count,
            detected_frames=detected_frames,
            missing_frames=missing_frames,
            visibility=visibility,
            mean_conf=mean_conf,
            min_conf=min_conf,
            max_conf=max_conf,
            visible_segments=_count_true_segments(detected),
            longest_missing_run=_longest_false_run(detected),
        )

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= frame_count:
            break

        is_visible = bool(detected[idx])
        box = boxes[idx]
        score = float(conf[idx]) if idx < len(conf) else 0.0

        if is_visible and np.isfinite(box).all():
            x1 = int(np.clip(box[0], 0.0, 1.0) * width)
            y1 = int(np.clip(box[1], 0.0, 1.0) * height)
            x2 = int(np.clip(box[2], 0.0, 1.0) * width)
            y2 = int(np.clip(box[3], 0.0, 1.0) * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 220, 45), 3)
            _draw_text(frame, f"barbell conf={score:.3f}", (max(x1, 4), max(y1 - 8, 22)), color=(80, 255, 80))
            _draw_center_cross(frame, ((x1 + x2) // 2, (y1 + y2) // 2), (45, 220, 45))
        else:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 4)
            _draw_text(frame, "barbell MISSING (no cached box)", (18, 82), color=(0, 80, 255), scale=0.7, thickness=2)
            center = centers[idx]
            if np.isfinite(center).all():
                cx = int(np.clip(center[0], 0.0, 1.0) * width)
                cy = int(np.clip(center[1], 0.0, 1.0) * height)
                _draw_center_cross(frame, (cx, cy), (0, 220, 255))
                _draw_text(frame, "interpolated center", (max(cx - 80, 4), min(cy + 24, height - 8)), color=(0, 220, 255))

        _draw_text(
            frame,
            f"{tag.upper()} | {row.get('type', '')}/{row.get('name', video_path.name)}",
            (18, 26),
            color=(255, 255, 255),
            scale=0.62,
            thickness=2,
        )
        _draw_text(
            frame,
            f"frame {idx + 1}/{frame_count} | visible {visibility * 100:.2f}% "
            f"({detected_frames}/{frame_count}) | missing {missing_frames}",
            (18, 54),
            color=(255, 255, 255),
            scale=0.58,
        )

        progress_x = int(width * ((idx + 1) / max(frame_count, 1)))
        cv2.rectangle(frame, (0, height - 8), (width, height - 1), (35, 35, 35), -1)
        cv2.rectangle(frame, (0, height - 8), (progress_x, height - 1), (45, 220, 45) if is_visible else (0, 0, 255), -1)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    return VideoVisualization(
        selection=tag,
        exercise=str(row.get("type", "")),
        name=str(row.get("name", video_path.name)),
        video=str(video_path),
        cache=str(cache_path),
        output=str(output),
        frames=idx,
        detected_frames=int(detected[:idx].sum()) if idx else detected_frames,
        missing_frames=int(idx - detected[:idx].sum()) if idx else missing_frames,
        visibility=float(detected[:idx].mean()) if idx else visibility,
        mean_conf=mean_conf,
        min_conf=min_conf,
        max_conf=max_conf,
        visible_segments=_count_true_segments(detected[:idx]),
        longest_missing_run=_longest_false_run(detected[:idx]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw cached YOLO-World barbell boxes on best/worst source videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None, help="JSON path for visualization metadata.")
    parser.add_argument(
        "--selection",
        choices=("best-worst", "best", "worst", "per-exercise"),
        default="best-worst",
        help=(
            "best-worst selects global best/worst; per-exercise saves best/worst "
            "inside each exercise type. Criterion is visibility=detection_rate; "
            "ties are resolved by mean_conf."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary.resolve()
    base = Path.cwd()
    summary = _load_summary(summary_path)
    manifest_path = args.manifest or (args.out_dir / "barbell_visualization_manifest.json")

    outputs: list[VideoVisualization] = []
    for tag, row in select_rows(summary.get("rows", []), args.selection):
        video_path = _resolve_path(row["video"], base=base)
        cache_path = _resolve_path(row["out"], base=base)
        outputs.append(
            render_overlay(
                row=row,
                tag=tag,
                cache_path=cache_path,
                video_path=video_path,
                out_dir=args.out_dir,
                overwrite=args.overwrite,
                codec=args.codec,
            )
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "summary": str(summary_path),
        "selection": args.selection,
        "selection_criteria": {
            "visibility": "detected_frames / frames",
            "worst": "lowest visibility; tie-breaker lower mean_conf",
            "best": "highest visibility; tie-breaker higher mean_conf",
        },
        "outputs": [asdict(item) for item in outputs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(f"Saved {len(outputs)} visualization video(s)")
    for item in outputs:
        print(
            f"- {item.selection}: {item.exercise}/{item.name} -> {item.output} "
            f"visibility={item.visibility:.4f} detected={item.detected_frames}/{item.frames}"
        )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
