"""Create a visual MediaPipe-vs-YOLO pose overlay contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    from model import train_ablation as ta
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    import train_ablation as ta  # type: ignore


MEDIAPIPE_COLOR = (0, 220, 255)
YOLO_COLOR = (255, 120, 0)


def _sample_indices(total: int, frames: int) -> np.ndarray:
    if total <= 0:
        return np.zeros((0,), dtype=np.int64)
    frames = max(1, min(int(frames), total))
    return np.linspace(0, total - 1, frames, dtype=np.int64)


def _draw_pose(
    image: np.ndarray,
    kpts: np.ndarray,
    *,
    color: Tuple[int, int, int],
    edges: Iterable[Tuple[int, int]],
    min_conf: float = 0.05,
) -> None:
    import cv2

    h, w = image.shape[:2]
    pts = []
    for x, y, conf in kpts:
        if conf < min_conf or (x == 0 and y == 0):
            pts.append(None)
            continue
        pts.append((int(float(x) * w), int(float(y) * h)))
    for a, b in edges:
        if a >= len(pts) or b >= len(pts) or pts[a] is None or pts[b] is None:
            continue
        cv2.line(image, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        if pt is not None:
            cv2.circle(image, pt, 3, color, -1, cv2.LINE_AA)


def load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"open fail: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx}: {video_path}")
    return frame


def compare_pose_overlay(
    typ: str,
    name: str,
    *,
    video_dir: Path = ta.VIDEO_DIR,
    pose_dir: Path = ta.POSE_DIR,
    team_pose_dir: Path = ta.TEAM_POSE_DIR,
    yolo_pose_dir: Path = ta.YOLO_POSE_DIR,
    frames: int = 16,
    output: Path | None = None,
) -> Path:
    import cv2

    video_path = video_dir / typ / name
    mp_path = ta.pose_path(
        typ,
        name,
        pose_dir=pose_dir,
        team_pose_dir=team_pose_dir,
        pose_backend=ta.POSE_BACKEND_MEDIAPIPE,
    )
    yolo_path = ta.pose_path(typ, name, pose_backend=ta.POSE_BACKEND_YOLO, yolo_pose_dir=yolo_pose_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not mp_path.exists():
        raise FileNotFoundError(f"MediaPipe pose cache not found: {mp_path}")
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO pose cache not found: {yolo_path}")

    with np.load(mp_path) as mp_npz, np.load(yolo_path) as yolo_npz:
        mp_kpts = mp_npz["kpts"].astype(np.float32)
        yolo_kpts = yolo_npz["kpts"].astype(np.float32)
    total = min(len(mp_kpts), len(yolo_kpts))
    if total <= 0:
        raise ValueError("pose caches are empty")

    idxs = _sample_indices(total, frames)
    rendered = []
    for idx in idxs:
        frame = load_frame(video_path, int(idx))
        _draw_pose(frame, mp_kpts[int(idx)], color=MEDIAPIPE_COLOR, edges=ta.MP_EDGES)
        _draw_pose(frame, yolo_kpts[int(idx)], color=YOLO_COLOR, edges=ta.COCO_EDGES)
        cv2.putText(frame, f"frame {int(idx)}  MP=yellow  YOLO=blue", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        rendered.append(frame)

    cols = min(4, len(rendered))
    rows = int(np.ceil(len(rendered) / cols))
    h, w = rendered[0].shape[:2]
    thumb_w = 360
    thumb_h = max(1, int(h * (thumb_w / max(w, 1))))
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for i, frame in enumerate(rendered):
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        r, c = divmod(i, cols)
        sheet[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = thumb

    out = output or (ta.WORK_DIR / "pose_overlay" / f"{typ}_{Path(name).stem}_mediapipe_yolo.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay MediaPipe and YOLO pose caches for one labeled video.")
    parser.add_argument("--type", required=True, choices=ta.CLASS_LIST)
    parser.add_argument("--name", required=True, help="Video file name as it appears in labels/video directory.")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--video-dir", type=Path, default=ta.VIDEO_DIR)
    parser.add_argument("--pose-dir", type=Path, default=ta.POSE_DIR)
    parser.add_argument("--team-pose-dir", type=Path, default=ta.TEAM_POSE_DIR)
    parser.add_argument("--yolo-pose-dir", type=Path, default=ta.YOLO_POSE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = compare_pose_overlay(
        args.type,
        args.name,
        video_dir=args.video_dir,
        pose_dir=args.pose_dir,
        team_pose_dir=args.team_pose_dir,
        yolo_pose_dir=args.yolo_pose_dir,
        frames=args.frames,
        output=args.output,
    )
    print(f"saved overlay: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
