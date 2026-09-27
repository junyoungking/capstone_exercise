"""Extract YOLO-pose COCO-17 keypoints into the offline pose cache.

The module is import-safe when ``ultralytics`` is not installed.  The optional
dependency is imported only when extraction is actually requested.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

try:  # Import works both as `python -m model.extract_pose_yolo` and direct script.
    from model import train_ablation as ta
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    import train_ablation as ta  # type: ignore


DEFAULT_MODEL = "yolov8m-pose.pt"


def _load_yolo(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency optional in tests
        raise RuntimeError("YOLO-pose extraction requires `pip install ultralytics`.") from exc
    return YOLO(model_path)


def _best_person_index(confs: Optional[object], count: int) -> int:
    if count <= 0:
        return 0
    if confs is None:
        return 0
    sums = confs.sum(dim=1)
    return int(sums.argmax().item())


def extract_pose_yolo(
    video_path: Path | str,
    out_path: Path | str,
    *,
    model_path: str = DEFAULT_MODEL,
    model: Optional[object] = None,
    device: Optional[str] = None,
) -> Tuple[int, float]:
    """Extract one video's YOLO COCO-17 keypoints.

    Output npz keys match the MediaPipe cache contract: ``kpts``, ``fps``,
    ``num_frames``, and ``source``.  Coordinates are normalized to frame width
    and height, and confidence is stored in channel 2.
    """

    import cv2

    video_path = Path(video_path)
    out_path = Path(out_path)
    yolo = model if model is not None else _load_yolo(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"open fail: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    kpts_list = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        kwargs = {"verbose": False}
        if device:
            kwargs["device"] = device
        results = yolo(frame, **kwargs)
        arr = np.zeros((ta.YOLO_NUM_KPT, 3), dtype=np.float32)

        keypoints = results[0].keypoints if results and results[0].keypoints is not None else None
        if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) > 0:
            confs = keypoints.conf
            best = _best_person_index(confs, len(keypoints.xy))
            xy = keypoints.xy[best].cpu().numpy()
            conf = confs[best].cpu().numpy() if confs is not None else np.ones((ta.YOLO_NUM_KPT,), dtype=np.float32)
            arr[:, 0] = xy[:, 0] / max(width, 1)
            arr[:, 1] = xy[:, 1] / max(height, 1)
            arr[:, 2] = conf

        kpts_list.append(arr)

    cap.release()
    kpts = np.asarray(kpts_list, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, kpts=kpts, fps=fps, num_frames=len(kpts), source=str(video_path))
    return len(kpts), fps


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
        dst = ta.pose_out_path(typ, name, pose_backend=ta.POSE_BACKEND_YOLO, yolo_pose_dir=out_dir)
        if not src.exists():
            continue
        yield typ, name, src, dst
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract YOLO-pose COCO-17 keypoints for offline phase experiments.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics YOLO pose model path/name.")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. 0, cpu, cuda:0.")
    parser.add_argument("--type", choices=ta.CLASS_LIST, default=None, help="Optional exercise type filter.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of videos to process.")
    parser.add_argument("--video-dir", type=Path, default=ta.VIDEO_DIR)
    parser.add_argument("--label-dir", type=Path, default=ta.LABEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=ta.YOLO_POSE_DIR)
    parser.add_argument("--fail-log", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fail_log = args.fail_log or (ta.WORK_DIR / "yolo_extract_fail.txt")
    yolo = _load_yolo(args.model)
    targets = list(
        iter_labeled_videos(
            label_dir=args.label_dir,
            video_dir=args.video_dir,
            out_dir=args.out_dir,
            exercise_type=args.type,
            limit=args.limit,
        )
    )
    ok_count = 0
    fail_rows = []
    for typ, name, src, dst in tqdm(targets, desc="yolo-pose"):
        if dst.exists() and not args.overwrite:
            ok_count += 1
            continue
        try:
            extract_pose_yolo(src, dst, model_path=args.model, model=yolo, device=args.device)
            ok_count += 1
        except Exception as exc:  # pragma: no cover - video/model environment dependent
            fail_rows.append(f"{typ},{name},{src},{exc}")

    if fail_rows:
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(fail_rows) + "\n", encoding="utf-8")
    print(f"YOLO extraction complete: success={ok_count} failed={len(fail_rows)} fail_log={fail_log if fail_rows else 'n/a'}")
    return 1 if fail_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
