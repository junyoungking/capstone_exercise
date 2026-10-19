"""Pick frames from your own videos for labelling.

python -m train.plate.extract_frames --videos videos.csv --out data/own --per-video 15

videos.csv columns: path,session,exercise
  session  = recording day + place (e.g. 1021_gymA). Train / val / test are split by session.
  exercise = squat | bench | deadlift
Frames are spread over the video: the video is cut into --per-video equal parts and in each part the
frame that differs most from the frames already taken (so standing still is not picked many times).
Writes data/own/images/<stem>_<t>.jpg and data/own/frames.json (COCO, no boxes yet).
"""
import argparse
import csv
import os

import cv2
import numpy as np

from core.io import video
from train.plate import coco


def pick_times(path, n, probe_fps=4.0):
    """Times of n frames: one per equal part, the one with the biggest change from the previous pick."""
    small, ts = [], []
    step, nxt = 1.0 / probe_fps, 0.0
    for t, g in video.frames(path, gray=True):
        if t + 1e-6 < nxt:
            continue
        nxt = t + step
        small.append(cv2.resize(g, (48, 64)).astype(np.float32))
        ts.append(t)
    if not ts:
        return []
    parts = np.array_split(np.arange(len(ts)), min(n, len(ts)))
    out, last = [], None
    for idx in parts:
        if last is None:
            k = idx[len(idx) // 2]
        else:
            k = idx[int(np.argmax([np.abs(small[i] - last).mean() for i in idx]))]
        out.append(ts[k])
        last = small[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-video", type=int, default=15)
    a = ap.parse_args()
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    d = coco.empty()
    for row in csv.DictReader(open(a.videos)):
        path = row["path"]
        stem = os.path.splitext(os.path.basename(path))[0]
        want = pick_times(path, a.per_video)
        got = 0
        for t, img in video.frames(path):
            if not want:
                break
            if t + 1e-6 >= want[0]:
                name = f"{stem}_{t:07.2f}.jpg"
                cv2.imwrite(os.path.join(img_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                coco.add_image(d, name, img.shape[1], img.shape[0], session=row["session"],
                               exercise=row["exercise"], source="own", video=os.path.basename(path), t=round(t, 3))
                want.pop(0)
                got += 1
        print(f"{stem}: {got} frames")
    coco.save(d, os.path.join(a.out, "frames.json"))
    print(f"{len(d['images'])} frames -> {a.out}")


if __name__ == "__main__":
    main()
