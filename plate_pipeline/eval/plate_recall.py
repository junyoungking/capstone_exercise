"""Plate detector on held-out frames (pipeline tab, 마지막 표 "원판 검출": recall >= 95% at IoU 0.5, bench
counted separately).

python -m eval.plate_recall --model models/plate_tiny.onnx \
    --ann datasets/plate/annotations/instances_test2017.json --images datasets/plate/test2017 [--out eval_out/recall]
Prints recall, precision, mean IoU of the hits overall and per exercise, and the detector speed; writes the
missed and extra boxes drawn on the frames to --out for a look.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from core.recog.plate import OnnxPlateDetector
from train.plate import coco


def evaluate(det, d, img_dir, thr=0.5, out=None):
    truth = coco.boxes_by_image(d)
    per = {}
    times = []
    for im in d["images"]:
        img = cv2.imread(os.path.join(img_dir, im["file_name"]))
        t0 = time.perf_counter()
        pred = [(p.box, p.score) for p in det(img)]
        times.append(time.perf_counter() - t0)
        tb = truth.get(im["id"], [])
        tp, fn, fp, ious = coco.match(tb, pred, thr)
        for key in ("all", im.get("exercise", "unknown")):
            s = per.setdefault(key, dict(truth=0, tp=0, fp=0, ious=[], frames=0))
            s["truth"] += len(tb); s["tp"] += tp; s["fp"] += fp; s["ious"] += ious; s["frames"] += 1
        if out and (fn or fp):
            os.makedirs(out, exist_ok=True)
            for b in tb:
                cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 200, 0), 3)
            for b, s_ in pred:
                cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), 2)
                cv2.putText(img, f"{s_:.2f}", (int(b[0]), int(b[1]) - 5), 0, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(out, im["file_name"]), img)
    res = {}
    for k, s in per.items():
        res[k] = dict(frames=s["frames"], plates=s["truth"], recall=s["tp"] / max(s["truth"], 1),
                      precision=s["tp"] / max(s["tp"] + s["fp"], 1),
                      mean_iou=float(np.mean(s["ious"])) if s["ious"] else None)
    res["speed_ms"] = 1000 * float(np.median(times)) if times else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    det = OnnxPlateDetector(a.model, score_min=a.score)
    res = evaluate(det, coco.load(a.ann), a.images, out=a.out)
    for k, r in res.items():
        if k == "speed_ms":
            continue
        flag = "OK" if r["recall"] >= 0.95 else "below 95%"
        print(f"{k:10s} frames {r['frames']:4d} plates {r['plates']:4d} recall {r['recall']:.3f} ({flag}) "
              f"precision {r['precision']:.3f} IoU {r['mean_iou'] if r['mean_iou'] is None else round(r['mean_iou'], 3)}")
    print(f"detector {res['speed_ms']:.1f} ms per frame (median, CPU)")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, "recall.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
