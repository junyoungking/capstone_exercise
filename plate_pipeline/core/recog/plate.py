"""Plate detectors. All return a list of Detection (box in display pixels, score 0-1).

OnnxPlateDetector : the trained one-class detector (YOLOX-Tiny exported to ONNX), later (train/plate).
BoxFileDetector   : boxes from a label file {"<seq>": [x0, y0, x1, y1], ...} - stands in for the detector
                    while there is no model (and gives the "truth" boxes for the recall test).
find_plate_circle : a first-frame helper: the largest dark circle (Hough). Only to place the first box when
                    nothing else is given; not a detector.
"""
import json
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple
    score: float
    src: str = "det"             # "det" (model or label box) | "rim" (refine_circle stand-in)

    @property
    def center(self):
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    @property
    def size(self):
        x0, y0, x1, y1 = self.box
        return (x1 - x0, y1 - y0)


class BoxFileDetector:
    def __init__(self, path):
        self.boxes = {int(k): v for k, v in json.load(open(path)).items()}

    def __call__(self, image, seq) -> List[Detection]:
        b = self.boxes.get(seq)
        return [Detection(tuple(b), 1.0)] if b else []


STRIDES = (8, 16, 32)


def letterbox(image, size):
    """YOLOX preprocessing: resize keeping the aspect ratio into a size x size canvas filled with 114,
    top-left aligned; BGR 0-255, CHW float32. Returns (input, ratio)."""
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    img = np.full((size, size, 3), 114, np.uint8)
    rs = cv2.resize(image, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_LINEAR)
    img[:rs.shape[0], :rs.shape[1]] = rs
    return img.transpose(2, 0, 1)[None].astype(np.float32), r


def yolox_grids(size):
    gs, ss = [], []
    for st in STRIDES:
        h = w = size // st
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        gs.append(np.stack((xv, yv), 2).reshape(-1, 2))
        ss.append(np.full((h * w, 1), st))
    return np.concatenate(gs).astype(np.float32), np.concatenate(ss).astype(np.float32)


def decode_yolox(out, size, ratio, score_min=0.3, nms_iou=0.45) -> List[Detection]:
    """Raw YOLOX head output (exported without decode_in_inference): rows of
    (dx, dy, log w, log h, objectness, class score) over the stride 8/16/32 grids, sigmoids already applied
    to the last two. Box centre = (d + grid) x stride, size = exp(log) x stride; back to image pixels / ratio."""
    grid, stride = yolox_grids(size)
    assert out.shape[0] == grid.shape[0], (out.shape, grid.shape)
    xy = (out[:, :2] + grid) * stride
    wh = np.exp(out[:, 2:4]) * stride
    score = out[:, 4] * out[:, 5:].max(1)
    keep = score >= score_min
    if not keep.any():
        return []
    b = np.concatenate([xy[keep] - wh[keep] / 2, xy[keep] + wh[keep] / 2], 1) / ratio
    s = score[keep]
    idx = cv2.dnn.NMSBoxes([[float(q[0]), float(q[1]), float(q[2] - q[0]), float(q[3] - q[1])] for q in b],
                           s.tolist(), score_min, nms_iou)
    idx = np.array(idx).reshape(-1)
    return [Detection(tuple(map(float, b[i])), float(s[i])) for i in idx]


class OnnxPlateDetector:
    """The trained one-class plate detector (train/plate: YOLOX-Tiny exported to ONNX)."""

    def __init__(self, path, size=416, score_min=0.3, nms_iou=0.45, providers=None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else size
        self.score_min, self.nms_iou = score_min, nms_iou

    def __call__(self, image, seq=None) -> List[Detection]:
        x, r = letterbox(image, self.size)
        out = self.sess.run(None, {self.inp: x})[0][0]
        return decode_yolox(out, self.size, r, self.score_min, self.nms_iou)


def find_plate_circle(image, min_frac=0.08, max_frac=0.35) -> Optional[Detection]:
    """Largest dark, round blob that HoughCircles finds; radius between min_frac and max_frac of the
    shorter image side. Returns its bounding box or None."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = 640.0 / max(g.shape)
    small = cv2.resize(g, None, fx=scale, fy=scale)
    small = cv2.medianBlur(small, 5)
    side = min(small.shape)
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.2, minDist=side * 0.2, param1=120, param2=40,
                               minRadius=int(side * min_frac), maxRadius=int(side * max_frac))
    if circles is None:
        return None
    best, best_v = None, None
    for cx, cy, r in circles[0]:
        m = np.zeros_like(small)
        cv2.circle(m, (int(cx), int(cy)), int(r * 0.9), 255, -1)
        dark = 255 - float(small[m > 0].mean())
        v = dark * r
        if best_v is None or v > best_v:
            best, best_v = (cx, cy, r), v
    cx, cy, r = (v / scale for v in best)
    return Detection((cx - r, cy - r, cx + r, cy + r), 0.5)


def refine_circle(gray, box, r_lo=0.8, r_hi=1.25, search=0.6, work_px=320):
    """Stand-in detector while there is no model: the plate rim near a predicted box.
    Looks for circles (Hough) in the box grown by `search` x size on each side, radius r_lo..r_hi x the
    predicted radius, and returns the one whose centre is closest to the prediction (None if none).
    The rim is used rather than the middle because the sleeve end (and anything stuck on it) sits in
    front of the plate and shifts against it with the viewing angle."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) / 2
    pad = r * (1 + 2 * search)
    H, W = gray.shape[:2]
    a0, b0 = int(max(0, cx - pad)), int(max(0, cy - pad))
    a1, b1 = int(min(W, cx + pad)), int(min(H, cy + pad))
    roi = gray[b0:b1, a0:a1]
    if roi.size == 0:
        return None
    k = work_px / max(roi.shape)
    small = cv2.medianBlur(cv2.resize(roi, None, fx=k, fy=k), 5)
    cs = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.0, minDist=r * k * 0.5, param1=100, param2=30,
                          minRadius=int(r * k * r_lo), maxRadius=int(r * k * r_hi) + 1)
    if cs is None:
        return None
    best = None
    for qx, qy, qr in cs[0]:
        X, Y, R = a0 + qx / k, b0 + qy / k, qr / k
        d = np.hypot(X - cx, Y - cy)
        if best is None or d < best[0]:
            best = (d, X, Y, R)
    _, X, Y, R = best
    return Detection((X - R, Y - R, X + R, Y + R), 0.5, "rim")
