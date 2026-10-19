"""Plate tracking between detections (pipeline tab: detection + KLT; 구현 설계 tab, recog/track.py).

KLT points inside the plate (a disc slightly smaller than the box) are tracked frame to frame with a
forward-backward check; a similarity transform (RANSAC) moves and scales the box. A detection close to the
prediction (centre within gate x plate size) replaces the box and re-seeds the points. Points are re-seeded
when fewer than min_pts survive. Too few points and no detection -> the plate is lost for that frame.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackCfg:
    max_pts: int = 150
    min_pts: int = 25
    inner: float = 0.85          # seed points inside this fraction of the plate radius ...
    hub: float = 0.45            # ... and outside this one (the sleeve end sits in front of the plate)
    fb_px: float = 1.0           # forward-backward error limit
    gate: float = 0.5            # detection accepted if its centre is within gate x plate size
    win: int = 21
    levels: int = 3


@dataclass
class PlateState:
    box: tuple
    src: str                     # "det" | "klt" | "lost"
    n_pts: int
    score: float = 1.0

    @property
    def center(self):
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    @property
    def size(self):
        x0, y0, x1, y1 = self.box
        return (x1 - x0, y1 - y0)


class PlateTracker:
    def __init__(self, cfg: TrackCfg = TrackCfg()):
        self.cfg = cfg
        self.prev = None
        self.pts = None
        self.box = None

    def _seed(self, gray, box):
        x0, y0, x1, y1 = box
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0) / 2
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(cx), int(cy)), max(2, int(r * self.cfg.inner)), 255, -1)
        cv2.circle(mask, (int(cx), int(cy)), int(r * self.cfg.hub), 0, -1)
        p = cv2.goodFeaturesToTrack(gray, self.cfg.max_pts, 0.01, 5, mask=mask)
        self.pts = p if p is not None else np.empty((0, 1, 2), np.float32)

    def start(self, gray, box):
        self.box = tuple(map(float, box))
        self.prev = gray
        self._seed(gray, self.box)
        return PlateState(self.box, "det", len(self.pts))

    def step(self, gray, det=None) -> Optional[PlateState]:
        c = self.cfg
        if self.box is None:
            if det is None:
                return None
            return self.start(gray, det.box)
        moved = None
        n = 0
        if len(self.pts):
            lk = dict(winSize=(c.win, c.win), maxLevel=c.levels,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None, **lk)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **lk)
            fb = np.linalg.norm((self.pts - p0r).reshape(-1, 2), axis=1)
            good = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < c.fb_px)
            a, b = self.pts[good].reshape(-1, 2), p1[good].reshape(-1, 2)
            if len(a) >= 3:
                M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is not None:
                    inl = inl.reshape(-1).astype(bool)
                    n = int(inl.sum())
                    s = float(np.hypot(M[0, 0], M[1, 0]))
                    cx, cy = (np.array(self.box[:2]) + np.array(self.box[2:])) / 2
                    ncx, ncy = M @ np.array([cx, cy, 1.0])
                    w, h = (self.box[2] - self.box[0]) * s, (self.box[3] - self.box[1]) * s
                    moved = (ncx - w / 2, ncy - h / 2, ncx + w / 2, ncy + h / 2)
                    self.pts = b[inl].reshape(-1, 1, 2).astype(np.float32)
        self.prev = gray
        if det is not None:
            ref = moved or self.box
            pc = ((ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2)
            size = max(ref[2] - ref[0], ref[3] - ref[1])
            if np.hypot(det.center[0] - pc[0], det.center[1] - pc[1]) <= c.gate * size:
                self.box = tuple(map(float, det.box))
                self._seed(gray, self.box)
                return PlateState(self.box, getattr(det, "src", "det"), len(self.pts), det.score)
        if moved is None or n < c.min_pts // 2:
            return PlateState(self.box, "lost", n, 0.0)
        self.box = moved
        if n < c.min_pts:
            self._seed(gray, self.box)
        return PlateState(self.box, "klt", n, 1.0)
