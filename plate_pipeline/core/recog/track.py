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


from core.config import TrackCfg


@dataclass
class PlateState:
    box: tuple
    src: str                     # det | det_low | klt | rim | lost (rows may become klt_drift)
    n_pts: int
    score: float = 1.0
    drift: bool = False

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
        self.time = None
        self.lost_since = None
        self.selected = None

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

    def step(self, gray, det=None, *, dets=None, t=None) -> Optional[PlateState]:
        c = self.cfg
        now = (self.time + 1 / 30 if self.time is not None else 0.0) if t is None else t
        dt = max(0.0, now - self.time) if self.time is not None else 0.0
        self.time = now
        candidates = list(dets) if dets is not None else ([det] if det is not None else [])
        high = [d for d in candidates if d.score >= c.score_high]
        low = [d for d in candidates if c.score_low <= d.score < c.score_high]
        self.selected = None
        if self.box is None:
            if not high:
                return None
            det = max(high, key=lambda d: d.score * d.size[0] * d.size[1])
            self.selected = det
            state = self.start(gray, det.box)
            state.src, state.score = getattr(det, "src", "det"), det.score
            return state
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
                    s = float(np.clip(np.hypot(M[0, 0], M[1, 0]),
                                      max(0.0, 1 - c.scale_rate_s * dt), 1 + c.scale_rate_s * dt))
                    cx, cy = (np.array(self.box[:2]) + np.array(self.box[2:])) / 2
                    ncx, ncy = M @ np.array([cx, cy, 1.0])
                    w, h = (self.box[2] - self.box[0]) * s, (self.box[3] - self.box[1]) * s
                    moved = (ncx - w / 2, ncy - h / 2, ncx + w / 2, ncy + h / 2)
                    self.pts = b[inl].reshape(-1, 1, 2).astype(np.float32)
        self.prev = gray
        ref = moved or self.box
        pc = (np.array(ref[:2]) + np.array(ref[2:])) / 2
        size = max(ref[2] - ref[0], ref[3] - ref[1])
        recovering = self.lost_since is not None
        corridor = recovering and now - self.lost_since > c.lost_corridor_s
        def eligible(d):
            if corridor:
                last = self.box
                last_size = max(last[2] - last[0], last[3] - last[1])
                ratio = max(d.size) / last_size
                return (abs(d.center[0] - (last[0] + last[2]) / 2) <= c.gate * last_size
                        and c.reacquire_size_min <= ratio <= c.reacquire_size_max)
            return np.linalg.norm(np.array(d.center) - pc) <= c.gate * size
        det = None
        for group in (high, [] if recovering else low):
            accepted = [d for d in group if eligible(d)]
            if accepted:
                if corridor:
                    det = min(accepted, key=lambda d: abs(d.center[1] - (self.box[1] + self.box[3]) / 2))
                else:
                    # Preserve nearest-candidate selection, within the prediction gate.
                    cx, cy = (np.array(self.box[:2]) + np.array(self.box[2:])) / 2
                    det = min(accepted, key=lambda d: (d.center[0]-cx)**2 + (d.center[1]-cy)**2)
                break
        if det is not None:
            drift = (getattr(det, "src", "det") == "det" and moved is not None
                     and np.linalg.norm(np.array(det.center) - pc) > c.drift_fraction * size)
            self.selected = det
            self.box = tuple(map(float, det.box))
            self._seed(gray, self.box)
            self.lost_since = None
            src = getattr(det, "src", "det")
            if src == "det" and det.score < c.score_high:
                src = "det_low"
            return PlateState(self.box, src, len(self.pts), det.score, drift)
        if moved is None or n < c.min_pts // 2:
            if self.lost_since is None:
                self.lost_since = now
            self.pts = np.empty((0, 1, 2), np.float32)
            return PlateState(self.box, "lost", n, 0.0)
        self.lost_since = None
        self.box = moved
        if n < c.min_pts:
            self._seed(gray, self.box)
        return PlateState(self.box, "klt", n, 1.0)
