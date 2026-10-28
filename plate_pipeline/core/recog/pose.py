"""Optional pose observations in display pixels; never used by the plate signal.

SideLock, connections and fallback person scoring come from unified_feedback_v4.
MediaPipe is loaded only when an estimator is constructed. Frames are already
oriented by core.io.video: this module does not rotate them.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import DEFAULT, PoseCfg

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]
SIDE_IDS = {
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27, "heel": 29, "foot": 31, "index": 19, "thumb": 21},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28, "heel": 30, "foot": 32, "index": 20, "thumb": 22},
}


def _exercise(name):
    name = name.lower().strip().replace('_', '').replace('-', '')
    for canonical, aliases in [('benchpress', ('bench', 'benchpress', 'bp')),
                               ('squat', ('squat', 'sq')), ('deadlift', ('deadlift', 'dl'))]:
        if name in aliases:
            return canonical
    raise ValueError(f'Unknown exercise: {name}')


def _valid(lms):
    return isinstance(lms, np.ndarray) and lms.shape == (33, 4) and np.isfinite(lms).all()


class SideLock:
    """v4 visibility hysteresis, including its margin, hold and keypoint groups."""

    def __init__(self, exercise: str, margin: float = 0.25, hold: int = 6):
        self.exercise = _exercise(exercise)
        self.margin = margin
        self.hold = hold
        self.side: Optional[str] = None
        self.candidate: Optional[str] = None
        self.candidate_count = 0

    def _side_score(self, lms, side):
        ids = SIDE_IDS[side]
        if self.exercise == 'benchpress':
            names = ['shoulder', 'elbow', 'wrist', 'index', 'thumb']
        else:
            names = ['shoulder', 'hip', 'knee', 'ankle', 'foot', 'elbow', 'wrist']
        return float(sum(max(0.0, min(1.0, float(lms[ids[k], 3]))) for k in names))

    def update(self, lms):
        if not _valid(lms):
            return self.side or 'right'
        ls = self._side_score(lms, 'left')
        rs = self._side_score(lms, 'right')
        best = 'left' if ls >= rs else 'right'
        if self.side is None:
            self.side = best
            return self.side
        cur_score = ls if self.side == 'left' else rs
        alt_score = rs if self.side == 'left' else ls
        alt_side = 'right' if self.side == 'left' else 'left'
        if alt_score > cur_score + self.margin:
            if self.candidate == alt_side:
                self.candidate_count += 1
            else:
                self.candidate = alt_side
                self.candidate_count = 1
            if self.candidate_count >= self.hold:
                self.side = alt_side
                self.candidate = None
                self.candidate_count = 0
        else:
            self.candidate = None
            self.candidate_count = 0
        return self.side


def person_box(lms, vis_thr=0.3):
    """Bounds of all visible points, independent of foot visibility."""
    if not _valid(lms):
        return None
    visible = lms[:, 3] >= vis_thr
    if not visible.any():
        return None
    pts = lms[visible, :2]
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def feet_y(lms, vis_thr, frame_h):
    """Lowest visible ankle/heel/foot within the frame's lower 97% (or None)."""
    if not _valid(lms):
        return None
    feet = lms[[27, 28, 29, 30, 31, 32]]
    good = (feet[:,3] >= vis_thr) & (feet[:,1] >= 0) & (feet[:,1] <= .97*frame_h)
    return float(feet[good,1].max()) if good.any() else None


def near_wrist(kp, plate_box, vis_thr=.5, frame_h=None, *, frame_w=None):
    """Nearest visible wrist to the plate center; kp is 33x[x,y,visibility].

    frame_h is required when selecting: observations above/below the displayed
    frame are rejected. Supply frame_w to also reject wrists beyond the right
    edge; frame_h alone cannot establish the image width.
    """
    if kp is None or plate_box is None or frame_h is None:
        return None
    points = np.asarray(kp, float)
    if points.shape != (33,3):
        return None
    wrists = points[[15,16]]
    good = (np.isfinite(wrists).all(axis=1) & (wrists[:,2] >= vis_thr)
            & (wrists[:,0] >= 0) & (wrists[:,1] >= 0) & (wrists[:,1] <= frame_h))
    if frame_w is not None:
        good &= wrists[:,0] <= frame_w
    if not good.any():
        return None
    center = (np.asarray(plate_box[:2])+np.asarray(plate_box[2:]))/2
    choices = wrists[good,:2]
    selected = choices[np.argmin(np.linalg.norm(choices-center,axis=1))]
    return tuple(map(float,selected))


def person_center(lms, vis_thr=0.3):
    """v4 torso center, in the same coordinates as lms."""
    ids = np.array([11, 12, 23, 24])
    ids = ids[lms[ids, 3] >= vis_thr]
    return lms[ids, :2].mean(axis=0) if len(ids) >= 2 else None


def _fallback_score(lms, prev_center):
    # lms and prev_center are normalized here, matching v4's units and weights.
    def mid(a, b):
        if lms[a, 3] >= .45 and lms[b, 3] >= .45:
            return (lms[a, :2] + lms[b, :2]) / 2
        return lms[a, :2] if lms[a, 3] >= lms[b, 3] else lms[b, :2]
    v = mid(11, 12) - mid(23, 24)
    length = float(np.hypot(*v))
    horizontal = abs(float(v[0])) / length if length >= 1e-6 else 0.0
    ids = np.array([0, 11, 12, 23, 24, 25, 26, 27, 28])
    pts = lms[ids[lms[ids, 3] >= .30], :2]
    size = float(np.hypot(*(pts.max(axis=0)-pts.min(axis=0)))) if len(pts) >= 3 else 0.0
    center = person_center(lms)
    continuity = 0.0
    if center is not None and prev_center is not None:
        continuity = 1.5 * max(0.0, 1.0-float(np.hypot(*(center-prev_center)))/.5)
    return horizontal, size + continuity


def select_lifter(cands, plate_box, prev_center, exercise, *, cfg=DEFAULT.pose, image_size=None):
    """Return an index, or None if no person qualifies near the current plate.

    image_size=(width,height) preserves v4 normalized fallback scoring for pixel
    inputs. Without it, use the visible coordinate extent (at least 1 per axis),
    which also accepts normalized inputs. Plate gating uses original coordinates.
    With a plate, rank by visible in-frame wrist distance, falling back to the
    visible person box only when neither wrist qualifies. Break distance ties
    by previous torso center, then candidate index. Without a plate the original
    normalized fallback scoring is unchanged.
    """
    boxes = [(i, person_box(p, cfg.vis_thr)) for i, p in enumerate(cands)]
    boxes = [(i, b) for i, b in boxes if b is not None]
    # The legacy MediaPipe comparison retains M0 selection exactly.
    if cfg.backend == 'rtmpose' and _exercise(exercise) == 'benchpress':
        reclining = []
        for i, box in boxes:
            p = cands[i]
            if np.all(p[[11, 12, 23, 24], 3] >= .3):
                v = p[[11, 12], :2].mean(axis=0) - p[[23, 24], :2].mean(axis=0)
                if abs(v[0]) > abs(v[1]):  # >45 degrees from vertical, in pixels
                    reclining.append((i, box))
        if reclining:
            boxes = reclining
    if plate_box is not None:
        x0,y0,x1,y1 = plate_box
        h = y1-y0
        if h <= 0:
            return None
        cx,cy = (x0+x1)/2, (y0+y1)/2
        # S3a stage0: pooled single-candidate box p99=1.015149, wrist
        # p99=3.407489 plate heights across all 10 videos. Ceil(max(...))=4.
        # One shared limit for S1 and feedback; no per-exercise/video tuning.
        limit = 4.0
        frame_w, frame_h = image_size if image_size is not None else (float('inf'), float('inf'))
        ranked = []
        for i, (a,b,c,d) in boxes:
            wrist = near_wrist(cands[i][:, [0, 1, 3]], plate_box, vis_thr=.5,
                               frame_h=frame_h, frame_w=frame_w)
            distance = (float(np.hypot(wrist[0]-cx, wrist[1]-cy)) / h if wrist is not None
                        else float(np.hypot(max(a-cx, 0, cx-c), max(b-cy, 0, cy-d))) / h)
            if distance > limit:
                continue
            center = person_center(cands[i], cfg.vis_thr)
            prev_dist = float(np.linalg.norm(center-prev_center)) if center is not None and prev_center is not None else float('inf')
            ranked.append(((distance, prev_dist), i))
        if not ranked:
            return None
        best = min(ranked)
        if cfg.backend == 'rtmpose' and prev_center is not None and best[0][1] > h:
            incumbents = [r for r in ranked if r[0][1] <= h]
            if incumbents:
                incumbent = min(incumbents, key=lambda r: (r[0][1], r[0][0], r[1]))
                new_d, old_d = best[0][0], incumbent[0][0]
                # A switch needs a visible wrist on BOTH people; box fallback
                # cannot establish a clearly closer grip.
                wrists = [near_wrist(cands[r[1]][:, [0, 1, 3]], plate_box,
                                     frame_h=frame_h, frame_w=frame_w)
                          for r in (best, incumbent)]
                if (any(w is None for w in wrists) or old_d-new_d < cfg.switch_margin
                        or new_d > .75*old_d):
                    best = incumbent
            else:
                # Previous lifter is not observed: cannot prove a closer wrist.
                return None
        return best[1]
    if not boxes:
        return None
    if image_size is None:
        points = np.concatenate([cands[i][cands[i][:,3] >= cfg.vis_thr,:2] for i,_ in boxes])
        scale = np.maximum(1.0, points.max(axis=0))
    else:
        scale = np.asarray(image_size, float)
    prev = np.asarray(prev_center)/scale if prev_center is not None else None
    ranked = []
    for i, _ in boxes:
        normalized = cands[i].copy()
        normalized[:, :2] /= scale
        horizontal, score = _fallback_score(normalized, prev)
        if _exercise(exercise) == 'benchpress':
            score += 3.0 * horizontal
        ranked.append((score, -i))
    return -max(ranked)[1] if ranked else None


class MediaPipeEstimator:
    def __init__(self, model_path=None, cfg: PoseCfg = DEFAULT.pose):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.cfg = cfg
        self.every_n = cfg.every_n
        path = Path(model_path or cfg.model_path or Path(__file__).resolve().parents[3] / 'pose_landmarker_lite.task')
        if not path.is_file():
            raise FileNotFoundError(f'PoseLandmarker model not found: {path}')
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=cfg.num_poses,
            min_pose_detection_confidence=cfg.min_conf,
            min_pose_presence_confidence=cfg.min_conf,
            min_tracking_confidence=cfg.min_conf, output_segmentation_masks=False)
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mp = mp
        self._last_ms = -1

    def __call__(self, image_bgr, t_s):
        if not np.isfinite(t_s) or t_s < 0:
            raise ValueError('Pose timestamp must be finite and nonnegative')
        timestamp = max(self._last_ms+1, int(round(t_s*1000)))
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp)
        self._last_ms = timestamp
        h,w = image_bgr.shape[:2]
        return [np.array([[p.x*w, p.y*h, p.z, p.visibility] for p in pose], dtype=float)
                for pose in result.pose_landmarks]

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Halpe26: COCO17 + head/neck/hip + L/R big toe, small toe, heel.
HALPE_TO_MP = {0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
               11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
               23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
               29: 24, 30: 25, 31: 20, 32: 21}


def score_visibility(scores, reference=.6):
    """Support score, not probability: .3/.45/.5 map to .18/.27/.30.

    .30 is our fixed localization support gate; scale that to the
    feedback's .5 observation gate. No claim of MP occlusion calibration.
    """
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('score_reference must be positive and finite')
    scores = np.asarray(scores, dtype=float)
    return np.where(np.isfinite(scores), np.clip(scores/reference, 0, 1), 0.)


def halpe_to_mp33(points, scores, reference=.6):
    points, scores = np.asarray(points), np.asarray(scores)
    if points.shape != (26, 2) or scores.shape != (26,):
        raise ValueError('Expected Halpe26 points (26,2) and scores (26,)')
    result = np.zeros((33, 4), dtype=float)
    visibility = score_visibility(scores, reference)
    for mp, halpe in HALPE_TO_MP.items():
        if np.isfinite(points[halpe]).all():
            result[mp, :2] = points[halpe]
            result[mp, 3] = visibility[halpe]
    # z=0 is an explicit 2D placeholder, never MediaPipe world coordinates.
    return result


class RTMPoseEstimator:
    """BodyWithFeet detector -> all person crops -> Halpe26, ORT CPU only."""

    def __init__(self, model_path=None, cfg=DEFAULT.pose):
        from rtmlib import BodyWithFeet
        if cfg.mode not in ('lightweight', 'balanced'):
            raise ValueError('RTMPose mode must be lightweight or balanced')
        if model_path or cfg.model_path:
            raise ValueError('Use rtmlib BodyWithFeet mode models, not a MediaPipe task')
        self.cfg, self.every_n = cfg, cfg.every_n
        self.model = BodyWithFeet(mode=cfg.mode, to_openpose=False,
                                  backend='onnxruntime', device='cpu')
        self.timings = dict(person_detection=0., pose=0.)
        self.scores = []

    def __call__(self, image_bgr, t_s):
        import time
        if not np.isfinite(t_s) or t_s < 0:
            raise ValueError('Pose timestamp must be finite and nonnegative')
        start = time.perf_counter()
        boxes = self.model.det_model(image_bgr)
        detected = time.perf_counter()
        # RTMPose's empty bboxes default is a whole-image pose; skip it.
        if len(boxes):
            points, scores = self.model.pose_model(image_bgr, bboxes=boxes)
        else:
            points, scores = [], []
        self.timings['person_detection'] += detected-start
        self.timings['pose'] += time.perf_counter()-detected
        self.scores.extend(np.asarray(scores).reshape(-1).tolist())
        return [halpe_to_mp33(p, s, self.cfg.score_reference) for p, s in zip(points, scores)]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PoseEstimator(MediaPipeEstimator):
    """Select a backend explicitly; keep MediaPipe lazy and available for M0."""

    def __new__(cls, model_path=None, cfg=DEFAULT.pose):
        if cfg.backend == 'rtmpose':
            return RTMPoseEstimator(model_path=model_path, cfg=cfg)
        if cfg.backend != 'mediapipe':
            raise ValueError('Pose backend must be rtmpose or mediapipe')
        return super().__new__(cls)
