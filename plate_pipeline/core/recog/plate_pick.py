"""Select one physical plate using only detections from the complete video.

Tracklets use the tracker's existing centre/size gates, with a bounded time gap
and one-to-one frame assignment. They do not infer a detection across a gap.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from core.config import TrackCfg
from core.geometry import box_touches_edge


@dataclass
class Candidate:
    items: list = field(default_factory=list)  # (sequence, presentation time, Detection)

    @property
    def first(self):
        return self.items[0]

    def first_high(self, cfg):
        return next((item for item in self.items if item[2].score >= cfg.score_high), None)

    def motion(self):
        if len(self.items) < 3:
            return 0.0
        ys = [d.center[1] for _, _, d in self.items]
        diameter = float(np.median([max(d.size) for _, _, d in self.items]))
        if diameter <= 0:
            return 0.0
        # Reject a single stray box before comparing to a rep-sized excursion.
        return float((np.percentile(ys, 95) - np.percentile(ys, 5)) / diameter)


def candidate_tracks(frames, cfg=TrackCfg()):
    """Link detections from (seq, timestamp, detections) without multiple assignment."""
    tracks, active = [], []
    for seq, t, dets in frames:
        dets = [d for d in dets if d.score >= cfg.score_low and min(d.size) > 0]
        pairs = []
        active = [i for i in active if t - tracks[i].items[-1][1] <= cfg.pick_gap_s]
        for i in active:
            tr = tracks[i]
            _, last_t, last = tr.items[-1]
            if t - last_t > cfg.pick_gap_s or t < last_t:
                continue
            size = max(last.size)
            for j, det in enumerate(dets):
                ratio = max(det.size) / size
                distance = math.dist(det.center, last.center)
                if (distance <= cfg.gate * size and
                        cfg.reacquire_size_min <= ratio <= cfg.reacquire_size_max):
                    pairs.append((distance / size, i, j))
        used_tracks, used_dets = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_tracks or j in used_dets:
                continue
            tracks[i].items.append((seq, t, dets[j]))
            used_tracks.add(i); used_dets.add(j)
        for j, det in enumerate(dets):
            if j not in used_dets:
                tracks.append(Candidate([(seq, t, det)]))
                active.append(len(tracks) - 1)
    return tracks


def tracked_motion(rows):
    """Legacy vertical range, using detected positions and sizes, not KLT drift.

    None is insufficient evidence, not proof that the original plate is still.
    """
    measured=[r['plate'] for r in rows if r.get('plate',{}).get('used')
              and r['plate'].get('src') in ('det','det_low')]
    centers=[];sizes=[]
    for plate in measured:
        center=plate.get('center');size=plate.get('size_px')
        if center is None or size is None: continue
        y=float(center[1]);d=max(map(float,size))
        if np.isfinite(y) and np.isfinite(d) and d>0:
            centers.append(y);sizes.append(d)
    if len(centers)<3: return None
    return float((np.percentile(centers,95)-np.percentile(centers,5))/np.median(sizes))


def safe_start(candidate, image_size, cfg=TrackCfg()):
    """Backtrack only to the last unclipped box after the nearest edge hit."""
    high=candidate.first_high(cfg)
    if high is None:  # experimental weak-only option
        return candidate.first
    index=next(i for i,item in enumerate(candidate.items) if item is high)
    start=high
    for item in reversed(candidate.items[:index]):
        if box_touches_edge(item[2].box,image_size): break
        start=item
    return start


def moving_alternatives(tracks, legacy, cfg=TrackCfg()):
    """Confirmed moving candidates in the original first-high/score-area order."""
    moving=[tr for tr in tracks if tr is not legacy and tr.first_high(cfg) is not None
            and tr.motion() >= cfg.pick_motion_diameters]
    def priority(tr):
        seq, _, det=tr.first_high(cfg)
        return (seq,-det.score*det.size[0]*det.size[1])
    return sorted(moving,key=priority)


def choose_plate(frames, cfg=TrackCfg(), legacy_rows=None):
    """Return selected/legacy candidates; both are None without confirmation.

    With legacy_rows, choose an alternative only if the actual original track
    has sufficient measured evidence of being still. Without legacy_rows this
    retains the Mc2 tracklet-only behavior for compatibility and comparison.
    """
    tracks = candidate_tracks(frames, cfg)
    confirmed = [tr for tr in tracks if tr.first_high(cfg) is not None]
    def priority(tr):
        seq, _, d = tr.first_high(cfg) or tr.first
        return (seq, -d.score * d.size[0] * d.size[1])
    legacy = min(confirmed, key=priority) if confirmed else None
    moving = [tr for tr in confirmed if tr.motion() >= cfg.pick_motion_diameters]
    if cfg.plate_pick_weak:
        # Research option: sustained low-only detections, never used by default.
        moving += [tr for tr in tracks if tr.first_high(cfg) is None and len(tr.items) >= 6
                   and tr.items[-1][1] - tr.first[1] >= 1.0
                   and tr.motion() >= cfg.pick_motion_diameters]
    observed = tracked_motion(legacy_rows) if legacy_rows is not None else None
    if legacy_rows is not None:
        moving = [tr for tr in moving if tr is not legacy]
    if legacy is None:
        chosen = min(moving, key=priority) if moving else None
    elif legacy_rows is not None and (observed is None or observed >= cfg.pick_motion_diameters):
        chosen = legacy
    elif (legacy_rows is None and legacy in moving) or not moving:
        chosen = legacy
    else:
        chosen = min(moving, key=priority)
    return chosen, legacy, tracks
