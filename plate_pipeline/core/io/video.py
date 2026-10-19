"""Video frames with presentation times (구현 설계 tab: PyAV, time comes from timestamps, not frame numbers).

Frames are returned in display orientation (the container's rotation is applied).
"""
from dataclasses import dataclass

import av
import numpy as np


@dataclass
class VideoInfo:
    width: int            # display orientation
    height: int
    fps: float            # average rate
    duration: float
    frames: int
    rotation: int


def _turns(rotation):
    """Clockwise quarter turns that bring the stored frame to display orientation."""
    return (-int(rotation or 0) // 90) % 4


def info(path) -> VideoInfo:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rot = 0
        for f in c.decode(s):
            rot = int(getattr(f, "rotation", 0) or 0)
            w, h = f.width, f.height
            break
        fps = float(s.average_rate) if s.average_rate else 0.0
        dur = float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / 1e6
        if _turns(rot) % 2:
            w, h = h, w
        return VideoInfo(w, h, fps, dur, int(s.frames or 0), rot)


def frames(path, gray=False, start_s=0.0, end_s=None):
    """Yields (t, image). image is BGR uint8 (or gray) in display orientation; t in seconds."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for f in c.decode(s):
            if f.time is None:
                continue
            t = float(f.time)
            if t < start_s:
                continue
            if end_s is not None and t > end_s:
                break
            img = f.to_ndarray(format="gray" if gray else "bgr24")
            k = _turns(getattr(f, "rotation", 0))
            if k:
                img = np.ascontiguousarray(np.rot90(img, -k))
            yield t, img
