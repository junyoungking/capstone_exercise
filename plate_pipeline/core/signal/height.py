"""Frame rows -> bar height and sideways position in metres (pipeline tab, 스케일; 구현 설계 tab height_signal).

Vertical/horizontal scale uses measured box height/width medians within +/-measured_half_window_s,
or the nearest measurement outside the window. With no measurements, use all sizes
with the legacy 0.5 s median. KLT size estimates do not set physical scale. The camera's own
movement (row["cam"], px) is subtracted. Height is integrated from frame-to-frame centre moves with the
mean scale of the two frames, so the result is relative (starts at 0), up is +.
"""
import numpy as np

from core.config import TrackCfg


def _rolling_median(t, v, win_s):
    out = np.empty_like(v)
    j0 = j1 = 0
    n = len(t)
    for i in range(n):
        while t[j0] < t[i] - win_s / 2:
            j0 += 1
        while j1 < n and t[j1] <= t[i] + win_s / 2:
            j1 += 1
        out[i] = np.median(v[j0:j1])
    return out


def height_signal(rows, plate_mm=450.0, win_s=0.5):
    """Returns t, y, x (m) for the frames with a plate in use."""
    use = [r for r in rows if r.get("plate") and r["plate"].get("used")]
    if not use:
        e = np.array([])
        return e, e, e
    t = np.array([r["t"] for r in use], float)
    c = np.array([r["plate"]["center"] for r in use], float)
    s = np.array([r["plate"]["size_px"] for r in use], float)
    cam = np.array([r.get("cam") or (0.0, 0.0) for r in use], float)
    c = c - cam
    plate_m = plate_mm / 1000.0
    measured = [r for r in use if r["plate"].get("src") in ("det", "det_low", "rim")]
    if measured:
        mt = np.array([r["t"] for r in measured])
        ms = np.array([r["plate"]["size_px"] for r in measured], float)
        half = TrackCfg().measured_half_window_s
        sizes = []
        for ti in t:
            lo = np.searchsorted(mt, ti-half, side="left")
            hi = np.searchsorted(mt, ti+half, side="right")
            sizes.append(np.median(ms[lo:hi], axis=0) if hi > lo else ms[np.argmin(abs(mt-ti))])
        scale = plate_m / np.asarray(sizes)
        sx, sy = scale[:, 0], scale[:, 1]
    else:
        sy = plate_m / _rolling_median(t, s[:, 1], win_s)
        sx = plate_m / _rolling_median(t, s[:, 0], win_s)
    dy = -np.diff(c[:, 1]) * (sy[1:] + sy[:-1]) / 2
    dx = np.diff(c[:, 0]) * (sx[1:] + sx[:-1]) / 2
    y = np.concatenate([[0.0], np.cumsum(dy)])
    x = np.concatenate([[0.0], np.cumsum(dx)])
    return t, y, x
