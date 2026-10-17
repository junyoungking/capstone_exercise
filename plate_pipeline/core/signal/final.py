"""Final calculation of the bar signal (pipeline tab, "신호와 rep 판정" table, 최종 계산 column).

clean_final(t, y, x, fps) takes the raw plate samples of one recording (times in s, heights and
horizontal positions in m, up is +; dropped frames are simply absent) and returns
  * the cleaned raw samples (outliers removed, nothing interpolated), used for the speed blanking rule
  * the signal the state machine runs on: per unbroken stretch, resampled every 1/fps and filtered
    with a zero-phase Butterworth. Gaps longer than cfg.interp_s are left open.
Same input -> same output (no randomness, no state).
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from ..config import DEFAULT, FinalCfg


@dataclass
class Signal:
    t: np.ndarray            # sample times (s)
    y: np.ndarray            # height (m, up +)
    x: np.ndarray            # horizontal position (m)
    gap_before: np.ndarray   # length of the gap before each sample (0 inside a stretch)

    def __len__(self):
        return len(self.t)


@dataclass
class FinalResult:
    raw_t: np.ndarray        # cleaned raw samples
    raw_y: np.ndarray
    raw_x: np.ndarray
    sig: Signal              # resampled + filtered, stretches concatenated


def _neighbours(t, v, side, n, win):
    """Values of up to n neighbours on one side within win seconds. Returns (vals, dts) with NaN where absent."""
    m = len(t)
    vals = np.full((m, n), np.nan)
    dts = np.full((m, n), np.nan)
    for k in range(1, n + 1):
        if side < 0:
            idx = np.arange(m) - k
            ok = idx >= 0
        else:
            idx = np.arange(m) + k
            ok = idx < m
        j = np.where(ok, idx, 0)
        dt = np.abs(t - t[j])
        ok &= dt <= win
        vals[ok, k - 1] = v[j[ok]]
        dts[ok, k - 1] = dt[ok]
    return vals, dts


def _bad(t, v, first_pass, cfg):
    lv, ld = _neighbours(t, v, -1, cfg.nb_n, cfg.nb_s)
    rv, rd = _neighbours(t, v, +1, cfg.nb_n, cfg.nb_s)
    nl = np.sum(~np.isnan(lv), axis=1)
    nr = np.sum(~np.isnan(rv), axis=1)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ml = np.nanmedian(lv, axis=1)
        mr = np.nanmedian(rv, axis=1)
        dl = np.nanmax(ld, axis=1)
        dr = np.nanmax(rd, axis=1)
    bad = np.zeros(len(t), bool)

    both = (nl > 0) & (nr > 0)
    dt = np.maximum(dl, dr)
    tol = cfg.out_add_m + cfg.out_acc * dt ** 2
    lo = np.minimum(ml, mr) - tol
    hi = np.maximum(ml, mr) + tol
    bad[both] = (v[both] < lo[both]) | (v[both] > hi[both])

    one = (nl > 0) ^ (nr > 0)
    if first_pass:
        bad[one] = True
    else:
        m1 = np.where(nl > 0, ml, mr)
        d1 = np.where(nl > 0, dl, dr)
        tol1 = cfg.edge_add_m + cfg.edge_speed * d1
        bad[one] = np.abs(v[one] - m1[one]) > tol1[one]

    none = (nl == 0) & (nr == 0)
    bad[none] = True  # an isolated sample cannot be checked
    return bad


def remove_outliers(t, y, x, cfg: FinalCfg = DEFAULT.final):
    """Three passes of the neighbour check. Vertical and horizontal are checked separately;
    a sample rejected by either is dropped from both. Returns a boolean keep mask."""
    keep = np.ones(len(t), bool)
    for p in range(cfg.passes):
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            break
        tt = t[idx]
        bad = _bad(tt, y[idx], p == 0, cfg) | _bad(tt, x[idx], p == 0, cfg)
        keep[idx[bad]] = False
    return keep


def _lowpass(v, fps, hz):
    b, a = butter(2, hz / (fps / 2.0))
    return filtfilt(b, a, v)


def resample_filter(t, y, x, fps, cfg: FinalCfg = DEFAULT.final):
    """Split at gaps > interp_s, resample each stretch every 1/fps (linear), filter stretches > filter_min_n."""
    if len(t) == 0:
        e = np.array([])
        return Signal(e, e, e, e)
    cut = np.flatnonzero(np.diff(t) > cfg.interp_s) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(t)]
    out_t, out_y, out_x, out_g = [], [], [], []
    prev_end_t = None
    for s, e in zip(starts, ends):
        ts, ys, xs = t[s:e], y[s:e], x[s:e]
        if len(ts) == 1:
            tg = ts.copy()
        else:
            n = int(np.floor((ts[-1] - ts[0]) * fps + 1e-9)) + 1
            tg = ts[0] + np.arange(n) / fps
        yg = np.interp(tg, ts, ys)
        xg = np.interp(tg, ts, xs)
        if len(tg) > cfg.filter_min_n:
            yg = _lowpass(yg, fps, cfg.filter_hz)
            xg = _lowpass(xg, fps, cfg.filter_hz)
        g = np.zeros(len(tg))
        if prev_end_t is not None:
            g[0] = tg[0] - prev_end_t
        prev_end_t = tg[-1]
        out_t.append(tg); out_y.append(yg); out_x.append(xg); out_g.append(g)
    return Signal(np.concatenate(out_t), np.concatenate(out_y), np.concatenate(out_x), np.concatenate(out_g))


def clean_final(t, y, x, fps, cfg: FinalCfg = DEFAULT.final) -> FinalResult:
    t = np.asarray(t, float); y = np.asarray(y, float); x = np.asarray(x, float)
    order = np.argsort(t, kind="stable")
    t, y, x = t[order], y[order], x[order]
    keep = remove_outliers(t, y, x, cfg)
    rt, ry, rx = t[keep], y[keep], x[keep]
    return FinalResult(rt, ry, rx, resample_filter(rt, ry, rx, fps, cfg))
