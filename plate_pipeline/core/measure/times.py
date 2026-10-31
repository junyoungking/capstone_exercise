"""Rep values from the final signal (pipeline tab: 평균 속도, 템포; 구현 설계 tab: 지표 계산).

Four time points per rep: start (leaves the rest side), far_in (arrives at the far point),
far_out (leaves it), back (arrives back on the rest side).
Each is found with the endpoint method: d = max(d_frac x ROM, d_min_m); with t1 the time the bar is d
away from the end level and t2 the time it is 4d away, the endpoint is 2*t1 - t2 (constant
acceleration). End levels are the mean height within +-end_avg_s of the extreme sample.
The raise ends where the bar comes back down by drop_m.
Speed (mean, m/s) = ROM / raise time. No speed when ROM < min_rom_m, a point is not found, or the
cleaned raw samples leave a gap >= blank_s inside the raise (+-end_avg_s).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import DEFAULT, SpeedCfg


@dataclass
class RepValues:
    start: Optional[float] = None
    far_in: Optional[float] = None
    far_out: Optional[float] = None
    back: Optional[float] = None
    rom: Optional[float] = None          # raised distance (m)
    far_dist: Optional[float] = None     # rest level before the rep -> far point (m)
    v_mean: Optional[float] = None
    down: Optional[float] = None
    up: Optional[float] = None
    far_pause: Optional[float] = None
    blank: str = ""                      # why the speed is empty


def _stretch(sig, i):
    """Index range [a, b) of the unbroken stretch containing sample i."""
    starts = np.flatnonzero(sig.gap_before > 0)
    a = starts[starts <= i].max() if np.any(starts <= i) else 0
    later = starts[starts > i]
    b = later.min() if len(later) else len(sig.t)
    return int(a), int(b)


def _level(t, y, i, half):
    m = np.abs(t - t[i]) <= half + 1e-9
    return float(np.mean(y[m]))


def _cross(t, y, i0, i1, level, step):
    """First crossing of `level` walking from i0 towards i1 (step +1 or -1).
    Returns (time by linear interpolation, index reached) or (None, None)."""
    prev = None
    for i in range(i0, i1, step):
        if prev is not None:
            a, b = y[prev] - level, y[i] - level
            if a == 0:
                return float(t[prev]), prev
            if a * b <= 0 and a != b:
                f = a / (a - b)
                return float(t[prev] + f * (t[i] - t[prev])), i
        prev = i
    return None, None


def _endpoint(t, y, anchor, stop, level, d, step):
    """Endpoint of a motion that ends (or starts) at `level`.
    Walk from `anchor` (a point in the middle of the motion) towards `stop` with `step`:
    first the crossing of level + 4d (t2), then, going on, the crossing of level + d (t1).
    d carries the sign of the side the motion is on. Endpoint = 2*t1 - t2 (constant acceleration)."""
    t2, i2 = _cross(t, y, anchor, stop, level + 4 * d, step)
    if t2 is None:
        return None
    t1, _ = _cross(t, y, i2 - step, stop, level + d, step)
    if t1 is None:
        return None
    return 2 * t1 - t2


def _mid(t, y, i0, i1, level, step):
    _, i = _cross(t, y, i0, i1, level, step)
    return i


def _raise_end(y, i0, b, drop):
    """From i0 walk forward keeping the running max; stop when the bar is drop below it."""
    best = i0
    for i in range(i0, b):
        if y[i] > y[best]:
            best = i
        elif y[best] - y[i] >= drop:
            break
    return best


def _raw_gap(raw_t, w0, w1):
    m = (raw_t >= w0) & (raw_t <= w1)
    pts = np.r_[w0, raw_t[m], w1]
    return float(np.max(np.diff(pts))) if len(pts) > 1 else w1 - w0


def rep_values(sig, raw_t, trip, rest, prev_end_t=None, next_entry_t=None, cfg: SpeedCfg = DEFAULT.speed):
    t, y = sig.t, sig.y
    rv = RepValues()
    if len(t) == 0:
        rv.blank = "no_signal"
        return rv
    i_far = int(np.argmin(np.abs(t - trip.far_t)))
    a, b = _stretch(sig, i_far)
    # rest level before the rep: after the previous rep was counted, at most 3 s back
    lo_t = trip.entry_t - 3.0 if prev_end_t is None else max(prev_end_t, trip.entry_t - 3.0)
    win = np.flatnonzero((t >= lo_t) & (t <= trip.end_t + 1e-9))
    win = win[(win >= a) & (win < b)]
    if len(win) == 0:
        rv.blank = "gap"
        return rv
    own = win[t[win] >= trip.entry_t - 1e-9]          # this rep's own excursion: the far point is here
    if len(own) == 0:
        rv.blank = "gap"
        return rv
    h = cfg.end_avg_s
    cut_pre = t[a] > lo_t + 1e-9
    if rest == "top":
        f = own[np.argmin(y[own])]
        pre = win[t[win] <= trip.entry_t + 1e-9]
        if len(pre) == 0:
            rv.blank = "gap" if cut_pre else "no_start"
            return rv
        p = pre[np.argmax(y[pre])]
        q = _raise_end(y, f, b, cfg.drop_m)
        bot, top_b, top_a = _level(t, y, f, h), _level(t, y, p, h), _level(t, y, q, h)
        rom = top_a - bot
        rv.rom, rv.far_dist = rom, top_b - bot
        d = max(cfg.d_frac * rom, cfg.d_min_m)
        m1 = _mid(t, y, f, a - 1, 0.5 * (top_b + bot), -1)     # middle of the descent
        m2 = _mid(t, y, f, b, 0.5 * (top_a + bot), +1)          # middle of the raise
        if m1 is not None:
            rv.start = _endpoint(t, y, m1, a - 1, top_b, -d, -1)
            rv.far_in = _endpoint(t, y, m1, f + 1, bot, +d, +1)
        if m2 is not None:
            rv.far_out = _endpoint(t, y, m2, a - 1, bot, +d, -1)
            rv.back = _endpoint(t, y, m2, b, top_a, -d, +1)
        # A transient rise followed by a >3 cm dip can make _raise_end pick
        # an early peak even though the FSM later observes a 70% return.
        # Retry a missing endpoint or that contradiction in the same stretch
        # and before the next rep; never invent a missing crossing.
        if cfg.recover_early_back and (rv.back is None or rv.back < trip.end_t) and not trip.far_gap:
            end_i = int(np.searchsorted(t, trip.end_t, side="left"))
            stop = min(b, int(np.searchsorted(t, next_entry_t, side="left"))) if next_entry_t is not None else b
            if a <= end_i < stop:
                q2 = _raise_end(y, end_i, stop, cfg.drop_m)
                top2 = _level(t, y, q2, h)
                rom2 = top2 - bot
                if rom2 >= cfg.min_rom_m:
                    d2 = max(cfg.d_frac * rom2, cfg.d_min_m)
                    mid2 = _mid(t, y, f, q2 + 1, 0.5 * (top2 + bot), +1)
                    if mid2 is not None:
                        out2 = _endpoint(t, y, mid2, a - 1, bot, +d2, -1)
                        back2 = _endpoint(t, y, mid2, stop, top2, -d2, +1)
                        if (out2 is not None and back2 is not None and out2 < back2
                                and trip.end_t <= back2 <= t[stop-1]):
                            rv.far_out, rv.back, rv.rom, rom = out2, back2, rom2, rom2
        r0, r1 = rv.far_out, rv.back
        if rv.start is not None and rv.far_in is not None:
            rv.down = rv.far_in - rv.start
    else:
        f = own[np.argmax(y[own])]
        pre = win[t[win] <= trip.entry_t + 1e-9]
        if len(pre) == 0:
            rv.blank = "gap" if cut_pre else "no_start"
            return rv
        p = pre[np.argmin(y[pre])]
        q = _raise_end(y, p, b, cfg.drop_m)
        flo, top = _level(t, y, p, h), _level(t, y, q, h)
        rom = top - flo
        rv.rom, rv.far_dist = rom, _level(t, y, f, h) - flo
        d = max(cfg.d_frac * rom, cfg.d_min_m)
        m1 = _mid(t, y, p, q + 1, 0.5 * (flo + top), +1)        # middle of the raise
        if m1 is not None:
            rv.start = _endpoint(t, y, m1, a - 1, flo, +d, -1)
            rv.far_in = _endpoint(t, y, m1, b, top, -d, +1)
        # back on the floor: lowest point after the top, before the next rep
        hi_t = next_entry_t if next_entry_t is not None else trip.end_t + 3.0
        post = np.flatnonzero((t >= t[q]) & (t <= hi_t))
        post = post[(post >= a) & (post < b)]
        if len(post):
            e = post[np.argmin(y[post])]
            flo_a = _level(t, y, e, h)
            m2 = _mid(t, y, q, e + 1, 0.5 * (top + flo_a), +1)  # middle of the lowering
            if m2 is not None:
                rv.far_out = _endpoint(t, y, m2, a - 1, top, -d, -1)
                rv.back = _endpoint(t, y, m2, b, flo_a, +d, +1)
        r0, r1 = rv.start, rv.far_in
        if rv.far_out is not None and rv.back is not None:
            rv.down = rv.back - rv.far_out
    if rv.far_in is not None and rv.far_out is not None:
        rv.far_pause = max(0.0, rv.far_out - rv.far_in)
    if r0 is not None and r1 is not None and r1 > r0:
        rv.up = r1 - r0
    cut = t[a] > lo_t + 1e-9 or (b < len(t) and t[b - 1] < trip.end_t + 0.5)
    if rom < cfg.min_rom_m:
        rv.blank = "gap" if cut else "small_rom"
    elif rv.up is None:
        rv.blank = "gap" if cut else "no_points"
    elif _raw_gap(raw_t, r0 - h, r1 + h) >= cfg.blank_s:
        rv.blank = "gap"
    else:
        rv.v_mean = rom / rv.up
    return rv


def all_rep_values(sig, raw_t, reps, rest, cfg: SpeedCfg = DEFAULT.speed):
    out = []
    for k, r in enumerate(reps):
        prev_end = reps[k - 1].end_t if k > 0 else None
        nxt = reps[k + 1].entry_t if k + 1 < len(reps) else None
        out.append(rep_values(sig, raw_t, r, rest, prev_end, nxt, cfg))
    return out
