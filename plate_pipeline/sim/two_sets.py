"""Two sets in one recording (근거·출처 tab, 그 밖의 시험: 세트 간격, 종목 변경).

Set 1 and set 2 are built with sim.gen.build and joined by an idle stretch so that the set gap
(last rep of set 1 back on the rest side -> first rep of set 2 leaves it) is `gap` seconds.
  same exercise : the bar stays where set 1 left it (rack or floor); set 2 starts from there.
  other exercise: somewhere in the idle stretch the lifter moves to the other bar. For trans_s seconds the
                  plate is not seen (other bar, other plates); the label switches to the new exercise when
                  that starts ("exact moment", as in the doc's test).
Labels are given for every frame, with or without a plate sample.
"""
from dataclasses import dataclass, replace
from typing import List

import numpy as np

from sim.gen import DT, Opts, TrueRep, build, observe


@dataclass
class TwoSets:
    exercises: tuple
    gap: float
    fps: float
    t: np.ndarray
    y: np.ndarray
    x: np.ndarray
    frame_t: np.ndarray
    labels: List[str]
    reps: tuple                     # (TrueRep list of set 1, of set 2)
    trans: tuple = None             # (start, end) of the unseen stretch, other exercise only


def _shift(r: TrueRep, dt):
    return replace(r, far_in=r.far_in + dt, far_out=r.far_out + dt, raise0=r.raise0 + dt, raise1=r.raise1 + dt,
                   t0=r.t0 + dt, t1=r.t1 + dt)


def make_two_sets(ex1, ex2, gap, rng, fps=30.0, noise=0.005, trans_s=(5.0, 15.0), tries=50):
    """Returns None if this gap cannot be built (the two sets' own lead and tail are longer than the gap)."""
    for _ in range(tries):
        t1, y1, x1, r1 = build(ex1, rng, Opts())
        t2, y2, x2, r2 = build(ex2, rng, Opts())
        if not r1 or not r2:
            continue
        idle = gap - (t1[-1] - r1[-1].t1) - r2[0].t0
        if idle < (2.0 if ex1 != ex2 else 0.0):
            continue
        break
    else:
        return None
    same = ex1 == ex2
    if same:
        y2 = y2 + (y1[-1] - y2[0])
        x2 = x2 + (x1[-1] - x2[0])
    n_idle = max(1, int(round(idle / DT)))
    ti = t1[-1] + DT * np.arange(1, n_idle + 1)
    yi, xi = np.full(n_idle, y1[-1]), np.full(n_idle, x1[-1])
    trans = None
    if not same:
        T = min(float(rng.uniform(*trans_s)), idle - 1.0)
        ts = t1[-1] + float(rng.uniform(0.5, idle - T - 0.5))
        trans = (ts, ts + T)
        after = ti >= ts + T / 2                 # the bar is somewhere else once the plate is out of sight
        yi[after], xi[after] = y2[0], x2[0]
    off = ti[-1] + DT
    t = np.concatenate([t1, ti, t2 + off])
    y = np.concatenate([y1, yi, y2])
    x = np.concatenate([x1, xi, x2])
    reps2 = [_shift(r, off) for r in r2]
    ft, ty, tx, ot, oy, ox = observe(rng, t, y, x, fps, noise, Opts(), list(r1) + reps2)
    if trans is not None:
        keep = (ot < trans[0]) | (ot >= trans[1])
        ot, oy, ox = ot[keep], oy[keep], ox[keep]
        labels = [ex1 if f < trans[0] else ex2 for f in ft]
    else:
        labels = [ex1] * len(ft)
    real_gap = reps2[0].t0 - r1[-1].t1
    return TwoSets((ex1, ex2), float(real_gap), fps, ot, oy, ox, ft, labels, (list(r1), reps2), trans)
