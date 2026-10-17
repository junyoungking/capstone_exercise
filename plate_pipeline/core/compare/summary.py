"""Set summary for comparisons (pipeline tab, "비교와 텍스트 피드백").

Only counted reps go in (partial, other-spot and failed attempts are already out). A rep with low
confidence or an empty value leaves that value out. n is counted per value.
Scatter s: second differences of three consecutive reps, s^2 = sum (x[i+1] - 2x[i] + x[i-1])^2 / (6m),
m = number of such triples. "Consecutive" means counted rep numbers i, i+1, i+2 all have the value; a
missing value or a low-confidence rep breaks the chain. No triple -> no scatter.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

VALUES = ("far_dist", "down", "far_pause", "up", "rom", "v_mean", "rest_pause")


@dataclass
class Series:
    name: str
    reps: List[int]             # counted rep numbers (1-based) that have the value
    vals: np.ndarray

    @property
    def n(self):
        return len(self.vals)

    def mean(self):
        return float(np.mean(self.vals)) if self.n else None

    def median(self):
        return float(np.median(self.vals)) if self.n else None

    def first(self, g):
        return float(np.mean(self.vals[:g])) if self.n >= g else None

    def last(self, g):
        return float(np.mean(self.vals[-g:])) if self.n >= g else None

    def scatter(self) -> Optional[float]:
        pos = {r: v for r, v in zip(self.reps, self.vals)}
        d2 = [pos[r + 1] - 2 * pos[r] + pos[r - 1] for r in self.reps if r - 1 in pos and r + 1 in pos]
        if not d2:
            return None
        d2 = np.asarray(d2)
        return float(np.sqrt(np.sum(d2 ** 2) / (6 * len(d2))))


@dataclass
class SetSummary:
    exercise: str
    fps: float
    n_reps: int
    series: Dict[str, Series] = field(default_factory=dict)

    def values(self, name):
        return self.series[name].vals if name in self.series else np.array([])

    def scatter(self, name):
        return self.series[name].scatter() if name in self.series else None

    def get(self, name) -> Series:
        return self.series.get(name, Series(name, [], np.array([])))


def summarize(exercise, fps, trips, vals, low=None) -> SetSummary:
    """trips: counted reps in order; vals: their RepValues; low: optional list of bools (low confidence).
    A rep whose far point sits next to a gap is low confidence (pipeline tab, 신뢰도 표)."""
    if low is None:
        low = [bool(getattr(t, "far_gap", False)) for t in trips]
    per = {k: ([], []) for k in VALUES}
    for i, (t, v) in enumerate(zip(trips, vals)):
        if low[i]:
            continue
        rest = None
        if i + 1 < len(vals) and v.back is not None and vals[i + 1].start is not None:
            rest = max(0.0, vals[i + 1].start - v.back)
        row = dict(far_dist=v.far_dist, down=v.down, far_pause=v.far_pause, up=v.up, rom=v.rom,
                   v_mean=v.v_mean, rest_pause=rest)
        for k in VALUES:
            if row[k] is not None:
                per[k][0].append(i + 1)
                per[k][1].append(float(row[k]))
    s = SetSummary(exercise, fps, len(trips))
    for k, (r, x) in per.items():
        s.series[k] = Series(k, r, np.asarray(x, float))
    return s


def from_values(exercise, fps, **series) -> SetSummary:
    """Build a summary straight from value lists (tests and the statistics check).
    Each keyword is a list of values for reps 1..n; None leaves a hole."""
    s = SetSummary(exercise, fps, max((len(v) for v in series.values()), default=0))
    for k, xs in series.items():
        reps = [i + 1 for i, x in enumerate(xs) if x is not None]
        s.series[k] = Series(k, reps, np.asarray([x for x in xs if x is not None], float))
    return s
