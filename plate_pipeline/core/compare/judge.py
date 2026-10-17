"""Comparison rules (pipeline tab, "비교와 텍스트 피드백"; 구현 설계 tab, "비교와 판정 규칙").

judged when |delta| > k * SE and |delta| >= min_change.
  in-set     : delta = mean of the last g values - mean of the first g; SE = s; k = table A (n)
  vs ref     : delta = my median - reference median
               reference with >= short_ref_n values: SE = sqrt(s_me^2/n_me + s_ref^2/n_ref), k = table B
               reference with 1-2 values: SE = s_me * sqrt(1/n_me + 1/n_ref), k = table C (n_me)
Scatter below the floor is raised to it (time: max(1/fps, floor_s); distance: floor_m).
In-set judges far_dist and down; vs ref judges down and far_pause (far_pause only with >= 3 values on
both sides). Distance, speed and raise time are shown but never judged against a reference.
"""
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional

from ..config import DEFAULT, CmpCfg

INSET_VALUES = ("far_dist", "down")
REF_VALUES = ("down", "far_pause")
DIST_VALUES = ("far_dist", "rom")

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Multipliers:
    def __init__(self, path=None):
        path = path or os.path.join(_HERE, DEFAULT.cmp.table)
        t = json.load(open(path))
        self.alpha = t["alpha"]
        self._inset = {int(k): v for k, v in t["inset"].items()}
        self._short = {int(k): v for k, v in t["short"].items()}
        self._ref = {tuple(int(x) for x in k.split(",")): v for k, v in t["ref"].items()}

    @staticmethod
    def _cap(d, n):
        hi = max(d)
        return d[min(n, hi)] if n >= min(d) else None

    def inset(self, n):
        return self._cap(self._inset, n)

    def short(self, n):
        return self._cap(self._short, n)

    def ref(self, n_me, n_ref):
        hi = max(a for a, _ in self._ref)
        return self._ref.get((min(n_me, hi), min(n_ref, hi)))


_K = None


def multipliers():
    global _K
    if _K is None:
        _K = Multipliers()
    return _K


@dataclass
class Compare:
    kind: str                         # "inset" | "vs_ref"
    value: str
    result: str                       # "different" | "same" | "skip"
    reason: Optional[str] = None      # skip reason
    side: Optional[str] = None        # no_scatter: "mine" | "ref"
    n_a: int = 0
    n_b: int = 0
    need: Optional[int] = None
    a: Optional[float] = None         # in-set: first-g mean; vs ref: my median
    b: Optional[float] = None         # in-set: last-g mean;  vs ref: reference median
    s_a: Optional[float] = None
    s_b: Optional[float] = None
    delta: Optional[float] = None     # in-set: b - a; vs ref: a - b (mine - reference)
    se: Optional[float] = None
    k: Optional[float] = None
    min_change: Optional[float] = None

    @property
    def ratio(self):
        if self.delta is None or not self.se or not self.k:
            return None
        return abs(self.delta) / (self.k * self.se)


def floor_for(value, fps, cfg: CmpCfg = DEFAULT.cmp):
    if value in DIST_VALUES:
        return cfg.floor_m
    return max(1.0 / fps if fps else 0.0, cfg.floor_s)


def min_change_for(value, cfg: CmpCfg = DEFAULT.cmp):
    return cfg.min_change_m if value in DIST_VALUES else cfg.min_change_s


def judged(delta, se, k, min_change):
    return abs(delta) > k * se and abs(delta) >= min_change


def _finish(c: Compare):
    c.result = "different" if judged(c.delta, c.se, c.k, c.min_change) else "same"
    return c


def compare_inset(sm, value, cfg: CmpCfg = DEFAULT.cmp, K: Multipliers = None) -> Compare:
    K = K or multipliers()
    se_ = sm.get(value)
    c = Compare("inset", value, "skip", n_a=se_.n)
    if se_.n < cfg.set_min_n:
        c.reason, c.need = "set_too_short", cfg.set_min_n
        return c
    s = se_.scatter()
    if s is None:
        c.reason, c.side = "no_scatter", "mine"
        return c
    g = cfg.set_group_n
    c.s_a = max(s, floor_for(value, sm.fps, cfg))
    c.a, c.b = se_.first(g), se_.last(g)
    c.delta = c.b - c.a
    c.se, c.k, c.min_change = c.s_a, K.inset(se_.n), min_change_for(value, cfg)
    return _finish(c)


def compare_ref(mine, ref, value, cfg: CmpCfg = DEFAULT.cmp, K: Multipliers = None) -> Compare:
    K = K or multipliers()
    a, b = mine.get(value), ref.get(value)
    c = Compare("vs_ref", value, "skip", n_a=a.n, n_b=b.n)
    if b.n == 0:                      # the reference has other values but not this one
        c.reason, c.need = "ref_too_short", (cfg.short_ref_n if value == "far_pause" else 1)
        return c
    short = b.n < cfg.short_ref_n
    if value == "far_pause" and short:
        c.reason, c.need = "ref_too_short", cfg.short_ref_n
        return c
    need = cfg.short_ref_user_min_n if short else cfg.ref_user_min_n
    if a.n < need:
        c.reason, c.need = "set_too_short", need
        return c
    sa = a.scatter()
    if sa is None:
        c.reason, c.side = "no_scatter", "mine"
        return c
    fl = floor_for(value, mine.fps, cfg)
    c.s_a = max(sa, fl)
    if short:
        c.se = c.s_a * math.sqrt(1 / a.n + 1 / b.n)
        c.k = K.short(a.n)
    else:
        sb = b.scatter()
        if sb is None:
            c.reason, c.side = "no_scatter", "ref"
            return c
        c.s_b = max(sb, floor_for(value, ref.fps, cfg))
        c.se = math.sqrt(c.s_a ** 2 / a.n + c.s_b ** 2 / b.n)
        c.k = K.ref(a.n, b.n)
    c.a, c.b = a.median(), b.median()
    c.delta = c.a - c.b
    c.min_change = min_change_for(value, cfg)
    return _finish(c)


def run_compare(mine, ref=None, cfg: CmpCfg = DEFAULT.cmp) -> List[Compare]:
    """All comparisons for one set: in-set always, vs reference when a reference set is given."""
    out = [compare_inset(mine, v, cfg) for v in INSET_VALUES]
    if ref is not None:
        if ref.exercise != mine.exercise:
            out.append(Compare("vs_ref", "*", "skip", reason="exercise_mismatch"))
        elif all(ref.get(v).n == 0 for v in REF_VALUES):
            out.append(Compare("vs_ref", "*", "skip", reason="no_ref_values"))
        else:
            out += [compare_ref(mine, ref, v, cfg) for v in REF_VALUES]
    return out
