"""Rep state machine (pipeline tab, "신호와 rep 판정"). The same rules run on the live and the final signal.

Direction is given: rest="top" (squat, bench: the bar rests high, the far point is the bottom) or
rest="bottom" (deadlift: the bar rests on the floor, the far point is the lockout).
s = distance from the reference point towards the far side (always >= 0 when leaving the rest side).

States
  wait : s < 0 (bar goes further to the rest side) -> move the reference there
         s >= entry_m -> progress
  prog : far point = largest s so far; A = far
         A < min_amp:  s <= (1 - ret) * A, or far point not updated for stall_s -> cancel, reference = here
         A >= min_amp: s <= (1 - ret) * A -> rep (dropped if it took < bounce_s since entry, unless the
                       entry sample came right after a gap > gap_s); reference = here
                       no return for fail_s -> failed attempt; reference = here
Gap rules
  gap > restart_s: restart at the current position unless an excursion >= min_amp is in progress
  far point right before/after a gap > gap_s -> rep.far_gap (depth unknown: partial/other-spot checks skipped)
Partial reps in the live mode are decided here (median of the reps counted so far); the final mode
decides them over the whole set in sets.py.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import DEFAULT, RepCfg


@dataclass
class Trip:
    kind: str               # "rep" | "failed"
    entry_t: float
    far_t: float
    far_y: float
    far_x: float
    amp: float              # far point distance from the reference (m)
    ref_y: float
    end_t: float            # time it was counted (return) or given up
    far_gap: bool = False
    entry_gap: bool = False
    flags: List[str] = field(default_factory=list)   # "partial", "other_spot", "rerack", "bounce"

    @property
    def counted(self):
        return self.kind == "rep" and not self.flags


class RepMachine:
    def __init__(self, rest: str, mode: str = "final", cfg: RepCfg = DEFAULT.rep):
        assert rest in ("top", "bottom")
        assert mode in ("final", "live")
        self.sign = -1.0 if rest == "top" else 1.0
        self.mode, self.cfg = mode, cfg
        self.trips: List[Trip] = []
        self.ref: Optional[float] = None
        self.phase = "wait"
        self.prev_t = None
        self._far_is_prev = False

    # distance from the reference towards the far side
    def _s(self, y):
        return self.sign * (y - self.ref)

    def _counted_amps(self):
        return [r.amp for r in self.trips if r.counted]

    def _counted_xs(self):
        return [r.far_x for r in self.trips if r.counted]

    def push(self, t, y, x, gap=0.0):
        c = self.cfg
        if self.ref is None:
            self.ref, self.prev_t = y, t
            return
        if gap > c.restart_s and not (self.phase == "prog" and self.far >= c.min_amp_m):
            self.phase = "wait"
            self.ref = y
            self.prev_t = t
            self._far_is_prev = False
            return
        if self.phase == "prog" and gap > c.gap_s and self._far_is_prev:
            self.far_gap = True           # the far sample is right before this gap
        s = self._s(y)
        self._far_is_prev = False
        if self.phase == "wait":
            if s < 0:
                self.ref = y
            elif s >= c.entry_m:
                self.phase = "prog"
                self.entry_t, self.entry_gap = t, gap > c.gap_s
                self.ref_at_entry = self.ref
                self._set_far(t, y, x, s, gap)
        else:
            if s > self.far:
                self._set_far(t, y, x, s, gap)
            A = self.far
            if A < c.min_amp_m:
                if s <= (1 - c.ret) * A or t - self.far_t >= c.stall_s:
                    self.phase = "wait"
                    self.ref = y
            else:
                if s <= (1 - c.ret) * A:
                    self._close("rep", t)
                    self.ref = y
                elif t - self.entry_t > c.fail_s:
                    self._close("failed", t)
                    self.ref = y
        self.prev_t = t

    def _set_far(self, t, y, x, s, gap):
        self.far, self.far_t, self.far_y, self.far_x = s, t, y, x
        self.far_gap = gap > self.cfg.gap_s       # right after a gap
        self._far_is_prev = True

    def _close(self, kind, t):
        c = self.cfg
        trip = Trip(kind, self.entry_t, self.far_t, self.far_y, self.far_x, self.far, self.ref_at_entry, t,
                    self.far_gap, self.entry_gap)
        self.phase = "wait"
        if kind == "rep":
            if t - self.entry_t < c.bounce_s and not self.entry_gap:
                trip.flags.append("bounce")
            elif self.mode == "live" and not trip.far_gap:
                amps = self._counted_amps()
                if amps and trip.amp < c.partial * float(np.median(amps)):
                    trip.flags.append("partial")
        else:
            amps, xs = self._counted_amps(), self._counted_xs()
            if amps and (trip.amp < c.partial * float(np.median(amps))
                         or abs(trip.far_x - float(np.median(xs))) > c.other_dx_m):
                trip.flags.append("rerack")
        self.trips.append(trip)

    def finish(self, t=None):
        """End of the recording / set: an excursion >= min_amp still open becomes a failed attempt."""
        if self.phase == "prog" and self.far >= self.cfg.min_amp_m:
            self._close("failed", self.prev_t if t is None else t)
        return self.trips


def run_states(t, y, x, gap, rest, mode="final", cfg: RepCfg = DEFAULT.rep):
    m = RepMachine(rest, mode, cfg)
    for a, b, c_, g in zip(t, y, x, gap):
        m.push(float(a), float(b), float(c_), float(g))
    return m.finish()
