"""Closing sets (pipeline tab, "세트"; 구현 설계 tab, close_set).

One recording can hold several sets. A Session runs the rep state machine and closes the open set when
  * gap      : sets.gap_s after the last rep with no excursion >= min_amp in progress (a failed attempt
               starts the clock only in a set without a rep: a rerack 13 cm below lockout is "failed" 20 s
               later and must not hold the set open)
  * exercise : no excursion in progress and the exercise label has differed from the set's exercise for
               sets.label_hold_s
  * end      : the recording ends (an open excursion >= min_amp becomes a failed attempt)
The time is measured from the moment the last rep was counted, and an excursion counts as in progress once
it is min_amp away, so the state machine sees the gap about 1 s longer than "last rep back -> next rep
starts" (근거·출처 tab, 그 밖의 시험).
Live: after sets.provisional_s with no excursion in progress a provisional report is due; a rep within
gap_s still joins the same set. Final: the set's trips go through sets.finalize (partial, other spot).
When the exercise changes to one with the other rest side, the state machine starts over; otherwise it keeps
its reference point and only its trip list is cleared.
Labels: label(t, name) at any time (frames without a plate included); None = no label.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import DEFAULT, Config
from .state import RepMachine, Trip
from .sets import finalize, counted, failed

REST = {"squat": "top", "bench": "top", "deadlift": "bottom"}


@dataclass
class SetResult:
    exercise: str
    trips: List[Trip]
    open_t: float                 # first sample after the previous set closed
    close_t: float
    reason: str                   # "gap" | "exercise" | "end"
    provisional: List[float] = field(default_factory=list)   # live: times a provisional report was due

    @property
    def reps(self):
        return counted(self.trips)

    @property
    def failed(self):
        return failed(self.trips)


class Session:
    def __init__(self, exercise: str, mode: str = "final", cfg: Config = DEFAULT):
        self.mode, self.cfg = mode, cfg
        self.exercise = exercise
        self.m = RepMachine(REST[exercise], mode, cfg.rep)
        self.sets: List[SetResult] = []
        self.label, self.label_since = None, None
        self._reset_set()

    def _reset_set(self):
        self.open_t = None
        self.last_t = None
        self.has_rep = False
        self.prov: List[float] = []
        self.prov_due = False

    def in_excursion(self):
        return self.m.phase == "prog" and self.m.far >= self.cfg.rep.min_amp_m

    # ---- input ---------------------------------------------------------------------------
    def push(self, t, y, x, gap=0.0):
        if self.open_t is None:
            self.open_t = t
        n0 = len(self.m.trips)
        self.m.push(t, y, x, gap)
        for tr in self.m.trips[n0:]:
            # the clock runs from the last rep; a failed attempt (a rerack, often) only starts it when the
            # set has no rep yet
            if tr.kind == "rep" and "bounce" not in tr.flags:
                self.has_rep = True
            elif tr.kind != "failed" or self.has_rep:
                continue
            self.last_t = tr.end_t
            self.prov_due = True
        self.tick(t)

    def label_at(self, t, name: Optional[str]):
        if name != self.label:
            self.label, self.label_since = name, t
        self.tick(t)

    def tick(self, t):
        sc = self.cfg.sets
        busy = self.in_excursion()
        if self.last_t is not None and not busy:
            idle = t - self.last_t
            if self.mode == "live" and self.prov_due and idle >= sc.provisional_s:
                self.prov.append(t)
                self.prov_due = False
            if idle >= sc.gap_s:
                self._close(t, "gap")
                return
        if (self.label is not None and self.label != self.exercise and not busy
                and t - self.label_since >= sc.label_hold_s):
            self._close(t, "exercise", new=self.label)

    def finish(self, t=None) -> List[SetResult]:
        self.m.finish(t)
        self._close(self.m.prev_t if t is None else t, "end")
        return self.sets

    # ---- closing -------------------------------------------------------------------------
    def _close(self, t, reason, new=None):
        trips = self.m.trips
        if trips:
            if self.mode == "final":
                trips = finalize(trips, self.cfg.rep)
            if counted(trips) or failed(trips):      # only partial / bounce / rerack trips: no set
                self.sets.append(SetResult(self.exercise, list(trips), self.open_t, t, reason, self.prov))
        self.m.trips = []
        self._reset_set()
        if new is not None and new != self.exercise:
            if REST[new] != REST[self.exercise]:
                self.m = RepMachine(REST[new], self.mode, self.cfg.rep)
            self.exercise = new


def run_session(sig_t, sig_y, sig_x, sig_gap, exercise, mode="final", label_t=(), label_v=(),
                cfg: Config = DEFAULT) -> List[SetResult]:
    """Signal samples and label events merged in time order (a label at the same time goes first)."""
    s = Session(exercise, mode, cfg)
    i = j = 0
    n, nl = len(sig_t), len(label_t)
    while i < n or j < nl:
        if j < nl and (i >= n or label_t[j] <= sig_t[i]):
            s.label_at(float(label_t[j]), label_v[j])
            j += 1
        else:
            s.push(float(sig_t[i]), float(sig_y[i]), float(sig_x[i]), float(sig_gap[i]))
            i += 1
    return s.finish()
