"""Run the core on synthetic recordings and score it against the truth."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.config import DEFAULT, Config
from core.signal.final import clean_final
from core.signal.live import LiveCleaner
from core.judge.state import RepMachine
from core.judge.sets import finalize, counted, failed
from core.measure.times import all_rep_values, rep_values
from core.signal.final import resample_filter, remove_outliers


@dataclass
class Result:
    n_true: int
    final_n: int
    live_n: int
    final_ok: bool
    live_ok: bool
    final_time_ok: bool
    live_time_ok: bool
    final_failed: int = 0
    live_failed: int = 0
    final_partial: int = 0
    live_partial: int = 0
    speed: list = field(default_factory=list)       # (v, v_ref, blank) per matched rep, final
    speed_live: list = field(default_factory=list)  # (v, v_ref) per matched rep, live recompute
    count_lead: list = field(default_factory=list)  # live count time - true return time


def run_final(rec, cfg: Config = DEFAULT):
    fr = clean_final(rec.t, rec.y, rec.x, rec.fps, cfg.final)
    s = fr.sig
    m = RepMachine(rec.rest, "final", cfg.rep)
    for a, b, c, g in zip(s.t, s.y, s.x, s.gap_before):
        m.push(float(a), float(b), float(c), float(g))
    trips = finalize(m.finish(), cfg.rep)
    return fr, trips


def run_live(rec, cfg: Config = DEFAULT, with_signal=False):
    lc = LiveCleaner(rec.fps, cfg.live)
    m = RepMachine(rec.rest, "live", cfg.rep)
    sig = []
    for a, b, c in zip(rec.t, rec.y, rec.x):
        r = lc.push(float(a), float(b), float(c))
        if r is not None:
            m.push(*r)
            sig.append(r)
    trips = m.finish()
    return (trips, np.array(sig)) if with_signal else trips


def live_raise_end(sig, trip, rest, cfg: Config = DEFAULT):
    """When the live calculation knows the raise is over: no new high for live_flat_s, or the bar is
    drop_m below the high. The raise starts at the far point (squat, bench) or at the entry (deadlift)."""
    sc = cfg.speed
    t0 = trip.far_t if rest == "top" else trip.entry_t
    m = sig[:, 0] >= t0
    best_t, best = None, -np.inf
    for t, y in sig[m][:, :2]:
        if y > best:
            best, best_t = y, t
        if t - best_t >= sc.live_flat_s or best - y >= sc.drop_m:
            return float(t)
    return None


def time_match(reps, truth, tol=1.0):
    """Each counted rep's far time within tol of a distinct true far interval, in order."""
    if len(reps) != len(truth):
        return False
    for r, tr in zip(reps, truth):
        if not (tr.far_in - tol <= r.far_t <= tr.far_out + tol):
            return False
    return True


def _match_speed(reps_a, vals_a, reps_b, vals_b, tol=1.0):
    out = []
    j0 = 0
    for ra, va in zip(reps_a, vals_a):
        best, bd = None, tol
        for j in range(len(reps_b)):
            d = abs(reps_b[j].far_t - ra.far_t)
            if d < bd:
                best, bd = j, d
        if best is not None:
            out.append((va, vals_b[best]))
    return out


def live_recompute_speed(rec, trip, sig, cfg: Config = DEFAULT):
    """Live speed right after the raise: raw samples from live_pre_s before the rep entry up to the moment
    the live calculation knows the raise ended, cleaned and filtered like the final calculation."""
    sc = cfg.speed
    t1 = live_raise_end(sig, trip, rec.rest, cfg)
    if t1 is None:
        return None
    t0 = trip.entry_t - sc.live_pre_s
    m = (rec.t >= t0) & (rec.t <= t1)
    if m.sum() < 10:
        return None
    t, y, x = rec.t[m], rec.y[m], rec.x[m]
    keep = remove_outliers(t, y, x, cfg.final)
    sig = resample_filter(t[keep], y[keep], x[keep], rec.fps, cfg.final)
    return rep_values(sig, t[keep], trip, rec.rest, None, None, sc)


def evaluate(rec, cfg: Config = DEFAULT, speed=True, live=True):
    fr, trips = run_final(rec, cfg)
    reps_f = counted(trips)
    res_live, live_sig = run_live(rec, cfg, with_signal=True) if live else ([], None)
    reps_l = counted(res_live)
    n = len(rec.reps)
    r = Result(n, len(reps_f), len(reps_l), len(reps_f) == n, len(reps_l) == n,
               time_match(reps_f, rec.reps), time_match(reps_l, rec.reps),
               len(failed(trips)), len(failed(res_live)),
               sum("partial" in t.flags for t in trips), sum("partial" in t.flags for t in res_live))
    if speed and r.final_ok:
        vals = all_rep_values(fr.sig, fr.raw_t, reps_f, rec.rest, cfg.speed)
        # reference: the same calculation on the noise-free signal at the same frame times
        ref = clean_final(rec.frame_t, rec.true_y, rec.true_x, rec.fps, cfg.final)
        m = RepMachine(rec.rest, "final", cfg.rep)
        for a, b, c, g in zip(ref.sig.t, ref.sig.y, ref.sig.x, ref.sig.gap_before):
            m.push(float(a), float(b), float(c), float(g))
        ref_reps = counted(finalize(m.finish(), cfg.rep))
        ref_vals = all_rep_values(ref.sig, ref.raw_t, ref_reps, rec.rest, cfg.speed)
        for va, vb in _match_speed(reps_f, vals, ref_reps, ref_vals):
            r.speed.append((va.v_mean, vb.v_mean, va.blank))
        if live and r.live_ok:
            lv = [live_recompute_speed(rec, tr, live_sig, cfg) for tr in reps_l]
            for va, vb in _match_speed(reps_l, lv, ref_reps, ref_vals):
                if va is not None:
                    r.speed_live.append((va.v_mean, vb.v_mean))
    return r


# ---- several sets in one recording (core.judge.session) ---------------------------------------

def run_final_sets(t, y, x, fps, exercise, label_t=(), label_v=(), cfg: Config = DEFAULT):
    from core.judge.session import run_session
    fr = clean_final(t, y, x, fps, cfg.final)
    s = fr.sig
    return fr, run_session(s.t, s.y, s.x, s.gap_before, exercise, "final", label_t, label_v, cfg)


def run_live_sets(t, y, x, fps, exercise, label_t=(), label_v=(), cfg: Config = DEFAULT):
    from core.judge.session import Session
    lc = LiveCleaner(fps, cfg.live)
    sess = Session(exercise, "live", cfg)
    j, nl = 0, len(label_t)
    for a, b, c in zip(t, y, x):
        while j < nl and label_t[j] <= a:
            sess.label_at(float(label_t[j]), label_v[j])
            j += 1
        r = lc.push(float(a), float(b), float(c))
        if r is not None:
            sess.push(*r)
    while j < nl:
        sess.label_at(float(label_t[j]), label_v[j])
        j += 1
    return sess.finish()
