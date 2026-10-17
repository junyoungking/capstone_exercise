import numpy as np
import pytest

from core.config import DEFAULT
from core.signal.final import clean_final, remove_outliers
from core.signal.live import LiveCleaner, clean_live_all
from core.judge.state import RepMachine, run_states
from core.judge.sets import finalize, counted, failed
from core.measure.times import all_rep_values
from sim.gen import Timeline, make_recording, Opts, build


def squat_like(n_reps=3, rom=0.5, fps=30.0, top=1.5, pause=1.0):
    tl = Timeline(top, 0.0)
    tl.hold(1.0)
    for _ in range(n_reps):
        tl.move(1.5, y1=top - rom)
        tl.hold(0.1)
        tl.move(1.0, y1=top)
        tl.hold(pause)
    t, y, x = tl.arrays()
    ft = np.arange(0, t[-1], 1 / fps)
    return ft, np.interp(ft, t, y), np.interp(ft, t, x)


# ---- final signal ---------------------------------------------------------------------------

def test_outlier_spike_removed():
    t = np.arange(0, 3, 1 / 30)
    y = np.full_like(t, 1.0)
    y[45] += 0.5
    keep = remove_outliers(t, y, np.zeros_like(t))
    assert not keep[45]
    assert keep[10:40].all() and keep[50:80].all()


def test_gap_edges_dropped_and_split():
    t = np.r_[np.arange(0, 2, 1 / 30), np.arange(2.8, 5, 1 / 30)]
    y = np.full_like(t, 1.0)
    fr = clean_final(t, y, np.zeros_like(t), 30.0)
    # the samples on both sides of the 0.8 s gap are dropped in the first pass
    assert fr.raw_t.max() < 2.8 + 1e-9 or 2.8 not in fr.raw_t
    assert np.isclose(fr.raw_t[fr.raw_t < 2.5].max(), t[t < 2].max() - 1 / 30)
    # gap > 0.5 s stays open: two stretches
    assert np.sum(fr.sig.gap_before > 0.5) == 1


def test_short_gap_interpolated():
    t = np.r_[np.arange(0, 2, 1 / 30), np.arange(2.3, 4, 1 / 30)]
    y = np.full_like(t, 1.0)
    fr = clean_final(t, y, np.zeros_like(t), 30.0)
    assert np.all(fr.sig.gap_before == 0)
    assert np.allclose(np.diff(fr.sig.t), 1 / 30)


# ---- live signal ----------------------------------------------------------------------------

def test_live_start_and_outliers():
    lc = LiveCleaner(30.0)
    outs = [lc.push(i / 30, 1.0, 0.0) for i in range(5)]
    assert outs[:4] == [None] * 4 and outs[4] is not None
    assert lc.push(5 / 30, 1.6, 0.0) is None        # spike rejected
    r = lc.push(6 / 30, 1.0, 0.0)
    assert r is not None and abs(r[1] - 1.0) < 1e-6


def test_live_accepts_after_four_rejects():
    lc = LiveCleaner(30.0)
    for i in range(5):
        lc.push(i / 30, 1.0, 0.0)
    res = [lc.push((5 + i) / 30, 3.0, 0.0) for i in range(5)]
    assert res[:4] == [None] * 4 and res[4] is not None


def test_live_restart_after_gap():
    lc = LiveCleaner(30.0)
    for i in range(10):
        lc.push(i / 30, 1.0, 0.0)
    assert lc.push(1.0, 1.0, 0.0) is None            # 0.7 s gap -> restart, needs 5 samples again
    assert lc.started is False or len(lc.buf) == 1


# ---- state machine --------------------------------------------------------------------------

def test_counts_clean_squat():
    t, y, x = squat_like(4)
    trips = finalize(run_states(t, y, x, np.zeros_like(t), "top"))
    assert len(counted(trips)) == 4


def test_small_excursion_not_counted():
    t, y, x = squat_like(3, rom=0.10)
    trips = finalize(run_states(t, y, x, np.zeros_like(t), "top"))
    assert len(counted(trips)) == 0


def test_return_ratio():
    tl = Timeline(1.5, 0.0)
    tl.hold(1.0)
    tl.move(1.5, y1=1.0)            # 50 cm down
    tl.move(1.0, y1=1.5 - 0.5 * 0.35)   # back only 65%
    tl.hold(3.0)
    t, y, x = tl.arrays()
    trips = run_states(t[::33], y[::33], x[::33], np.zeros(len(t[::33])), "top")
    assert not any(tr.kind == "rep" for tr in trips)
    assert any(tr.kind == "failed" for tr in trips)    # still out at the end


def test_bounce_dropped():
    tl = Timeline(0.225, 0.0)
    tl.hold(1.0)
    tl.move(0.15, y1=0.225 + 0.18)
    tl.move(0.15, y1=0.225)
    tl.hold(2.0)
    t, y, x = tl.arrays()
    t, y, x = t[::33], y[::33], x[::33]
    trips = run_states(t, y, x, np.zeros_like(t), "bottom")
    assert trips and all("bounce" in tr.flags for tr in trips if tr.kind == "rep")


def test_partial_flagged_final():
    tl = Timeline(1.5, 0.0)
    tl.hold(1.0)
    for rom in (0.5, 0.5, 0.2, 0.5):
        tl.move(1.2, y1=1.5 - rom)
        tl.move(1.0, y1=1.5)
        tl.hold(1.0)
    t, y, x = tl.arrays()
    t, y, x = t[::33], y[::33], x[::33]
    trips = finalize(run_states(t, y, x, np.zeros_like(t), "top"))
    assert len(counted(trips)) == 3
    assert sum("partial" in tr.flags for tr in trips) == 1


# ---- rep values -----------------------------------------------------------------------------

def test_speed_close_to_true_on_clean_signal():
    t, y, x = squat_like(3, rom=0.5)
    fr = clean_final(t, y, x, 30.0)
    trips = finalize(run_states(fr.sig.t, fr.sig.y, fr.sig.x, fr.sig.gap_before, "top"))
    reps = counted(trips)
    vals = all_rep_values(fr.sig, fr.raw_t, reps, "top")
    for v in vals:
        assert v.v_mean is not None
        assert abs(v.v_mean - 0.5 / 1.0) / 0.5 < 0.05      # cosine raise of 0.5 m in 1.0 s
        assert abs(v.down - 1.5) < 0.1


# ---- generator ------------------------------------------------------------------------------

def test_generator_deterministic():
    a = make_recording("squat", np.random.default_rng(7), 30, 0.01)
    b = make_recording("squat", np.random.default_rng(7), 30, 0.01)
    assert np.array_equal(a.t, b.t) and np.array_equal(a.y, b.y)
    assert len(a.reps) == len(b.reps) >= 2


@pytest.mark.parametrize("ex", ["squat", "bench", "deadlift"])
def test_base_set_counts(ex):
    rng = np.random.default_rng(11)
    from sim.evaluate import evaluate
    oks = [evaluate(make_recording(ex, rng, 30, 0.005), speed=False).final_ok for _ in range(10)]
    assert np.mean(oks) >= 0.9


def test_each_rep_measured_on_its_own_excursion():
    t, y, x = squat_like(4, rom=0.5, pause=0.6)
    fr = clean_final(t, y, x, 30.0)
    reps = counted(finalize(run_states(fr.sig.t, fr.sig.y, fr.sig.x, fr.sig.gap_before, "top")))
    vals = all_rep_values(fr.sig, fr.raw_t, reps, "top")
    for r, v in zip(reps, vals):
        assert abs(v.far_in - r.far_t) < 0.3
        assert v.start < r.entry_t < v.far_in
