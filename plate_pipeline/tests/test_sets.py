import numpy as np

from core.judge.session import Session, run_session
from sim.gen import Timeline

FPS = 30.0


def reps(tl, n, rom=0.5, down=1.5, up=1.0, pause=1.0, sign=-1):
    for _ in range(n):
        top = tl.cy
        tl.move(down if sign < 0 else up, y1=top + sign * rom)
        tl.hold(0.1)
        tl.move(up if sign < 0 else down, y1=top)
        tl.hold(pause)


def sample(tl, holes=()):
    t, y, x = tl.arrays()
    ft = np.arange(0, t[-1], 1 / FPS)
    keep = np.ones(len(ft), bool)
    for a, b in holes:
        keep &= ~((ft >= a) & (ft < b))
    st = ft[keep]
    gap = np.diff(st, prepend=st[0]) - 1 / FPS
    return ft, st, np.interp(st, t, y), np.interp(st, t, x), np.clip(gap, 0, None)


def run(tl, ex="squat", mode="final", labels=None, holes=()):
    ft, t, y, x, g = sample(tl, holes)
    lt, lv = (ft, [labels(f) for f in ft]) if labels else ((), ())
    return run_session(t, y, x, g, ex, mode, lt, lv)


def squat_sets(rest_s, n1=3, n2=2):
    tl = Timeline(1.5, 0.0)
    tl.hold(2.0)
    reps(tl, n1)
    tl.hold(rest_s)
    reps(tl, n2)
    tl.hold(2.0)
    return tl


def test_gap_over_45s_splits():
    out = run(squat_sets(60.0))
    assert [(len(s.reps), s.reason) for s in out] == [(3, "gap"), (2, "end")]


def test_gap_under_45s_merges():
    out = run(squat_sets(30.0))
    assert [len(s.reps) for s in out] == [5]


def test_excursion_keeps_the_set_open():
    tl = Timeline(1.5, 0.0)
    tl.hold(2.0)
    reps(tl, 3)
    tl.hold(40.0)
    tl.move(1.5, y1=1.0)          # down, held at the bottom past the 45 s mark, then up
    tl.hold(10.0)
    tl.move(1.0, y1=1.5)
    tl.hold(2.0)
    out = run(tl)
    assert [len(s.reps) for s in out] == [4]


def test_exercise_change_restarts_the_machine():
    tl = Timeline(1.5, 0.0)
    tl.hold(2.0)
    reps(tl, 3)
    tl.hold(5.0)
    t_switch = tl.now
    tl.hold(4.0)
    tl.move(0.2, y1=0.225)        # unseen: the lifter goes to the deadlift bar
    tl.hold(6.0)
    reps(tl, 3, rom=0.5, down=1.0, up=1.0, sign=+1)
    tl.hold(2.0)
    holes = [(t_switch + 1.0, t_switch + 9.0)]
    out = run(tl, labels=lambda f: "squat" if f < t_switch else "deadlift", holes=holes)
    assert [(s.exercise, len(s.reps), s.reason) for s in out] == [("squat", 3, "exercise"), ("deadlift", 3, "end")]


def test_short_label_flicker_is_ignored():
    tl = squat_sets(20.0)
    out = run(tl, labels=lambda f: "bench" if 20.0 <= f < 21.0 else "squat")
    assert [(s.exercise, len(s.reps)) for s in out] == [("squat", 5)]


def test_live_provisional_report_and_same_set():
    out = run(squat_sets(20.0), mode="live")
    assert len(out) == 1 and len(out[0].reps) == 5
    assert len(out[0].provisional) == 1        # during the 20 s rest; the 2 s tail is too short for another


def test_failed_last_rep():
    tl = Timeline(1.5, 0.0)
    tl.hold(2.0)
    reps(tl, 3)
    tl.move(1.5, y1=1.0)
    tl.move(0.5, y1=1.1)
    tl.move(0.5, y1=0.98)          # onto the safeties
    tl.hold(3.0)
    for mode in ("final", "live"):
        out = run(tl, mode=mode)
        assert len(out) == 1 and len(out[0].reps) == 3 and len(out[0].failed) == 1


def test_session_without_reps_emits_nothing():
    tl = Timeline(1.5, 0.0)
    tl.hold(10.0)
    assert run(tl) == []


def test_rerack_below_13cm_does_not_hold_the_set_open():
    # bench: the hooks are 14 cm under lockout; resting there becomes a "failed" (rerack) trip 20 s later
    tl = Timeline(1.5, 0.0)
    tl.hold(2.0)
    reps(tl, 3)
    tl.move(1.0, y1=1.36)
    tl.hold(45.0)
    tl.move(1.0, y1=1.5)
    tl.hold(3.0)
    reps(tl, 2)
    tl.hold(2.0)
    out = run(tl)
    assert [len(s.reps) for s in out] == [3, 2]
