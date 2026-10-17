from types import SimpleNamespace as NS

import numpy as np
import pytest

from core.compare.summary import from_values, summarize
from core.compare.judge import (Multipliers, compare_inset, compare_ref, run_compare, floor_for)


# ---- summary ---------------------------------------------------------------------------------

def test_scatter_ignores_linear_trend():
    s = from_values("squat", 30, down=[2.0, 2.1, 2.2, 2.3, 2.4])
    assert s.get("down").scatter() == pytest.approx(0.0)


def test_scatter_value():
    # second differences 1, -1 -> s^2 = 2 / (6 * 2)
    s = from_values("squat", 30, down=[0.0, 1.0, 1.0, 2.0])
    assert s.get("down").scatter() == pytest.approx(np.sqrt(2 / 12))


def test_scatter_needs_three_in_a_row():
    s = from_values("squat", 30, down=[1.0, None, 1.2, 1.1, None, 1.3])
    assert s.get("down").n == 4
    assert s.get("down").scatter() is None
    s = from_values("squat", 30, down=[1.0, None, 1.2, 1.1, 1.4, None])
    assert s.get("down").scatter() is not None


def _rv(**k):
    base = dict(start=None, far_in=None, far_out=None, back=None, rom=None, far_dist=None, v_mean=None,
                down=None, up=None, far_pause=None, blank="")
    base.update(k)
    return NS(**base)


def test_summarize_drops_low_confidence_and_keeps_rep_numbers():
    trips = [NS(far_gap=False), NS(far_gap=True), NS(far_gap=False), NS(far_gap=False)]
    vals = [_rv(down=1.0, start=0.0, back=2.0), _rv(down=5.0, start=3.0, back=5.0),
            _rv(down=1.2, start=6.0, back=8.0), _rv(down=1.1, start=9.0, back=11.0)]
    s = summarize("squat", 30, trips, vals)
    d = s.get("down")
    assert d.reps == [1, 3, 4] and list(d.vals) == [1.0, 1.2, 1.1]
    assert d.scatter() is None                  # rep 2 left out breaks the chain
    assert list(s.get("rest_pause").vals) == [1.0, 1.0]   # rep 1 -> 2 is low confidence (rep 2 out)


# ---- multipliers -----------------------------------------------------------------------------

def test_multiplier_table_matches_doc():
    K = Multipliers()
    assert [K.inset(n) for n in range(4, 13)] == [6.5, 4.9, 4.2, 3.7, 3.4, 3.3, 3.1, 3.0, 2.9]
    assert [K.short(n) for n in range(3, 11)] == [25.7, 7.2, 5.3, 4.4, 4.0, 3.6, 3.5, 3.3]
    assert K.inset(3) is None and K.inset(30) == 2.9 and K.short(30) == 3.3
    assert K.ref(3, 3) == 6.5 and K.ref(10, 10) == 3.1 and K.ref(25, 40) == 3.1


# ---- in-set ----------------------------------------------------------------------------------

def test_inset_too_short():
    c = compare_inset(from_values("squat", 30, down=[2.0, 2.1, 2.2]), "down")
    assert (c.result, c.reason, c.n_a, c.need) == ("skip", "set_too_short", 3, 4)


def test_inset_no_scatter():
    c = compare_inset(from_values("squat", 30, down=[1.0, None, 1.2, 1.1, None, 1.3]), "down")
    assert (c.result, c.reason, c.side) == ("skip", "no_scatter", "mine")


def test_inset_floor_and_multiplier():
    # straight line: scatter 0 -> raised to the 0.07 s floor
    c = compare_inset(from_values("squat", 30, down=[2.0, 2.1, 2.2, 2.3]), "down")
    assert c.s_a == pytest.approx(0.07) and c.k == 6.5
    assert c.delta == pytest.approx(0.2) and c.result == "same"          # 0.2 < 6.5 * 0.07
    c = compare_inset(from_values("squat", 30, down=[2.0 + 0.1 * i for i in range(8)]), "down")
    assert c.delta == pytest.approx(0.6) and c.k == 3.4 and c.result == "different"


def test_min_change():
    # 10 x 10 against a reference: k * SE = 3.1 * 0.07 * sqrt(0.2) = 0.097 s; 0.12 s is over that but under 0.15 s
    mine = from_values("squat", 30, down=[1.72 + 0.01 * i for i in range(10)], far_pause=[0.1] * 10)
    ref = from_values("squat", 30, down=[1.60 + 0.01 * i for i in range(10)], far_pause=[0.1] * 10)
    c = compare_ref(mine, ref, "down")
    assert c.k * c.se < abs(c.delta) < 0.15 and c.result == "same"
    # in-set the floor already keeps k * SE above the minimum change (2.9 * 0.01 m > 0.02 m)
    c = compare_inset(from_values("squat", 30, far_dist=[0.50] * 8 + [0.50, 0.50, 0.45, 0.45]), "far_dist")
    assert c.result == "different" and c.delta == pytest.approx(-0.05)


def test_time_floor_follows_fps():
    assert floor_for("down", 10) == pytest.approx(0.1)
    assert floor_for("down", 60) == pytest.approx(0.07)
    assert floor_for("far_dist", 10) == pytest.approx(0.01)


# ---- vs reference ----------------------------------------------------------------------------

def _set(down, pause=None, ex="squat"):
    return from_values(ex, 30, down=down, far_pause=pause if pause is not None else [0.1] * len(down))


def test_ref_full():
    mine = _set([2.6, 2.5, 2.7, 2.6, 2.5])
    ref = _set([1.6, 1.5, 1.7, 1.6, 1.5])
    c = compare_ref(mine, ref, "down")
    assert c.k == Multipliers().ref(5, 5) and c.delta == pytest.approx(1.0) and c.result == "different"


def test_ref_short():
    ref = _set([1.6])
    c = compare_ref(_set([2.6, 2.5, 2.7]), ref, "down")
    assert (c.result, c.reason, c.need) == ("skip", "set_too_short", 4)
    c = compare_ref(_set([2.6, 2.5, 2.7, 2.6]), ref, "down")
    assert c.k == 7.2 and c.se == pytest.approx(c.s_a * np.sqrt(1 / 4 + 1))
    c = compare_ref(_set([2.6, 2.5, 2.7, 2.6]), ref, "far_pause")
    assert (c.result, c.reason, c.need, c.n_b) == ("skip", "ref_too_short", 3, 1)


def test_ref_no_scatter_on_ref_side():
    ref = from_values("squat", 30, down=[1.6, None, 1.5, None, 1.7], far_pause=[0.1, None, 0.1, None, 0.1])
    c = compare_ref(_set([2.6, 2.5, 2.7, 2.6]), ref, "down")
    assert (c.result, c.reason, c.side) == ("skip", "no_scatter", "ref")


def test_run_compare_skips():
    mine = _set([2.6, 2.5, 2.7, 2.6])
    r = run_compare(mine, _set([1.6, 1.5, 1.7], ex="bench"))
    assert r[-1].reason == "exercise_mismatch" and len(r) == 3
    r = run_compare(mine, from_values("squat", 30, down=[], far_pause=[]))
    assert r[-1].reason == "no_ref_values"
    r = run_compare(mine, _set([1.6, 1.5, 1.7, 1.6]))
    assert [(c.kind, c.value) for c in r] == [("inset", "far_dist"), ("inset", "down"),
                                             ("vs_ref", "down"), ("vs_ref", "far_pause")]


def test_null_false_rate_small():
    """Independent normal values, no difference: each value 'different' at most ~2.5%."""
    rng = np.random.default_rng(7)
    hits = n = 0
    for _ in range(3000):
        x = rng.normal(2.0, 0.5, 6)
        c = compare_inset(from_values("squat", 30, down=list(x)), "down")
        hits += c.result == "different"
        n += 1
    assert hits / n < 0.035
