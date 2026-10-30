"""Transient-peak endpoint recovery, based on controlled complete motions."""
from dataclasses import replace
import numpy as np
import pytest
from core.config import DEFAULT
from core.judge.session import run_session
from core.measure.times import all_rep_values
from core.signal.final import clean_final


def waveform(peak,drop):
    t=np.arange(0,6,1/30)
    y=np.interp(t,[0,1,2,3,3.2,4.2,5,6],[0,0,-.5,peak,peak-drop,0,0,0])
    fr=clean_final(t,y,np.zeros_like(y),30)
    sets=run_session(fr.sig.t,fr.sig.y,fr.sig.x,fr.sig.gap_before,'squat')
    trips=[x for s in sets for x in s.reps]
    assert len(trips)==1
    return fr,trips


@pytest.mark.parametrize('peak,drop,missing',[(-.25,.06,False),(-.43,.06,True)])
def test_true_return_is_recovered_when_first_peak_is_premature(peak,drop,missing):
    fr,trips=waveform(peak,drop)
    off=all_rep_values(fr.sig,fr.raw_t,trips,'top',replace(DEFAULT.speed,recover_early_back=False))[0]
    on=all_rep_values(fr.sig,fr.raw_t,trips,'top',replace(DEFAULT.speed,recover_early_back=True))[0]
    assert (off.back is None)==missing
    if not missing:assert off.back < trips[0].end_t
    assert on.back>=trips[0].end_t and on.back==pytest.approx(4.2,abs=.2)
    assert on.rom==pytest.approx(.5,abs=.06) and on.far_out<on.back
    assert on.v_mean==pytest.approx(on.rom/(on.back-on.far_out))


def test_no_override_for_gap_and_disabled_is_exact():
    fr,trips=waveform(-.25,.06);trip=trips[0]
    old=all_rep_values(fr.sig,fr.raw_t,trips,'top',replace(DEFAULT.speed,recover_early_back=False))[0]
    gap_trip=replace(trip,far_gap=True)
    gap=all_rep_values(fr.sig,fr.raw_t,[gap_trip],'top',replace(DEFAULT.speed,recover_early_back=True))[0]
    assert gap==old
    legacy=all_rep_values(fr.sig,fr.raw_t,trips,'top',replace(DEFAULT.speed,recover_early_back=False))[0]
    assert legacy==old


def test_no_change_to_complete_return():
    fr,trips=waveform(-.25,.01)
    off=all_rep_values(fr.sig,fr.raw_t,trips,'top',replace(DEFAULT.speed,recover_early_back=False))[0]
    on=all_rep_values(fr.sig,fr.raw_t,trips,'top',replace(DEFAULT.speed,recover_early_back=True))[0]
    assert on==off and on.back>=trips[0].end_t
