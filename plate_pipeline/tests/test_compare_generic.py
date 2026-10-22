from dataclasses import asdict

import pytest

from core.compare.summary import from_values
from core.compare.judge import compare_ref,compare_series_ref,floor_for,min_change_for


@pytest.mark.parametrize('value',['down','far_pause','far_dist'])
@pytest.mark.parametrize('mine,ref',[
    ([1.6,1.5,1.7,1.6,1.5],[1.0,1.1,1.0,1.1]),
    ([1.6,1.5,1.7,1.6],[1.0]),
    ([1.6,1.5,1.7],[1.0]),
    ([1.6,None,1.7,1.6,None,1.5],[1,1.1,1]),
    ([1.6,1.5,1.7,1.6],[1,None,1.1,None,1]),
    ([1.6,1.5,1.7,1.6],[]),
])
def test_plate_wrapper_is_the_same_generic_rule_with_plate_units(value,mine,ref):
    a=from_values('squat',10,**{value:mine});b=from_values('squat',60,**{value:ref})
    generic=compare_series_ref(a.get(value),b.get(value),value,floor_a=floor_for(value,a.fps),
         floor_b=floor_for(value,b.fps),min_change=min_change_for(value),min_ref_n=3 if value=='far_pause' else 1)
    assert asdict(generic)==asdict(compare_ref(a,b,value))


def test_instability_guard_is_optional_and_does_not_change_plate_behavior():
    a=from_values('squat',30,down=[0,180,0,180,0,180]);b=from_values('squat',30,down=[100]*6)
    guarded=compare_series_ref(a.get('down'),b.get('down'),'down',floor_a=3,floor_b=3,min_change=18,max_user_scatter=36)
    assert guarded.reason=='unstable_user'
    assert compare_ref(a,b,'down').reason is None
