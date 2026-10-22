import numpy as np

from core.recog.pose import select_lifter


def person(cx,cy, wrists, spread=20):
    p=np.ones((33,4),float);p[:,:2]=(cx,cy);p[:,2]=0
    p[11,:2]=(cx-spread,cy-20);p[12,:2]=(cx+spread,cy-20)
    p[23:25,:2]=(cx,cy+20)
    p[15,:2],p[16,:2]=wrists
    return p


def test_wrist_distance_beats_larger_overlapping_person_box():
    large=person(110,110,[(180,110),(180,110)],spread=100)
    near=person(180,110,[(115,110),(115,110)])
    assert select_lifter([large,near],(100,100,120,120),None,'squat',image_size=(400,300)) == 1


def test_box_fallback_when_both_wrists_are_invisible():
    near=person(115,110,[(300,100),(300,100)])
    near[[15,16],3]=.49
    far=person(220,110,[(220,110),(220,110)])
    assert select_lifter([far,near],(100,100,120,120),None,'bench',image_size=(400,300)) == 1


def test_out_of_frame_wrists_do_not_count_as_visible_wrist_observations():
    near=person(115,110,[(115,-1),(115,-1)])
    other=person(150,110,[(140,110),(140,110)])
    assert select_lifter([other,near],(100,100,120,120),None,'deadlift',image_size=(400,300)) == 1


def test_limit_rejects_all_exercises_identically():
    distant=person(10000,10000,[(10000,10000),(10000,10000)])
    for exercise in ['squat','bench','deadlift']:
        assert select_lifter([distant],(100,100,120,120),None,exercise,image_size=(20000,20000)) is None


def test_equal_wrist_distance_uses_previous_center_then_stable_index():
    a=person(100,110,[(90,110),(90,110)],spread=60)
    b=person(200,110,[(130,110),(130,110)])
    plate=(100,100,120,120)
    assert select_lifter([a,b],plate,np.array([200,110]),'squat',image_size=(400,300)) == 1
    assert select_lifter([a,b],plate,None,'squat',image_size=(400,300)) == 0
