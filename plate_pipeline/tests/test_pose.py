import builtins
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.config import DEFAULT
from core.recog.pose import (PoseEstimator, SIDE_IDS, SideLock, person_box,
                             select_lifter)
from eval.pose_check import floor_intervals, summarize
from server.jobs import process_video
from test_upload import clip  # reuse the existing synthetic video fixture


def person(box):
    x0,y0,x1,y1 = box
    p = np.ones((33,4), float)
    p[:, :2] = ((x0+x1)/2, (y0+y1)/2)
    p[:, 2] = 0
    p[0, :2] = ((x0+x1)/2, y0)
    p[11, :2] = (x0, y0+(y1-y0)*.2)
    p[12, :2] = (x1, y0+(y1-y0)*.2)
    p[23:25, 1] = y0+(y1-y0)*.5
    p[27:33, 1] = y1
    return p


def test_person_box_visible_feet_only():
    p = person((10,20,100,200))
    p[15,1] = 500  # low hand must not define the lower edge
    p[31,1] = 210
    p[32] = [900,900,0,.1]  # invisible outlier
    assert person_box(p) == (10,20,100,210)
    p[27:33,3] = .2
    assert person_box(p) is None
    assert person_box(None) is None
    p[27,3] = .3
    assert person_box(p) == (10,20,100,200)


def test_plate_near_lifter_beats_large_background():
    background = person((0,0,250,250))
    lifter = person((300,100,410,300))
    assert select_lifter([background,lifter], (390,220,430,300), None, 'squat') == 1


def test_only_far_people_returns_none():
    assert select_lifter([person((0,0,100,100))], (300,300,340,340), None, 'squat') is None
    assert select_lifter([], None, None, 'squat') is None


def test_plate_ranking_distance_then_area_then_continuity():
    plate = (100,100,120,120)
    small = person((100,90,130,130))
    large = person((90,80,140,150))
    assert select_lifter([small,large], plate, None, 'squat') == 1
    # Both contain plate center and have equal area; continuity is last tie-breaker.
    a,b = person((90,60,130,160)),person((100,60,140,160))
    prev = b[[11,12,23,24],:2].mean(axis=0)
    assert select_lifter([a,b],plate,prev,'squat') == 1
    # Exact half-diameter distance qualifies, farther does not.
    edge = person((120,100,150,140))
    assert select_lifter([edge],plate,None,'squat') == 0
    assert select_lifter([edge],plate,None,'squat',cfg=replace(DEFAULT.pose,near_plate=.49)) is None


def test_no_plate_size_and_continuity():
    small = person((.1,.1,.3,.6))
    large = person((.5,.1,.9,.9))
    assert select_lifter([small,large],None,None,'squat') == 1
    a,b = person((.1,.1,.3,.6)),person((.7,.1,.9,.6))
    prev = b[[11,12,23,24],:2].mean(axis=0)
    assert select_lifter([a,b],None,prev,'squat') == 1
    # Pixel conversion preserves v4 scoring when image dimensions are supplied.
    scale = np.array([640,480,1,1])
    assert select_lifter([a*scale,b*scale],None,prev*scale[:2],'squat',image_size=(640,480)) == 1
    assert select_lifter([a*scale,b*scale],None,prev*scale[:2]+[8,0],'squat') == 1


def test_no_plate_bench_prefers_horizontal_torso():
    upright = person((.1,.05,.5,.95))
    horizontal = person((.2,.3,.9,.7))
    horizontal[11:13,:2] = (.8,.4)
    horizontal[23:25,:2] = (.3,.4)
    assert select_lifter([upright,horizontal],None,None,'bench') == 1


def test_side_lock_holds_and_resets_candidate():
    lock = SideLock('bench')
    p = person((0,0,1,1))
    left = list(SIDE_IDS['left'].values())
    right = list(SIDE_IDS['right'].values())
    p[left,3],p[right,3] = 1,.1
    assert lock.update(p) == 'left'
    p[left,3],p[right,3] = .1,1
    for _ in range(5):
        assert lock.update(p) == 'left'
    p[:,3] = .8
    assert lock.update(p) == 'left'
    p[left,3],p[right,3] = .1,1
    for _ in range(5):
        assert lock.update(p) == 'left'
    assert lock.update(p) == 'right'
    assert lock.update(None) == 'right'


def test_estimator_pixels_rgb_and_monotonic_ms_without_mediapipe():
    est = PoseEstimator.__new__(PoseEstimator)
    est._last_ms = -1
    calls = []
    def detect(image, ms):
        calls.append((image,ms))
        return SimpleNamespace(pose_landmarks=[[SimpleNamespace(x=.25,y=.75,z=-.2,visibility=.9)]*33])
    est._mp = SimpleNamespace(Image=lambda **kw: kw['data'], ImageFormat=SimpleNamespace(SRGB='rgb'))
    est.landmarker = SimpleNamespace(detect_for_video=detect)
    img = np.full((20,40,3),[1,2,3],dtype=np.uint8)
    lms = est(img, .1)[0]
    est(img,.1)
    assert lms.shape == (33,4)
    np.testing.assert_allclose(lms[0],[10,15,-.2,.9])
    assert calls[0][0][0,0].tolist() == [3,2,1]
    assert [c[1] for c in calls] == [100,101]


@pytest.mark.parametrize('every_n',[2,3])
def test_fake_pose_preserves_synthetic_reps(clip, monkeypatch, every_n):
    box = (120,110,200,190)
    baseline, base_rows, base_sig = process_video(clip,'squat',init_box=box)
    calls = []
    def fake(img,t):
        calls.append(t)
        return [person((40,30,270,450))]
    fake.every_n = every_n
    original_import = builtins.__import__
    def without_mediapipe(name,*args,**kwargs):
        if name == 'mediapipe' or name.startswith('mediapipe.'):
            raise ImportError('unit tests must not need MediaPipe')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',without_mediapipe)
    result,rows,sig = process_video(clip,'squat',init_box=box,pose=fake)
    assert result['sets'] == baseline['sets']
    assert sig == base_sig
    assert [s['reps'] for s in result['sets']] == [3]
    assert len(calls) == len(range(0,len(rows),every_n))
    for i,(r,b) in enumerate(zip(rows,base_rows)):
        assert r['plate'] == b['plate']
        assert ('pose' in r) == ('kp' in r) == (i % every_n == 0)
        if 'kp' in r:
            assert np.asarray(r['kp']).shape == (33,3)
            assert r['person']['vis'] == 1
            assert r['pose']['n'] == 1


def test_pose_check_stationary_windows_and_gaps():
    rows=[]
    for k in range(31):
        rows.append(dict(t=k/30,plate=dict(used=True,center=[50,50],size_px=[100,100],box=[0,0,100,100]),
                         person=dict(box=[0,0,100,150]),pose=dict(n=2,ms=3)))
    stats = summarize(rows)
    assert stats['pose_frames'] == 31 and stats['lifter_ratio'] == 1
    assert stats['candidates'] == {2:31}
    assert stats['pose_ms_median'] == 3
    assert stats['floor'] == [dict(t0=0,t1=1,n=31,median=.5)]
    for r in rows[5:26]:
        r['plate']['used'] = False
    assert floor_intervals(rows) == []


def test_mediapipe_synthetic_smoke(clip):
    pytest.importorskip('mediapipe')
    result,rows,_ = process_video(clip,'squat',init_box=(120,110,200,190),pose=True)
    assert [s['reps'] for s in result['sets']] == [3]
    assert sum('pose' in r for r in rows) == 210


def test_floor_intervals_do_not_merge_unbounded_slow_drift():
    rows = [dict(t=k/30,plate=dict(used=True,center=[50,100-k/3],
                 size_px=[100,100],box=[0,0,100,100])) for k in range(121)]
    # Each 0.5s window moves <3cm, but the full 4s drift is 18cm.
    intervals = floor_intervals(rows)
    assert len(intervals) > 1
    assert all(r['t1']-r['t0'] <= .7 for r in intervals)
