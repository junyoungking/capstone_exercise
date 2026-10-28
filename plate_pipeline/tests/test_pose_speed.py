"""M1b state/crop/capability regressions without model downloads."""
from dataclasses import replace
from types import SimpleNamespace
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from core.config import DEFAULT
from core.recog.pose import PoseEstimator, lifter_crop, lifter_lost, select_lifter
from test_rtmpose import person


@pytest.mark.parametrize("n", [3,5,10])
def test_interval_and_immediate_redetection(monkeypatch,n):
    calls=[]
    def detector(image):
        calls.append('det')
        return np.array([[30,20,180,180], [180,20,300,180]])
    def pose(image,bboxes):
        calls.append(len(bboxes))
        return np.ones((len(bboxes),26,2))*50,np.ones((len(bboxes),26))*.6
    monkeypatch.setitem(sys.modules,'rtmlib',SimpleNamespace(BodyWithFeet=lambda **kw:
        SimpleNamespace(det_model=detector,pose_model=pose)))
    est=PoseEstimator(cfg=replace(DEFAULT.pose,mode='tracked',det_frequency=n))
    p=person((100,100),(110,110));image=np.zeros((300,400,3),np.uint8)
    plate=(100,100,120,120)
    for frame in range(n+1):
        est(image,frame/30);est.observe(p,plate,(400,300))
    assert est.detection_frames==[0,n]
    assert calls==['det',2]+[1]*(n-1)+['det',2]
    p[[11,12,23,24],3]=.29
    est.observe(p,plate,(400,300));est(image,(n+1)/30)
    assert est.detection_frames==[0,n,n+1]
    est.observe(None,plate,(400,300));est(image,(n+2)/30)
    assert est.detection_frames[-1]==n+2


def test_crop_expansion_clamp_and_loss():
    p=person((50,50),(50,50));p[:,3]=0
    p[[11,12],:2]=[10,20];p[[23,24],:2]=[90,80]
    p[[11,12,23,24],3]=1
    np.testing.assert_allclose(lifter_crop(p,(100,100),1.25),[0,12.5,100,87.5])
    assert not lifter_lost(p,None,(100,100))
    assert lifter_lost(p,(30,30,50,50),(100,100))
    p[[15,16],:2]=[40,40];p[[15,16],3]=1
    assert not lifter_lost(p,(30,30,50,50),(100,100))
    p[[15,16],:2]=[99,99]
    assert lifter_lost(p,(30,30,50,50),(100,100))
    p[[11,12,23,24],3]=.29
    assert lifter_lost(p,None,(100,100))


def test_rep_bench_guard_and_mediapipe_invariance():
    plate=(100,100,120,120);standing=person((110,80),(110,110))
    lying=person((130,140),(125,110),True)
    assert select_lifter([standing],plate,None,'benchpress',in_rep=True) is None
    assert select_lifter([standing],plate,None,'benchpress',in_rep=False)==0
    assert select_lifter([standing,lying],plate,None,'benchpress',in_rep=True)==1
    assert select_lifter([standing],plate,None,'squat',in_rep=True)==0
    assert select_lifter([standing],plate,None,'benchpress',in_rep=True,
                         cfg=replace(DEFAULT.pose,bench_rep_guard=False))==0
    assert select_lifter([standing],plate,None,'benchpress',in_rep=True,
                         cfg=replace(DEFAULT.pose,backend='mediapipe'))==0


def test_capability_not_inferred_from_zero_fingers():
    from feedback.quality import required_joints,observation_reason
    from core.recog.pose import halpe_to_mp33
    p=halpe_to_mp33(np.ones((26,2))*.4,np.ones(26)*.6)
    assert required_joints('bar_proxy_conf','left',has_fingers=False)==[15,16]
    assert required_joints('bar_proxy_conf','left')==[15,16,19,20,21,22]
    assert observation_reason(p,'bar_proxy_conf','left',has_fingers=False) is None
    assert '손가락' in observation_reason(p,'bar_proxy_conf','left')
    p[16,3]=.49
    assert '손목' in observation_reason(p,'bar_proxy_conf','left',has_fingers=False)


def test_proxy_reference_eligibility_is_recovered():
    from feedback.metrics import build_metrics
    from feedback.set_judge import _eligible
    from core.recog.pose import halpe_to_mp33
    p=halpe_to_mp33(np.ones((26,2))*.4,np.ones(26)*.6)
    p[[11,12],:2]=[.4,.3];p[[23,24],:2]=[.4,.7]
    p[[13,14],:2]=[.6,.4];p[[15,16],:2]=[.5,.2]
    total=30
    pose=dict(lms=np.tile(p,(total,1,1)),raw_lms=np.tile(p,(total,1,1)),
              valid=np.ones(total,bool),has_fingers=False)
    positions=dict(zip(('START','MID_DOWN','BOTTOM','MID_UP','LOCKOUT'),(3,8,14,20,26)))
    seg=dict(exercise='benchpress',total=total,fps=30,source='plate',reps=[dict(rep=1,
        side='left',positions=positions,phases=dict(down=[3,14],bottom=[14,16],up=[16,27]))])
    current=build_metrics(pose,seg);legacy=build_metrics(dict(pose,has_fingers=True),seg)
    assert all(v['values']['bar_proxy_conf'] is not None for v in current['reps'][0]['positions'].values())
    assert all(v['values']['bar_proxy_conf'] is None for v in legacy['reps'][0]['positions'].values())
    from feedback.compare import _reference_available_ratio
    from feedback import config
    # Keep the full position x metric denominator; only proxy availability grows.
    denominator = len(config.POSITIONS) * len(config.DISPLAY_METRICS['benchpress'])
    before = _reference_available_ratio(legacy['reps'][0], 'benchpress')
    after = _reference_available_ratio(current['reps'][0], 'benchpress')
    assert after - before == pytest.approx(len(config.POSITIONS) / denominator)
    assert _eligible(current) >= _eligible(legacy)
    assert build_metrics({k:v for k,v in pose.items() if k!='has_fingers'},seg)==legacy


def test_original_cache_compatibility_requires_same_algorithm_and_models():
    from eval.baseline import compatible_original_cache
    old=dict(evaluator_sha256='old',adapter_sha256='same',mode='tracked',video_sha256='video')
    new=dict(old,evaluator_sha256='new',prepared_sha256='video')
    assert not compatible_original_cache(old,new)
    assert compatible_original_cache(old,new,allow_preparation_migration=True)
    assert not compatible_original_cache(old,dict(new,adapter_sha256='different'),allow_preparation_migration=True)
    assert not compatible_original_cache(old,dict(new,mode='tiny_m'),allow_preparation_migration=True)
    assert not compatible_original_cache(old,dict(new,video_sha256='different'),allow_preparation_migration=True)
