"""Synthetic backend, adapter, selection and feedback compatibility tests."""
from dataclasses import replace
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from core.config import DEFAULT
from core.recog.pose import (halpe_to_mp33, score_visibility, select_lifter,
                             PoseEstimator, RTMPoseEstimator, MediaPipeEstimator)


def person(center, wrist, lying=False):
    p=np.zeros((33,4));p[:,:2]=center;p[:,3]=1
    dx,dy=(40,0) if lying else (0,40)
    p[[11,12],:2]=np.asarray(center)+[dx/2,-dy/2]
    p[[23,24],:2]=np.asarray(center)+[-dx/2,dy/2]
    p[[15,16],:2]=wrist
    return p


def test_adapter_contract():
    points=np.arange(52).reshape(26,2)
    result=halpe_to_mp33(points,np.full(26,.3))
    mp=[0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]
    hp=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,25,20,21]
    np.testing.assert_array_equal(result[mp,:2],points[hp])
    np.testing.assert_allclose(result[mp,3],.5)
    assert (result[[1,3,4,6,9,10,17,18,19,20,21,22]]==0).all()
    assert (result[:,2]==0).all()


def test_visibility_thresholds_and_invalid():
    np.testing.assert_allclose(score_visibility([-.1,.18,.27,.30,.6,1.2,np.nan,np.inf]),
                               [0,.3,.45,.5,1,1,0,0])
    with pytest.raises(ValueError):score_visibility([1],0)
    pts=np.ones((26,2));pts[9,0]=np.nan
    assert halpe_to_mp33(pts,np.ones(26))[15,3]==0


def test_bench_reclining_priority_and_fallback():
    plate=(100,100,120,120)
    standing=person((110,80),(110,110));lifter=person((130,140),(125,110),True)
    assert select_lifter([standing,lifter],plate,None,'benchpress')==1
    assert select_lifter([standing,lifter],plate,None,'squat')==0
    assert select_lifter([standing],plate,None,'benchpress')==0
    diagonal=person((130,140),(125,110))
    diagonal[[11,12],:2]=[150,120]
    diagonal[[23,24],:2]=[110,160]  # exactly 45 degrees: no priority
    assert select_lifter([standing,diagonal],plate,None,'benchpress')==0
    assert select_lifter([standing,lifter],plate,None,'benchpress',
                         cfg=replace(DEFAULT.pose,backend='mediapipe'))==0


def test_continuity_wrist_margin_and_relative_improvement():
    plate=(100,100,120,120);prev=np.array([100,120])
    old=person(prev,(140,110));weak=person((200,120),(135,110))
    assert select_lifter([old,weak],plate,prev,'squat')==0
    strong=person((200,120),(120,110))
    assert select_lifter([old,strong],plate,prev,'squat')==1
    assert select_lifter([strong],plate,prev,'squat') is None
    strong[[15,16],3]=0
    assert select_lifter([old,strong],plate,prev,'squat')==0
    # Absolute improvement passes (.6) but relative improvement fails (15%).
    old=person(prev,(190,110));new=person((200,120),(178,110))
    assert select_lifter([old,new],plate,prev,'squat')==0
    # Relative improvement passes (60%) but absolute improvement fails (.3).
    old=person(prev,(120,110));new=person((200,120),(114,110))
    assert select_lifter([old,new],plate,prev,'squat')==0
    # Exactly .5 plate heights and 25% closer is sufficient.
    old=person(prev,(150,110));new=person((200,120),(140,110))
    assert select_lifter([old,new],plate,prev,'squat')==1


def test_backend_switch_all_people_and_empty_detection(monkeypatch):
    calls=[]
    def detect(image):calls.append('detect');return np.zeros((3,4))
    def pose(image,bboxes):
        calls.append(('pose',len(bboxes)))
        scores=np.full((3,26),.3);scores[:,0]=.9
        return np.ones((3,26,2)),scores
    def body(**kw):
        assert kw['backend']=='onnxruntime' and kw['device']=='cpu'
        return SimpleNamespace(det_model=detect,pose_model=pose)
    monkeypatch.setitem(sys.modules,'rtmlib',SimpleNamespace(BodyWithFeet=body))
    estimator=PoseEstimator(cfg=replace(DEFAULT.pose,mode='balanced'))
    assert isinstance(estimator,RTMPoseEstimator)
    people=estimator(np.zeros((40,60,3),np.uint8),0)
    assert len(people)==3
    assert estimator.frame_scores[0][0]==pytest.approx(.9) and people[0][0,3]==1.
    assert calls==['detect',('pose',3)] and estimator.every_n==1
    monkeypatch.setattr(MediaPipeEstimator,'__init__',lambda self,**kw:None)
    assert isinstance(PoseEstimator(cfg=replace(DEFAULT.pose,backend='mediapipe')),MediaPipeEstimator)
    with pytest.raises(ValueError):PoseEstimator(cfg=replace(DEFAULT.pose,backend='bad'))
    estimator.model.det_model=lambda image:[]
    assert estimator(np.zeros((40,60,3),np.uint8),0)==[]


def test_missing_fingers_feedback_behavior():
    import unified_feedback_v4 as v4
    from feedback.quality import observation_reason
    p=halpe_to_mp33(np.ones((26,2))*.4,np.ones(26)*.6)
    grip,conf=v4.hand_grip_point(p,'left')
    np.testing.assert_allclose(grip,p[15,:2]);assert conf==1
    assert '손가락' in observation_reason(p,'bar_proxy_conf','left')
    assert observation_reason(p,'elbow_angle','left') is None
    # Advisory proxy is still counted by the legacy reference availability gate.
    from copy import deepcopy
    from feedback import config as fb_config
    from feedback.gate import _reference_available_ratio as gate_ratio
    from feedback.compare import _reference_available_ratio as compare_ratio
    from feedback.set_judge import _eligible
    rep={'rep':1,'excluded':False,'positions':{
        pos:{'values':{metric:(90. if metric=='elbow_angle' else
                              .1 if metric=='wrist_elbow_x_diff' else
                              .8 if metric=='bar_proxy_conf' else None)
                       for metric in fb_config.DISPLAY_METRICS['benchpress']}}
        for pos in fb_config.POSITIONS}}
    missing=deepcopy(rep)
    for position in missing['positions'].values():position['values']['bar_proxy_conf']=None
    for ratio in (gate_ratio,compare_ratio):
        assert ratio(rep,'benchpress')==pytest.approx(.6)
        assert ratio(missing,'benchpress')==pytest.approx(.4)
    assert _eligible({'exercise':'benchpress','reps':[rep]})=={1}
    assert _eligible({'exercise':'benchpress','reps':[missing]})==set()
    smoother=v4.LandmarkSmoother()
    smoother.update(p)
    assert (smoother.update(p)[[1,3,4,6,9,10,17,18,19,20,21,22],3]==0).all()
    from core.recog.pose import SideLock
    # Losing asymmetric finger support can change the initial side decision.
    fingers=p.copy();fingers[[11,13,15],3]=.6;fingers[[12,14,16],3]=.5
    fingers[[20,22],3]=1;fingers[[20,22],:2]=fingers[16,:2]
    missing=fingers.copy();missing[17:23]=0
    for cls in (v4.SideLock,SideLock):
        assert cls('benchpress').update(fingers)=='right'
        assert cls('benchpress').update(missing)=='left'
    # The grip intentionally falls back from a three-point average to wrist.
    fingers=p.copy();fingers[[19,21],3]=1
    fingers[19,:2]=[.6,.4];fingers[21,:2]=[.8,.4]
    np.testing.assert_allclose(v4.hand_grip_point(fingers,'left')[0],[.6,.4])
    np.testing.assert_allclose(v4.hand_grip_point(p,'left')[0],[.4,.4])
    lock=v4.SideLock('benchpress');core_lock=SideLock('benchpress')
    assert lock.update(p)==core_lock.update(p)=='left'
    p[[11,13,15],3]=.1
    for _ in range(6):
        side=lock.update(p)
        assert core_lock.update(p)==side
    assert side=='right'


def test_third_candidate_is_not_truncated():
    from feedback.pose import choose_plate_candidate
    a=person((200,200),(160,110));b=person((250,200),(150,110))
    c=person((110,150),(110,110))
    candidates=[p/np.array([400,300,1,1]) for p in (a,b,c)]
    selected,_=choose_plate_candidate(candidates,{'plate':{'used':True,'box':[100,100,120,120]}},
                                      None,'squat',(400,300))
    assert selected is candidates[2]


def test_cache_identity_includes_backend_mode_and_score_mapping(monkeypatch,tmp_path):
    from feedback import pose as module
    import export_overlay
    calls=[]
    monkeypatch.setattr(export_overlay,'video_info',lambda p:{'fps':30,'w':640,'h':480})
    monkeypatch.setattr(module,'extract_backend_sequence',lambda *args:calls.append(args[-1]) or [None])
    monkeypatch.setattr(module,'extract_plate_sequence',lambda *args:calls.append('mp') or [None])
    video=tmp_path/'one.mp4';video.touch();cache=tmp_path/'pose.npz'
    rows=[{'seq':0}]
    module.extract_pose(video,cache,'squat',plate_rows=rows)
    assert module.extract_pose(video,cache,'squat',plate_rows=rows)['cache_hit']
    for cfg in (replace(DEFAULT.pose,mode='balanced'),replace(DEFAULT.pose,score_reference=.7),
                replace(DEFAULT.pose,backend='mediapipe')):
        assert not module.extract_pose(video,cache,'squat',plate_rows=rows,pose_cfg=cfg)['cache_hit']
    assert len(calls)==4


@pytest.mark.parametrize('backend,mode',[('rtmpose','lightweight'),('rtmpose','balanced'),('mediapipe','lightweight')])
def test_server_cli_routes_backend_and_mode_and_closes(monkeypatch,backend,mode):
    from server import jobs
    seen={}
    estimator=SimpleNamespace(close=lambda:seen.update(closed=True))
    def factory(*,cfg):
        seen['cfg']=cfg
        return estimator
    def process(*args,**kwargs):
        assert kwargs['pose'] is estimator
        return {'sets':[]},[],None
    monkeypatch.setattr(jobs,'PoseEstimator',factory)
    monkeypatch.setattr(jobs,'process_video',process)
    monkeypatch.setattr(sys,'argv',['jobs','fake.mp4','--exercise','squat','--pose',
                                  '--pose-backend',backend,'--pose-mode',mode])
    jobs.main()
    assert (seen['cfg'].backend,seen['cfg'].mode)==(backend,mode)
    assert seen['closed']
