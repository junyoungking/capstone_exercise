import importlib.util
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core.auto import v40
from core.auto.judge import _body_frame,_floor_value


def test_v40_features_match_training_preprocessing_when_scores_exist():
    ref=Path('/home/kojy1009/capstone_exercise/new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42/data.py')
    if not ref.is_file():
        import pytest;pytest.skip('local v40 research source unavailable')
    spec=importlib.util.spec_from_file_location('m5_v40_reference_data',ref);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(5);t=np.arange(60)/30.;pose=rng.uniform(10,200,(60,17,3)).astype(np.float32);pose[:,:,2]=rng.uniform(.1,.99,(60,17))
    reset=np.zeros(60,bool);reset[0]=True;reset[31]=True
    expected=module.make_features({'poses':pose,'timestamps':t,'selected_index':np.zeros(60),'selection_reset':reset},{'hz':8.,'joint_confidence':.3,'min_joints':6})
    got=v40.make_features(pose,t,reset)
    for x,y in zip(got,expected[:4]):np.testing.assert_array_equal(x,y)


def test_unavailable_checkpoint_does_not_produce_a_vote(tmp_path):
    assert v40.infer({},[],[],str(tmp_path/'missing.pt'))['exercise'] is None


def test_floor_rejects_offscreen_feet_and_bad_body_order():
    p=np.zeros((33,4));p[:,3]=.9;p[:,0]=.5
    for shoulder,hip,knee,ankle,foot in ((11,23,25,27,31),(12,24,26,28,32)):
        p[shoulder,1]=.2;p[hip,1]=.45;p[knee,1]=.65;p[ankle,1]=.85;p[foot,1]=.9
    plate={'box':[100,600,200,900],'size_px':[100,300]}
    assert _floor_value(p,plate,1000,1000)==0
    p[[27,28,31,32],1]=1.0
    assert _floor_value(p,plate,1000,1000) is None
    p[[27,28,31,32],1]=.9;p[[25,26],1]=.1
    assert _floor_value(p,plate,1000,1000) is None


def test_torso_direction_uses_display_pixels():
    p=np.zeros((33,4));p[:,3]=1;p[[23,24],:2]=[.5,.5];p[[11,12],:2]=[.5,.3]
    assert _body_frame(p,1000,1000)=='standing'
    p[[11,12],:2]=[.25,.5]
    assert _body_frame(p,1000,1000)=='lying'


def test_v40_checkpoint_adapter_uses_only_rep_windows(tmp_path):
    import torch
    from core.auto.v40_model import STGCN_TCN
    path=tmp_path/'v40.pt';torch.save({'model':STGCN_TCN(base=32).state_dict()},path)
    n=40;raw=np.zeros((n,33,4),np.float32);raw[:,:,3]=1
    raw[:,[11,12],:2]=[.5,.3];raw[:,[23,24],:2]=[.5,.6]
    pose={'raw_lms':raw,'raw_scores':np.full((n,17),.9,np.float32),'selection_reset':np.r_[True,np.zeros(n-1,bool)],
          'valid':np.ones(n,bool),'w':1000,'h':1000}
    rows=[{'t':i/20} for i in range(n)]
    got=v40.infer(pose,rows,[(.5,1.)],checkpoint=path,threads=1)
    assert got['status']=='ok' and sum(got['votes'].values())==5
    assert 'probabilities' not in got


def test_competing_reclining_lifter_with_nearby_wrist_is_flagged():
    from feedback.pose import _competing_reclining
    def person(lying):
        p=np.zeros((33,4));p[:,3]=.9;p[:,0]=.5;p[:,1]=.5
        p[[23,24],:2]=[.5,.5];p[[11,12],:2]=[.3,.5] if lying else [.5,.3]
        p[[15,16],:2]=[.55,.5];return p
    standing=person(False);lying=person(True)
    row={'plate':{'used':True,'box':[500,450,600,550]}}
    assert _competing_reclining([standing,lying],standing,row,(1000,1000))
    assert not _competing_reclining([standing],standing,row,(1000,1000))


def test_confirmation_requires_agreeing_classifier_and_unambiguous_lifter(monkeypatch):
    from core.auto.judge import assess
    from core.auto import judge
    n=120;ys=np.r_[np.full(25,300.),np.linspace(300,380,30),np.full(5,380.),np.linspace(380,300,30),np.full(30,300.)]
    rows=[{'seq':i,'t':i/30,'plate':{'used':True,'src':'det','box':[450,float(y-50),550,float(y+50)],'center':[500,float(y)],'size_px':[100,100]}} for i,y in enumerate(ys)]
    p=np.zeros((n,33,4));p[:,:,0]=.5;p[:,:,3]=.9
    for shoulder,hip,knee,ankle,foot in ((11,23,25,27,31),(12,24,26,28,32)):
        p[:,shoulder,1]=.2;p[:,hip,1]=.4;p[:,knee,1]=.6;p[:,ankle,1]=.85;p[:,foot,1]=.9
    pose={'valid':np.ones(n,bool),'raw_lms':p,'lms':p,'w':1000,'h':1000,'selection_ambiguous':np.zeros(n,bool)}
    monkeypatch.setattr(judge.v40,'infer',lambda *a,**k:{'status':'ok','exercise':'squat','votes':{'squat':1}})
    good=assess(rows,pose);assert good['status']=='confirmed' and good['exercise']=='squat'
    pose['selection_ambiguous'][:5]=True
    assert assess(rows,pose)['status']=='needs_confirmation'
    pose['selection_ambiguous'][:]=False
    monkeypatch.setattr(judge.v40,'infer',lambda *a,**k:{'status':'ok','exercise':'deadlift','votes':{'deadlift':1}})
    assert assess(rows,pose)['status']=='needs_confirmation'


def test_confirmed_bench_only_reinfers_if_selection_can_differ():
    from server.compare import reusable_auto_pose
    clear={'selection_ambiguous':np.zeros(8,bool)}
    changed={'selection_ambiguous':np.array([False,True]+[False]*6)}
    assert reusable_auto_pose(clear,'benchpress',8)
    assert not reusable_auto_pose(changed,'benchpress',8)
    assert not reusable_auto_pose({},'benchpress',8)
    assert reusable_auto_pose(changed,'squat',8)
    assert reusable_auto_pose(changed,'deadlift',8)
