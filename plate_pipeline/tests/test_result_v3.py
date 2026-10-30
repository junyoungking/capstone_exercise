"""Frozen JS parity and measured report data contract."""
from copy import deepcopy
import gzip,json
from pathlib import Path
import numpy as np
import pytest
from server.findings import calculate,judge,reps_showing
from server.schema import validate_result
SRC=Path(__file__).resolve().parents[1]/'docs/midterm/src'
@pytest.fixture
def bundle():
    with gzip.open(SRC/'bundle3.json.gz','rt') as f: return json.load(f)

def test_frozen_js_findings_all_fields(bundle):
    expected=json.loads((SRC/'findings_v3_expected.json').read_text());got,_=calculate(bundle,posture_cards=True)
    for group in ('cards','extra'):
        assert len(got[group])==len(expected[group])
        for actual,wanted in zip(got[group],expected[group]):
            for key in ('no','kind','key','pos','reps','all','best','text','val'): assert actual[key]==wanted[key],(group,key)
            assert actual['score']==pytest.approx(wanted['score'],abs=1e-3)
    assert got['outliers']==expected['outliers']
    assert got['per_rep_angles']==expected['per_rep_angles']
    for role in ('user','ref'):
        for actual,wanted in zip(got['pos_times'][role],expected['pos_times'][role]): assert actual==pytest.approx(wanted,abs=1e-10)

def test_nearest_reference_and_strict_threshold():
    assert not judge([20]*3,[12]*3,8)['hit']
    assert judge([20.01]*3,[12]*3,8)['hit']
    reps=[{'n':1},{'n':2}]
    assert reps_showing(reps,[30,40],[10,25],1,6)==[2]
    assert reps_showing(reps,[0,5],[10,25],-1,6)==[1]
    assert not judge([None,float('nan')],[10],8)['hit']

def product(bundle,exercise='squat',height=None):
    bundle.update(schema_version=3,exercise=exercise,inputs=dict(plate_mm=450,load_kg=None,height_cm=height),show_3d=exercise!='benchpress',pose3d_scale='model_native' if height is None else 'input_height')
    for role in ('user','ref'):
        bundle[role]['geo_check']=dict(stature_plate_cm=bundle[role]['geo']['stature_cm'],warn=False)
        bundle[role]['geo']['stature_cm']=height
    bundle['findings'],bundle['angle_judgments']=calculate(bundle,posture_cards=True)
    bundle['kpi']=bundle['angle_judgments']['FAR']['knee']
    return bundle

@pytest.mark.parametrize('height',[None,175])
@pytest.mark.parametrize('exercise',['squat','benchpress','deadlift'])
def test_schema_keys_types(bundle,height,exercise):
    b=product(bundle,exercise,height);assert validate_result(b) is b
    bad=deepcopy(b);bad['user']['kp2'][0]=[[1,2]]
    with pytest.raises(ValueError,match='kp2'): validate_result(bad)
    bad=deepcopy(b);del bad['inputs']['height_cm']
    with pytest.raises(ValueError,match='inputs'): validate_result(bad)
    bad=deepcopy(b);bad['user']['reps'][0]['rom_cm']='fake'
    with pytest.raises(ValueError,match='rep'): validate_result(bad)

def test_height_scale_and_reference_match():
    from core.geometry.pose3d import scale_floor
    user=np.zeros((3,17,3));user[:,9,1]=1.;ref=user*1.5;stand=np.ones(3,bool)
    u,_,why=scale_floor(user,stand,1.75);r,_,_=scale_floor(ref,stand,1.75)
    assert why is None and u[:,9,1]==pytest.approx([1.75*.92]*3)
    assert r==pytest.approx(u)
    native,scale,reason=scale_floor(user,stand,None)
    assert scale==1 and reason is not None and native==pytest.approx(user)

def test_offline_report_reads_python_values(bundle,tmp_path):
    from server.compare import write_report
    write_report(product(bundle),tmp_path);html=(tmp_path/'report.html').read_text()
    assert 'const FINDINGS = () => B.findings' in html and 'kn = B.kpi' in html
    assert 'function judge(' not in html and 'function prepReps(' not in html
    assert '<script src=' not in html and '<link ' not in html
    assert 'vu.src = "user.mp4"; if (HAS_REF) vr.src = "ref.mp4"' in html
    assert 'id="s-upload"' not in html and 'const SESSIONS' not in html
    assert '종목 자동 인식' not in html and 'B.show_3d' in html

@pytest.mark.parametrize('input_height',[None,175.])
@pytest.mark.parametrize('standing',[True,False])
def test_pipeline_3d_height_is_independent_of_plate(monkeypatch,input_height,standing):
    from server import compare as c
    # H36M skeleton in camera coordinates, with all 16 tree edges nonzero.
    p=np.array([[0,0,0],[.1,0,0],[.1,.5,0],[.1,1,0],[-.1,0,0],[-.1,.5,0],[-.1,1,0],
                [0,-.2,0],[0,-.4,0],[0,-.65,0],[0,-.75,0],[-.2,-.4,0],[-.35,-.2,0],[-.4,0,0],
                [.2,-.4,0],[.35,-.2,0],[.4,0,0]],float)
    native=np.repeat(p[None],3,axis=0)
    raw=np.ones((3,33,4));raw[:,:,0]=.5;raw[:,:,1]=.5
    pose=dict(raw_lms=raw,w=640,h=480,fps=30,valid=np.ones(3,bool))
    monkeypatch.setattr(c,'standing_frames',lambda _:np.full(3,standing))
    monkeypatch.setattr(c,'lift',lambda *args:(native.copy(),None,None))
    # No plate data is read for scaling or shared reference height.
    result=dict(sets=[])
    _,_,u,actual,scaled,_=c.aligned_3d(pose,[],result,None,input_height)
    assert np.isfinite(u).all()
    assert scaled==(input_height is not None)
    if input_height is not None: assert actual==pytest.approx(input_height)
    native*=1.4
    _,_,r,ref_height,_,_=c.aligned_3d(pose,[],result,None,actual)
    assert ref_height==pytest.approx(actual)
    # Relative pose and supplied height, rather than model's different native scale.
    assert r==pytest.approx(u)

def test_unavailable_speed_is_not_a_fake_zero_or_dropped_rep(bundle):
    b=product(bundle)
    count=len(b['user']['reps'])
    b['user']['reps'][-1]['v']=None
    b['findings'],_=calculate(b)
    assert len(b['user']['reps'])==count
    assert not any(c['kind']=='inset' for c in b['findings']['cards'])
    assert validate_result(b) is b


def test_ellipse_schema_and_retake_boundary(bundle):
    from core.geometry import retake_warning
    assert not retake_warning(60.) and retake_warning(60.001)
    b=product(bundle)
    v=b['user'];v['geo'].update(cam_angle=61.,angle_source='ellipse',retake_warn=True,
                               ellipse_stats=dict(count=1,fraction=.1,major_px=100.,minor_px=40.,tilt_deg=0.))
    v['ellipse']=[None]*len(v['t'])
    v['ellipse'][0]=dict(source='ellipse',quality=.8,center=[20.,30.],major=100.,minor=40.,angle=0.)
    assert validate_result(b) is b
    v['geo']['retake_warn']=False
    with pytest.raises(ValueError,match='angle'): validate_result(b)
