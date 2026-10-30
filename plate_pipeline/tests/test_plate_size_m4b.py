import gzip,json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from core.geometry import (plate_geometry,stature,box_angle_reliable,implied_plate_diameter,plate_suggestion)
from core.io.rows import write_rows
from server.app import create_app,Store
from server.compare import write_report
from test_result_v3 import product
from test_upload import _plate_y


def test_stature_inversion_round_trip_and_prior_range():
    # Constant height makes the M2 forward and per-frame inverse exact.
    n=4;raw=np.ones((n,33,4),float);raw[:,0,1]=50
    for j in (27,28,29,30,31,32):raw[:,j,1]=450
    boxes=np.tile([200.,200.,270.,300.],(n,1));stand=np.ones(n,bool)
    g=plate_geometry(boxes,(640,480),450);height,_=stature(raw,g,stand)
    got=implied_plate_diameter(raw,boxes,(640,480),stand,height*100,True)
    assert got==pytest.approx(450,abs=1e-9)
    lo=implied_plate_diameter(raw,boxes,(640,480),stand,160,True)
    hi=implied_plate_diameter(raw,boxes,(640,480),stand,180,True)
    assert lo<hi
    assert implied_plate_diameter(raw,boxes,(640,480),np.zeros(n,bool)) is None


def test_threshold_angle_and_clipping_rules():
    assert not plate_suggestion(315,450) and plate_suggestion(314.9,450)
    assert not plate_suggestion(585,450) and plate_suggestion(585.1,450)
    assert not box_angle_reliable([[100,100,195,200]],(640,480))
    assert box_angle_reliable([[100,100,194,200]],(640,480))
    assert not box_angle_reliable([[0,100,80,200]],(640,480))
    raw=np.ones((1,33,4));raw[:,0,1]=50;raw[:,[27,28,29,30,31,32],1]=450
    assert implied_plate_diameter(raw,[[0,100,80,200]],(640,480),[True]) is None


def session_fixture(tmp_path):
    src=Path(__file__).resolve().parents[1]/'docs/midterm/src/bundle3.json.gz'
    with gzip.open(src,'rt') as f:b=product(json.load(f))
    b['has_ref']=False;b['ref']=None;b['inputs']['height_cm']=None
    n=420;pose=np.ones((n,33,4),float);pose[:,:,:2]=.5;pose[:,:,3]=.8
    b['user']['j3']=[None]*n;b['user']['pose3d_warnings']=[]
    path=tmp_path/'sessions'/'synthetic';dest=path/'analysis'/'user';(dest/'plate').mkdir(parents=True)
    rows=[]
    for i in range(n):
        y=_plate_y(i/30);rows.append({'seq':i,'t':i/30,'cam':[0,0], 'plate':{'used':True,'src':'det','center':[160.,y], 'size_px':[80.,80.],'box':[120.,y-40,200.,y+40]}})
    write_rows(dest/'plate'/'rows.jsonl.gz',rows)
    (dest/'plate'/'result.json').write_text(json.dumps({'fps':30.,'exercise':'squat','video':'synthetic'}))
    np.savez(dest/'pose.npz',raw_lms=pose,lms=pose,valid=np.ones(n,bool),fps=30.,w=320,h=480,total=n)
    (path/'result_v3.json').write_text(json.dumps(b));write_report(b,path)
    store=Store(tmp_path);store.put('sessions','synthetic',{'id':'synthetic','reps':5,'vmax':1.,'has_ref':False})
    return path


def test_rescale_api_recounts_archives_and_does_not_infer(tmp_path,monkeypatch):
    p=session_fixture(tmp_path);before=(p/'result_v3.json').read_bytes();old_report=(p/'report.html').read_bytes()
    import server.compare as compare
    monkeypatch.setattr(compare,'load_model',lambda *a,**k:pytest.fail('3D inference called'))
    monkeypatch.setattr(compare,'lift',lambda *a,**k:pytest.fail('3D inference called'))
    import server.jobs as jobs
    monkeypatch.setattr(jobs,'process_video',lambda *a,**k:pytest.fail('plate detection called'))
    import feedback.pose as pose
    monkeypatch.setattr(pose,'extract_pose',lambda *a,**k:pytest.fail('pose inference called'))
    with TestClient(create_app(tmp_path,pipeline=lambda **k:None)) as c:
        r=c.post('/api/sessions/synthetic/rescale',json={'plate_mm_user':50});assert r.status_code==200,r.text
        assert r.json()['reps']==0
        archived=p/'history'/r.json()['version'];assert (archived/'result_v3.json').read_bytes()==before and (archived/'report.html').read_bytes()==old_report
        b=json.loads((p/'result_v3.json').read_text());assert not b['user']['reps'] and b['inputs']['plate_mm']==50
        geo=b['user']['geo'];check=b['user']['geo_check']
        assert geo['angle_reliable'] is False and geo['retake_warn'] is False
        assert set(('plate_mm_implied','plate_mm_range','plate_mm_basis','plate_suggest')) <= set(check)
        assert len(check['plate_mm_range'])==2 and check['plate_mm_basis']=='prior_170' and isinstance(check['plate_suggest'],bool)
        assert c.get('/api/sessions').json()[0]['reps']==0
        r2=c.post('/api/sessions/synthetic/rescale',json={'plate_mm_user':450});assert r2.status_code==200,r2.text
        assert r2.json()['reps']==3
        assert c.post('/api/sessions/synthetic/rescale',json={'plate_mm_ref':400}).status_code==422
        assert c.post('/api/sessions/synthetic/rescale',json={'plate_mm_user':-1}).status_code==422
        assert c.post('/api/sessions/unknown/rescale',json={'plate_mm_user':450}).status_code==404


def test_reference_and_user_diameters_are_independent(tmp_path):
    import shutil
    from copy import deepcopy
    p=session_fixture(tmp_path);b=json.loads((p/'result_v3.json').read_text())
    b['ref']=deepcopy(b['user']);b['has_ref']=True;b['inputs']['height_cm']=175
    (p/'result_v3.json').write_text(json.dumps(b));write_report(b,p)
    shutil.copytree(p/'analysis'/'user',p/'analysis'/'ref')
    store=Store(tmp_path);item=store.get('sessions','synthetic');item['has_ref']=True;store.put('sessions','synthetic',item)
    with TestClient(create_app(tmp_path,pipeline=lambda **k:None)) as c:
        r=c.post('/api/sessions/synthetic/rescale',json={'plate_mm_ref':300});assert r.status_code==200,r.text
        out=json.loads((p/'result_v3.json').read_text())
        assert out['inputs']['plate_mm']==450 and out['inputs']['plate_mm_ref']==300
        assert out['user']['geo_check']['plate_mm_basis']=='input_height'
        assert out['ref']['geo_check']['plate_mm_basis']=='prior_170'
        user=out['user']['depth']
        r=c.post('/api/sessions/synthetic/rescale',json={'plate_mm_ref':400});assert r.status_code==200,r.text
        out=json.loads((p/'result_v3.json').read_text());assert out['user']['depth']==user
