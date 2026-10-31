"""HTTP/queue contracts with a deterministic fake only inside tests."""
import json,time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from server.app import create_app,STAGES


def wait(client,id):
    samples=[]
    for _ in range(1000):
        job=client.get('/api/jobs/'+id).json();samples.append(job)
        if job['status'] in ('completed','failed'):return job,samples
        time.sleep(.002)
    raise AssertionError('job did not finish')


class Pipeline:
    def __init__(self):self.calls=[];self.cache=set();self.active=0;self.max_active=0
    def __call__(self,user,ref,exercise,out,progress,ref_cache,posture_cards,**kw):
        self.active+=1;self.max_active=max(self.active,self.max_active)
        try:
            self.calls.append(dict(user=user,ref=ref,posture_cards=posture_cards,**kw))
            progress('prepare',0)
            if Path(user).read_bytes()==b'broken':raise ValueError('영상을 읽을 수 없습니다')
            key=Path(ref).read_bytes() if ref else None;hit=key is not None and key in self.cache
            for stage in STAGES:
                if stage=='plate_ref' and ref is None:continue
                for n in range(4):
                    progress(stage,n/3,processed=n,total=3,cached=stage=='plate_ref' and hit)
                    time.sleep(.001)
            if ref:self.cache.add(key)
            out=Path(out);out.mkdir(exist_ok=True,parents=True)
            (out/'report.html').write_text('<h1>measured report</h1>')
            (out/'user.mp4').write_bytes(Path(user).read_bytes())
            result=dict(exercise=exercise,user=dict(reps=[dict(v=.5),dict(v=None)]),has_ref=ref is not None,ref_cache=dict(hit=hit))
            (out/'result_v3.json').write_text(json.dumps(result))
            return result
        finally:self.active-=1


def submit(client,ref=False,**data):
    files={'user':('user.mp4',b'video','video/mp4')}
    if ref:files['ref']=('ref.mp4',b'reference','video/mp4')
    return client.post('/api/jobs',files=files,data=dict(exercise='squat',**data))


def test_job_lifecycle_progress_queue_sessions_refs_cache(tmp_path):
    pipeline=Pipeline()
    app=create_app(tmp_path,pipeline=pipeline)
    with TestClient(app) as c:
        assert c.get('/api/sessions').json()==[]
        first=submit(c,True,height_cm=175).json()['id'];second=submit(c).json()['id']
        done,samples=wait(c,first);assert done['status']=='completed'
        for stage in STAGES:
            seq=[s['stages'][stage]['progress'] for s in samples]
            assert seq==sorted(seq)
        seq=[s['progress'] for s in samples];assert seq==sorted(seq)
        assert wait(c,second)[0]['status']=='completed'
        assert pipeline.max_active==1
        refs=c.get('/api/refs').json();assert len(refs)==1 and 'path' not in refs[0]
        third=submit(c,ref_id=refs[0]['id']).json()['id'];job,_=wait(c,third)
        assert job['ref_cache_hit'] is True
        assert job['stages']['plate_ref']['cached'] is True
        assert len(c.get('/api/sessions').json())==3
        assert c.get('/sessions/'+first).status_code==200
        assert c.get('/sessions/'+first+'/user.mp4',headers={'range':'bytes=0-1'}).status_code==206
        assert c.get('/sessions/'+first+'/analysis').status_code==404
        assert pipeline.calls[0]['height_cm']==175
        assert all(x['posture_cards'] is False for x in pipeline.calls)
        assert c.get('/api/jobs/unknown').status_code==404
    # Persisted sessions and refs survive an application restart.
    with TestClient(create_app(tmp_path,pipeline=Pipeline())) as c:
        assert len(c.get('/api/sessions').json())==3
        assert len(c.get('/api/refs').json())==1


def test_failures_optional_reference_validation_and_retry(tmp_path):
    p=Pipeline()
    with TestClient(create_app(tmp_path,pipeline=p,posture_cards=True,upload_limit=16)) as c:
        bad=c.post('/api/jobs',data={'exercise':'squat'},files={'user':('x.mp4',b'broken')}).json()['id']
        job,_=wait(c,bad);assert job['status']=='failed' and '영상을 읽을 수 없습니다' in job['error']
        assert c.get('/api/sessions').json()==[]
        assert c.get('/sessions/'+bad).status_code==404
        result=submit(c);assert result.status_code==202
        done,_=wait(c,result.json()['id']);assert done['status']=='completed'
        assert p.calls[-1]['ref'] is None and p.calls[-1]['posture_cards'] is True
        assert c.get('/api/sessions').json()[0]['has_ref'] is False
        assert submit(c,True,ref_id='nope').status_code==422
        assert submit(c,ref_id='nope').status_code==404
        assert submit(c,height_cm=-5).status_code==422
        assert c.post('/api/jobs',data={'exercise':'auto'},files={'user':('x',b'video')}).status_code==422
        assert c.post('/api/jobs',data={'exercise':'squat'},files={'user':('x',b'')}).status_code==422
        assert c.post('/api/jobs',data={'exercise':'squat'},files={'user':('x',b'x'*17)}).status_code==413
        assert c.post('/api/jobs',content=b'x'*(1024*1024+33)).status_code==413
        assert submit(c,**{}).status_code==202
        assert c.post('/api/jobs',headers={'Origin':'https://elsewhere.example'},data={'exercise':'squat'},files={'user':('x',b'video')}).status_code==403


def test_posture_toggle_and_single_set_findings():
    import gzip
    from server.findings import calculate
    from server.compare import clean
    from server.schema import validate_result
    src=Path(__file__).resolve().parents[1]/'docs/midterm/src/bundle3.json.gz'
    with gzip.open(src,'rt') as f:b=json.load(f)
    f,_=calculate(b,posture_cards=False)
    assert not any(c['key'] in ('knee','hip','trunk') for c in f['cards']+f['extra'])
    f,_=calculate(b,posture_cards=True)
    assert any(c['key']=='knee' for c in f['cards'])
    b['ref']=None;b['has_ref']=False;b['sentences']=[]
    f,_=calculate(b)
    assert 'ref' not in f['pos_times']
    assert all(c['kind']!='ref' for c in f['cards'])


def test_cache_key_ignores_display_inputs_and_invalidates_inference(tmp_path,monkeypatch):
    from server import cache
    # No models are loaded. Hash configuration and source/code bytes are real.
    video=tmp_path/'video';video.write_bytes(b'original')
    model=tmp_path/'model';model.write_bytes(b'weights')
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'3d weights')
    k,p=cache.cache_key(video,'squat',450,model,checkpoint)
    assert cache.cache_key(video,'squat',450,model,checkpoint)==(k,p)
    assert cache.cache_key(video,'squat',450,model,checkpoint,recover_early_back=True)[0] != cache.cache_key(video,'squat',450,model,checkpoint,recover_early_back=False)[0]
    assert cache.cache_key(video,'squat',450,model,checkpoint,plate_pick=True)[0] != cache.cache_key(video,'squat',450,model,checkpoint,plate_pick=False)[0]
    assert cache.cache_key(video,'squat',450,model,checkpoint,plate_pick=True,plate_pick_weak=True)[0] != cache.cache_key(video,'squat',450,model,checkpoint,plate_pick=True,plate_pick_weak=False)[0]
    assert cache.cache_key(video,'squat',400,model,checkpoint)[0]!=k
    assert cache.cache_key(video,'deadlift',450,model,checkpoint)[0]!=k
    model.write_bytes(b'changed weights');assert cache.cache_key(video,'squat',450,model,checkpoint)[0]!=k
    assert not cache.available(tmp_path/'absent')


def test_chunked_body_limit_without_content_length(tmp_path):
    app=create_app(tmp_path,pipeline=Pipeline(),upload_limit=16)
    with TestClient(app) as c:
        def stream():
            yield b'a'*(600*1024)
            yield b'b'*(600*1024)
        response=c.post('/api/jobs',content=stream(),headers={'content-type':'multipart/form-data; boundary=boundary','transfer-encoding':'chunked'})
        assert response.status_code==413
        assert c.get('/api/sessions').json()==[]


def test_interrupted_jobs_fail_on_restart(tmp_path):
    from server.app import Store
    store=Store(tmp_path)
    store.put('jobs','stopped',dict(id='stopped',status='running',stage='pose',error=None,stages={k:dict(progress=0.,status='pending',elapsed_s=0.) for k in STAGES}))
    with TestClient(create_app(tmp_path,pipeline=Pipeline())) as c:
        job=c.get('/api/jobs/stopped').json()
        assert job['status']=='failed' and '서버가 중단' in job['error']
