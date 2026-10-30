"""Local M3b application. Start with python -m server.app (127.0.0.1 only)."""
import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import threading
import time
import uuid

from fastapi import FastAPI,File,Form,HTTPException,Request,UploadFile
from fastapi.responses import FileResponse,HTMLResponse,JSONResponse,RedirectResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

STAGES=('prepare','plate_user','plate_ref','pose','3d','judge','report')
LIMIT=512*1024*1024
WEB=Path(__file__).parent/'web'


class BodyLimit:
    """Count the actual ASGI stream before multipart spooling, even if chunked."""
    def __init__(self,app,limit):self.app,self.limit=app,limit
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        headers=dict(scope['headers'])
        try: too_large=int(headers.get(b'content-length',b'0'))>self.limit
        except ValueError: too_large=True
        if too_large:
            return await JSONResponse({'detail':'업로드 요청 크기 제한을 넘었습니다.'},413)(scope,receive,send)
        consumed=0;exceeded=False;sent=False
        async def bounded():
            nonlocal consumed,exceeded
            msg=await receive();consumed+=len(msg.get('body',b''))
            if consumed>self.limit:
                exceeded=True
                return {'type':'http.disconnect'}
            return msg
        async def output(msg):
            nonlocal sent
            if exceeded:
                if not sent:
                    sent=True
                    await JSONResponse({'detail':'업로드 요청 크기 제한을 넘었습니다.'},413)(scope,receive,send)
                return
            await send(msg)
        try:await self.app(scope,bounded,output)
        except Exception:
            if not exceeded:raise
        if exceeded and not sent:
            await JSONResponse({'detail':'업로드 요청 크기 제한을 넘었습니다.'},413)(scope,receive,send)


class Store:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=self.root/'index.sqlite3'
        with self.connect() as db:
            for table in ('jobs','sessions','refs'):
                db.execute(f'CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    def connect(self):return sqlite3.connect(self.db,timeout=30)
    def put(self,table,id,payload):
        with self.connect() as db:db.execute(f'INSERT OR REPLACE INTO {table} VALUES (?,?)',(id,json.dumps(payload,ensure_ascii=False,allow_nan=False)))
    def get(self,table,id):
        with self.connect() as db:r=db.execute(f'SELECT payload FROM {table} WHERE id=?',(id,)).fetchone()
        return json.loads(r[0]) if r else None
    def all(self,table):
        with self.connect() as db:rows=db.execute(f'SELECT payload FROM {table} ORDER BY rowid DESC').fetchall()
        return [json.loads(r[0]) for r in rows]


class Worker:
    def __init__(self,store,pipeline,posture_cards):
        self.store,self.pipeline,self.posture_cards=store,pipeline,posture_cards
        self.queue=queue.Queue();self.lock=threading.RLock();self.live={}
        self.thread=threading.Thread(target=self.run,name='analysis-queue',daemon=True)
        # Interrupted jobs fail explicitly; no silent loss or fake completion.
        for job in store.all('jobs'):
            if job['status'] in ('queued','running'):
                job.update(status='failed',error='서버가 중단됐습니다. 다시 시도해 주세요.');store.put('jobs',job['id'],job)
    def start(self):self.thread.start()
    def close(self):self.queue.put(None);self.thread.join()
    def submit(self,job):
        self.store.put('jobs',job['id'],job)
        with self.lock:self.live[job['id']]=job
        self.queue.put(job['id'])
    def status(self,id):
        with self.lock:
            job=self.live.get(id) or self.store.get('jobs',id)
            if not job:return None
            copy=json.loads(json.dumps(job));now=time.monotonic()
            for stage in copy['stages'].values():
                if stage['status']=='running': stage['elapsed_s']=now-stage['started_at']
                stage.pop('started_at',None)
            copy['progress']=sum(s['progress'] for s in copy['stages'].values())/len(copy['stages'])
            return copy
    def run(self):
        while (id:=self.queue.get()) is not None:
            with self.lock:job=self.live[id];job['status']='running'
            dest=self.store.root/'sessions'/id;dest.mkdir(parents=True,exist_ok=True)
            current=None;last_save=0
            def progress(stage,value,**details):
                nonlocal current,last_save
                now=time.monotonic()
                with self.lock:
                    if current!=stage:
                        if current:
                            prior=job['stages'][current]
                            prior.update(status='completed',progress=1.,elapsed_s=now-prior['started_at'])
                        current=stage;job['stage']=stage
                        if stage=='pose' and job['inputs'].get('ref') is None:
                            job['stages']['plate_ref'].update(status='skipped',progress=1.)
                    item=job['stages'][stage]
                    item.setdefault('started_at',now)
                    item.update(progress=max(item['progress'],min(1.,max(0.,value))),status='running',elapsed_s=now-item['started_at'],**details)
                    if value==1:item['status']='cached' if details.get('cached') else 'completed'
                    if now-last_save>.2 or value in (0,1):self.store.put('jobs',id,job);last_save=now
            try:
                args=job['inputs'].copy();args.update(out=dest,progress=progress,ref_cache=self.store.root/'ref_cache',posture_cards=self.posture_cards)
                bundle=self.pipeline(**args)
                now=time.monotonic()
                with self.lock:
                    for item in job['stages'].values():
                        if item['status']=='pending': item.update(status='skipped',progress=1.)
                        elif item.get('started_at') is not None: item['elapsed_s']=max(item['elapsed_s'],now-item['started_at']) if item['status']=='running' else item['elapsed_s']
                    job.update(session_id=id,report_url=f'/sessions/{id}',error=None,ref_cache_hit=bundle.get('ref_cache',{}).get('hit',False))
                speeds=[r['v'] for r in bundle['user']['reps'] if r['v'] is not None]
                session=dict(id=id,date=job['created_at'],exercise=bundle['exercise'],reps=len(bundle['user']['reps']),has_ref=bundle.get('has_ref',bundle.get('ref') is not None),vmax=max(speeds) if speeds else None,report_url=f'/sessions/{id}',ref_id=job.get('ref_id'))
                self.store.put('sessions',id,session)
                with self.lock: job['status']='completed'
            except Exception as exc:
                with self.lock:
                    detail=f'{type(exc).__name__}: {exc}'
                    if isinstance(exc,FileNotFoundError): message='분석에 필요한 파일을 찾을 수 없습니다. 서버 설정을 확인해 주세요.'
                    elif current=='prepare': message='영상을 읽을 수 없습니다. 재생 가능한 영상 파일인지 확인해 주세요.'
                    elif '완료 rep' in str(exc): message='완료한 반복을 찾지 못했습니다. 원판과 동작이 보이는 영상을 선택해 주세요.'
                    else: message='분석을 마치지 못했습니다. 입력을 확인한 뒤 다시 시도해 주세요.'
                    job.update(status='failed',error=message,error_detail=detail)
                    if current:
                        item=job['stages'][current];item.update(status='failed',elapsed_s=time.monotonic()-item['started_at'])
            self.store.put('jobs',id,job)
            self.queue.task_done()


def create_app(data_dir=None,pipeline=None,posture_cards=False,upload_limit=LIMIT):
    if pipeline is None:
        from server.compare import compare
        pipeline=compare
    store=Store(data_dir or Path(__file__).resolve().parents[1]/'data')
    worker=Worker(store,pipeline,posture_cards)
    @asynccontextmanager
    async def lifespan(app):
        worker.start()
        yield
        await asyncio.to_thread(worker.close)
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None)
    app.state.store=store;app.state.worker=worker
    app.add_middleware(BodyLimit,limit=2*upload_limit+1024*1024)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
    @app.middleware('http')
    async def local_only(request,call_next):
        # Binding and request guards also reject cross-origin browser uploads.
        host=request.client.host if request.client else None
        if host not in ('127.0.0.1','::1','testclient'):
            return JSONResponse({'detail':'로컬 연결만 허용합니다.'},403)
        if request.method=='POST':
            origin=request.headers.get('origin')
            if origin and origin!=str(request.base_url).rstrip('/'):
                return JSONResponse({'detail':'같은 앱에서 업로드해 주세요.'},403)
        return await call_next(request)
    @app.get('/')
    def home():return FileResponse(WEB/'app.html',media_type='text/html')
    @app.get('/favicon.ico',include_in_schema=False)
    def favicon():return HTMLResponse('',status_code=204)
    @app.get('/api/sessions')
    def sessions():return store.all('sessions')
    @app.get('/api/refs')
    def refs():return [{k:v for k,v in r.items() if k!='path'} for r in store.all('refs')]
    @app.get('/api/jobs/{id}')
    def job_status(id):
        result=worker.status(id)
        if result is None:raise HTTPException(404,'작업을 찾을 수 없습니다.')
        # Stored input paths stay internal.
        result.pop('inputs',None)
        return result
    @app.post('/api/jobs',status_code=202)
    async def jobs(user:UploadFile=File(...),ref:UploadFile|None=File(None),ref_id:str|None=Form(None),
                   exercise:str=Form(...),plate_mm:float=Form(450),load_kg:float|None=Form(None),height_cm:float|None=Form(None)):
        import math
        if exercise not in ('squat','benchpress','deadlift'):raise HTTPException(422,'종목을 선택해 주세요.')
        if any(v is not None and (not math.isfinite(v) or v<=0) for v in (plate_mm,load_kg,height_cm)):
            raise HTTPException(422,'지름·중량·키는 양수여야 합니다.')
        if ref is not None and ref_id:raise HTTPException(422,'새 기준 영상과 저장된 기준 중 하나만 선택해 주세요.')
        stored=store.get('refs',ref_id) if ref_id else None
        if ref_id and stored is None:raise HTTPException(404,'저장된 기준 영상을 찾을 수 없습니다.')
        id=uuid.uuid4().hex;dest=store.root/'uploads'/id;dest.mkdir(parents=True)
        async def save(upload,path):
            size=0;h=hashlib.sha256()
            try:
                with path.open('wb') as f:
                    while block:=await upload.read(1024*1024):
                        size+=len(block)
                        if size>upload_limit:raise HTTPException(413,f'영상 하나는 {upload_limit//(1024*1024)} MiB까지 올릴 수 있습니다.')
                        h.update(block);f.write(block)
                if size==0:raise HTTPException(422,'빈 영상 파일입니다.')
            finally:await upload.close()
            return h.hexdigest()
        try:
            await save(user,dest/'user.mp4')
            if ref is not None:
                sha=await save(ref,dest/'ref.mp4')
                path=store.root/'refs'/sha/'source.mp4';path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():shutil.copy2(dest/'ref.mp4',path)
                stored=dict(id=sha,name=Path(ref.filename or '기준 영상').name[:200],date=datetime.now().astimezone().isoformat(),path=str(path),sha256=sha)
                store.put('refs',sha,stored);ref_id=sha
            args=dict(user=str(dest/'user.mp4'),ref=stored['path'] if stored else None,exercise=exercise,plate_mm=plate_mm,load_kg=load_kg,height_cm=height_cm)
            job=dict(id=id,status='queued',stage=None,created_at=datetime.now().astimezone().isoformat(),inputs=args,ref_id=ref_id,error=None,
                     stages={k:dict(progress=0.,elapsed_s=0.,status='pending') for k in STAGES})
            worker.submit(job)
            return dict(id=id)
        except Exception:
            shutil.rmtree(dest,ignore_errors=True);raise
    def report_path(id,name):
        if store.get('sessions',id) is None:raise HTTPException(404,'기록을 찾을 수 없습니다.')
        if name not in ('report.html','result_v3.json','user.mp4','ref.mp4'):raise HTTPException(404)
        p=store.root/'sessions'/id/name
        if not p.is_file():raise HTTPException(404)
        return p
    @app.get('/sessions/{id}')
    def report(id):
        # A canonical trailing slash lets offline-style relative asset URLs work.
        report_path(id,'report.html')
        return RedirectResponse(f'/sessions/{id}/',status_code=307)
    @app.get('/sessions/{id}/')
    def report_html(id):return FileResponse(report_path(id,'report.html'),media_type='text/html')
    @app.get('/sessions/{id}/{name}')
    def asset(id,name):return FileResponse(report_path(id,name))
    return app


def main():
    import uvicorn
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port',type=int,default=8000);ap.add_argument('--data',type=Path)
    ap.add_argument('--posture-cards',action='store_true')
    args=ap.parse_args()
    uvicorn.run(create_app(args.data,posture_cards=args.posture_cards),host='127.0.0.1',port=args.port,workers=1)

if __name__=='__main__':main()
