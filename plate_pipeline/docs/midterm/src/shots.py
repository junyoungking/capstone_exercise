"""Real-video midterm captures. No plots; all extra labels use PIL/Noto Sans CJK KR.

python plate_pipeline/docs/midterm/src/shots.py --user /path/squat_user_set.mp4 --ref /path/stu10_70.mp4
Generated recognition/cache/REPORT/ZIP/browser files remain in eval_out/midterm_shots.
Only this script and PNGs in docs/midterm/shots are intended for commit.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import functools
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[4]
PIPE = ROOT/'plate_pipeline'
TEAL, GREY, WHITE, DARK = '#17A08F', '#9AA3AE', '#FFFFFF', '#0E1116'


def rows_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def setup_core():
    sys.path.insert(0, str(PIPE))
    import cv2
    import onnxruntime as ort
    cv2.setNumThreads(1)
    original = ort.InferenceSession
    def session(*a, **kw):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        return original(*a, sess_options=options, **kw)
    ort.InferenceSession = session


class AllPoses:
    """Unchanged pipeline estimator, with every candidate retained for selection evidence."""
    def __init__(self):
        from core.recog.pose import PoseEstimator
        self.estimator = PoseEstimator()
        self.cfg, self.every_n = self.estimator.cfg, self.estimator.every_n
        self.records = []
    def __call__(self, image, t):
        candidates = self.estimator(image, t)
        self.records.append({'t': round(t, 4), 'candidates': [p.tolist() for p in candidates]})
        return candidates
    def close(self):
        self.estimator.close()


def infer(args, name, path, exercise):
    """--model + --pose + debug observations, with no diagnostic graphs generated."""
    from core.io.rows import read_rows
    from server.jobs import process_video
    from eval.debug_track import diagnose
    dest = args.output/'runs'/name
    signature = {'video': digest(path), 'model': digest(args.model),
                 'code': {f: digest(PIPE/f) for f in ['server/jobs.py','core/config.py',
                          'core/recog/plate.py','core/recog/track.py','core/recog/pose.py',
                          'core/signal/height.py','core/signal/final.py','core/judge/state.py']}}
    cache = dest/'signature.json'
    if not args.refresh and cache.exists() and json.loads(cache.read_text()) == signature:
        result = json.loads((dest/'result.json').read_text())
        rows = read_rows(dest/'rows.jsonl.gz')
        with gzip.open(dest/'all_poses.json.gz','rt') as f: poses=json.load(f)
    else:
        pose=AllPoses()
        try:
            result, rows, _ = process_video(path,exercise,model=args.model,pose=pose,
                                           out_dir=dest,device='auto')
        finally:
            pose.close()
        poses=pose.records
        with gzip.open(dest/'all_poses.json.gz','wt') as f:json.dump(poses,f)
        # Candidate/debug rows are produced by recognize; diagnostics mirror --debug.
        # No timeline/height plot is created for this screenshot-only task.
        dump(dest/'diagnostics.json', diagnose(rows,result))
        dump(cache,signature)
    print(f"인식 {name}: {sum(s['reps'] for s in result['sets'])}회, 포즈 {len(poses)}프레임, {result['detector_providers']}",flush=True)
    return {'name':name,'path':str(path),'exercise':exercise,'result':result,'rows':rows,'poses':poses}


def final_and_reps(run):
    from core.signal.height import height_signal
    from core.signal.final import clean_final
    from core.judge.session import run_session
    t,y,x=height_signal(run['rows'],run['result']['plate_mm'])
    final=clean_final(t,y,x,run['result']['fps'])
    s=final.sig
    sets=run_session(s.t,s.y,s.x,s.gap_before,run['exercise'])
    reps=[tr for st in sets for tr in st.reps]
    values=[v for st in run['result']['sets'] for v in st['rep_values']]
    return final,reps,values


def candidates(args):
    plate=rows_csv(PIPE/'results/midterm/per_video.csv')
    baseline={r['key']:r for r in rows_csv(args.v40/'arms/B_generic/all_test_per_video.csv')}
    answer=[]
    for p in plate:
        b=baseline[p['key']]
        if int(p['pred_count'])==int(p['gt_count']) and int(b['pred_count'])!=int(p['gt_count']):
            answer.append({'key':p['key'],'exercise':p['type'],'truth':int(p['gt_count']),
                           'pose':int(b['pred_count']),'plate':int(p['pred_count'])})
    return answer


def prepare(args):
    from eval.counts import resolve_video
    jobs=[('reference',args.ref,'squat'),('user',args.user,'squat')]
    team=args.local_root/'plate_pipeline/data/team/test9'
    for name,ex in [('stu2_10_edited','bench'),('stu7_1_1','deadlift'),('stu9_10','bench'),('deadlift_08','deadlift'),('stu8_7','bench')]:
        jobs.append((name,resolve_video(name+'.mp4',team),ex))
    for key in ['deadlift/deadlift_30.mp4','benchpress/bench press_40.mp4']:
        jobs.append((Path(key).stem,args.local_root/'new_data/new_dataset'/key,
                     'bench' if key.startswith('benchpress/') else 'deadlift'))
    return {name:infer(args,name,path,ex) for name,path,ex in jobs}

# Drawing and browser routines follow below.

def browser_capture(job):
    """Runs in a Python with Playwright, independently of the recognition environment."""
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    from playwright.sync_api import sync_playwright
    from PIL import Image
    output, shots = Path(job['output']), Path(job['shots'])
    shots.mkdir(parents=True,exist_ok=True)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=job['root'],**kw)
        def log_message(self,*a):pass
        def do_GET(self):
            if self.path=='/favicon.ico':self.send_response(204);self.end_headers();return
            super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    errors=[]; metadata={}; converted=False
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--enable-unsafe-swiftshader','--use-gl=angle',
                                     '--use-angle=swiftshader','--disable-dev-shm-usage'])
            context=browser.new_context(viewport={'width':1920,'height':1080},device_scale_factor=2)
            page=context.new_page()
            page.on('console',lambda msg: errors.append(msg.text) if msg.type=='error' else None)
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/plate_pipeline/docs/midterm/squat_compare_demo_v3.html",wait_until='load')
            page.evaluate('document.fonts.ready')
            playable=page.evaluate('document.createElement("video").canPlayType(\'video/mp4; codecs="avc1.42E01E"\')')
            inputs={k:Path(job[k]) for k in ['user','ref']}
            if not playable:
                converted=True
                for role, source in list(inputs.items()):
                    dest=output/f'{role}.webm';sig=output/f'{role}_webm_source.txt'
                    source_hash=digest(source)
                    if not dest.exists() or not sig.exists() or sig.read_text()!=source_hash:
                        subprocess.run(['ffmpeg','-y','-v','error','-i',str(source),'-map','0:v:0',
                                        '-c:v','libvpx-vp9','-b:v','0','-crf','28','-deadline','realtime',
                                        '-cpu-used','6','-row-mt','1','-threads','4','-an',str(dest)],check=True)
                        sig.write_text(source_hash)
                    inputs[role]=dest
            page.locator('#log-new').click()
            page.locator('#d-user input').set_input_files(inputs['user'])
            page.locator('#d-ref input').set_input_files(inputs['ref'])
            page.locator('#go').click()
            page.locator('#s-report.on').wait_for(timeout=120000)
            page.wait_for_timeout(2500)
            def settled():
                page.wait_for_function('''() => ['#v-user','#v-ref'].every(id => {
                  const v=document.querySelector(id);return v && !v.seeking && v.readyState>=2;
                })''',timeout=30000)
                page.evaluate('''() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))''')
                assert not errors, errors
            settled()
            page.locator('.reptab').nth(0).click()
            page.locator('#posbtns button').nth(2).click()
            page.locator('#views button[data-v="side"]').click()
            page.locator('#mode3d button').nth(0).click()
            page.evaluate('document.querySelectorAll("video").forEach(v=>v.pause())')
            settled()
            times=page.evaluate('''() => ({rep:S.rep,user:document.querySelector('#v-user').currentTime,
                     ref:document.querySelector('#v-ref').currentTime,
                     user_duration:document.querySelector('#v-user').duration,
                     ref_duration:document.querySelector('#v-ref').duration,
                     user_state:document.querySelector('#v-user').readyState,
                     ref_state:document.querySelector('#v-ref').readyState,
                     user_seeking:document.querySelector('#v-user').seeking,
                     ref_seeking:document.querySelector('#v-ref').seeking})''')
            assert times['rep']==1 and not times['user_seeking'] and not times['ref_seeking']
            specs=[('demo_side_by_side.png','#sgrid'),('demo_3d.png','#three'),
                   ('demo_timeline.png','#tline'),('demo_findings.png','#fb-list')]
            for name, selector in specs:
                loc=page.locator(selector);loc.scroll_into_view_if_needed();settled()
                target=shots/name
                if selector=='#tline':
                    # Keep the requested 24 CSS px margin without cutting text
                    # from adjacent controls/video overlays into the capture.
                    loc.screenshot(path=target)
                    from PIL import ImageOps
                    with Image.open(target) as im:
                        ImageOps.expand(im,border=48,fill='white').save(target)
                else:
                    loc.screenshot(path=target)
                with Image.open(target) as im:
                    before=im.size
                    if im.width>3200:
                        im.resize((3200,round(im.height*3200/im.width)),Image.Resampling.LANCZOS).save(target)
                with Image.open(target) as im:after=im.size
                metadata[name]={'selector':selector,'size':after,'original_capture_size':before,
                                'times':times,'viewport':[1920,1080],'device_scale_factor':2}
            assert not errors, errors
            browser.close()
            metadata.update(webm=converted,console_errors=errors,page_sha256=digest(Path(job['root'])/'plate_pipeline/docs/midterm/squat_compare_demo_v3.html'))
    finally:
        server.shutdown();server.server_close()
    dump(output/'browser_metadata.json',metadata)
    print('데모 캡처 완료',json.dumps(metadata,ensure_ascii=False),flush=True)
    return metadata

class Painter:
    def __init__(self,args):
        from PIL import ImageFont
        from fontTools.ttLib import TTCollection
        self.args=args
        font=Path(subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True).strip())
        if font.suffix=='.ttc':
            face=next(f for f in TTCollection(font).fonts if f['name'].getDebugName(1)=='Noto Sans CJK KR')
            font=args.output/'NotoSansCJK-KR.otf';face.save(font)
        self.font_path=font
        self.ImageFont=ImageFont
    def font(self,size):return self.ImageFont.truetype(str(self.font_path),size)
    def label(self,im,text,xy,anchor='lt',size=22):
        from PIL import Image,ImageDraw
        overlay=Image.new('RGBA',im.size)
        d=ImageDraw.Draw(overlay);font=self.font(size)
        bbox=d.multiline_textbbox((0,0),text,font=font,spacing=5)
        w,h=bbox[2]-bbox[0]+20,bbox[3]-bbox[1]+16
        x,y=xy
        if 'r' in anchor:x-=w
        if 'b' in anchor:y-=h
        x=max(4,min(im.width-w-4,x));y=max(4,min(im.height-h-4,y))
        d.rounded_rectangle((x,y,x+w,y+h),radius=9,fill=(14,17,22,191))
        d.multiline_text((x+10-bbox[0],y+8-bbox[1]),text,font=font,fill=WHITE,spacing=5)
        im.alpha_composite(overlay)
        return (x,y,x+w,y+h)
    def panel(self,im,title,width=1000):
        from PIL import Image,ImageDraw
        im=im.convert('RGBA')
        scale=min(1,width/im.width)
        if scale<1:im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
        return im,scale,title
    def combine(self,name,panels,metadata):
        from PIL import Image,ImageDraw
        # No enlargement/cropping: frames retain their individual aspect ratios.
        widths=[p[0].width for p in panels];height=max(p[0].height for p in panels)
        header=92 if any(len(p[2])>17 for p in panels) else 62
        total=sum(widths)+16*(len(panels)-1)
        assert total<=3200,(name,total)
        canvas=Image.new('RGB',(total,height+header),'white');d=ImageDraw.Draw(canvas)
        x=0
        for im,_,title in panels:
            font=self.font(25 if len(title)<20 else 22)
            if d.textlength(title,font=font)>im.width-20:
                # Split titles at a semantic boundary; never clip or stretch text.
                title=title.replace(' → ','\n→ ')
            d.multiline_text((x+10,12),title,font=font,fill='#1F2A5C',spacing=5)
            canvas.paste(im.convert('RGB'),(x,header));x+=im.width+16
        canvas.save(self.args.shots/name)
        metadata['size']=list(canvas.size)
        return metadata


def get_frame(path,seq):
    from core.io import video
    from PIL import Image
    for i,(t,bgr) in enumerate(video.frames(path)):
        if i==seq:return Image.fromarray(bgr[:,:,::-1]).convert('RGBA'),t
    raise IndexError((path,seq))


def nearest_row(run,t,used=False):
    rows=run['rows']
    if used:rows=[r for r in rows if r.get('plate',{}).get('used')]
    return min(rows,key=lambda r:abs(r['t']-t))


def direct_box(row):
    p=row.get('plate',{});i=row.get('pick')
    if p.get('src')!='det' or i is None:return None
    d=row['dets'][i]
    assert abs((d[0]+d[2])/2-p['center'][0])<.02
    return d


def plate_overlay(im,row,scale,painter,score=True):
    from PIL import ImageDraw
    box=direct_box(row) or row['plate']['box']
    b=[float(v)*scale for v in box[:4]]
    ImageDraw.Draw(im).rectangle(b,outline=TEAL,width=3)
    if score:
        value=box[4] if len(box)>4 else row['plate']['score']
        painter.label(im,f'원판 {value:.2f}',(b[0],max(6,b[1]-40)),size=21)
    return b


def trail(im,run,start,end,scale):
    from PIL import ImageDraw
    d=ImageDraw.Draw(im);last=None
    for row in run['rows']:
        if not start<=row['t']<=end:continue
        if not row.get('plate',{}).get('used'):
            last=None;continue
        point=tuple(v*scale for v in row['plate']['center'])
        if last is not None:d.line([last,point],fill=TEAL,width=3)
        last=point
    if last:d.ellipse((last[0]-4,last[1]-4,last[0]+4,last[1]+4),fill=TEAL)


def draw_skeleton(im,kp,scale,grey=False):
    from PIL import ImageDraw
    from core.recog.pose import POSE_CONNECTIONS
    import numpy as np
    a=np.asarray(kp,float)
    if a.shape[1]==4:a=a[:,[0,1,3]]
    d=ImageDraw.Draw(im)
    for u,v in POSE_CONNECTIONS:
        if not np.isfinite(a[[u,v]]).all():continue
        # Draw the actual estimates; visibility is reported separately, not repaired.
        color=GREY if grey else WHITE
        d.line([tuple(a[u,:2]*scale),tuple(a[v,:2]*scale)],fill=color,width=3)
    for x,y,vis in a:
        if not math.isfinite(x+y+vis):continue
        x*=scale;y*=scale
        color=GREY if grey else TEAL
        d.ellipse((x-4,y-4,x+4,y+4),fill=color)


def frame_meta(run,row,actual_t):
    return {'video':run['path'],'seq':row['seq'],'frame_t':actual_t,'row_t':row['t']}


def detection_shot(args,runs,painter):
    panels=[];records=[]
    for name,title in [('reference','스쿼트'),('stu2_10_edited','벤치프레스'),('stu7_1_1','데드리프트')]:
        run=runs[name];final,reps,values=final_and_reps(run)
        eligible=[]
        for row in run['rows']:
            box=direct_box(row)
            if box is None or box[4]<.85:continue
            if not (0<=box[0]<box[2]<run['result']['width'] and 0<=box[1]<box[3]<run['result']['height']):continue
            for tr in reps:
                if tr.entry_t<=row['t']<=tr.end_t:
                    eligible.append((abs(row['t']-tr.far_t),-box[4],row,tr));break
        _,_,row,tr=min(eligible,key=lambda r:r[:2])
        im,t=get_frame(run['path'],row['seq']);im,scale,title=painter.panel(im,title,width=640)
        plate_overlay(im,row,scale,painter)
        painter.label(im,f'{t:.3f}초',(im.width-10,im.height-10),anchor='rb',size=20)
        box=direct_box(row)
        panels.append((im,scale,title));records.append({**frame_meta(run,row,t),'detector_box':box[:4],'score':box[4],'src':'det','far_t':tr.far_t})
    return painter.combine('plate_detect.png',panels,{'frames':records})


def state_shot(args,run,painter):
    import numpy as np
    from PIL import ImageDraw
    from core.config import DEFAULT
    final,reps,values=final_and_reps(run);s=final.sig
    order=[1]+[i for i in range(len(reps)) if i!=1]
    chosen=next(i for i in order if i<len(reps) and all(values[i][k] is not None for k in ['start','far_in','far_out','back'])
                and not values[i]['blank'] and not reps[i].entry_gap and not reps[i].far_gap)
    tr,v=reps[chosen],values[chosen]
    targets=[tr.entry_t-.3,tr.entry_t,tr.far_t,tr.end_t]
    distances=[]
    for i,t in enumerate(targets):
        if i==0:y=float(np.interp(t,s.t,s.y))
        else:
            indices=np.flatnonzero(s.t==t);assert len(indices)==1
            y=float(s.y[indices[0]])
        distances.append(-(y-tr.ref_y)*100)
    assert np.isclose(distances[2],tr.amp*100,atol=1e-10)
    assert targets[1:]==[tr.entry_t,tr.far_t,tr.end_t]
    baseline=nearest_row(run,v['start'],True)['plate']['center'][1]
    titles=['① 대기','② 6 cm 벗어남 → 진행','③ 가장 먼 점 (바닥)','④ 70% 돌아옴 → 1회']
    states=['대기','진행 시작','진행','대기']
    panels=[];records=[]
    for i,(target,distance,title,state) in enumerate(zip(targets,distances,titles,states)):
        row=nearest_row(run,target);im,t=get_frame(run['path'],row['seq'])
        assert abs(t-target)<=.5/24+.0001
        im,scale,title=painter.panel(im,title,width=788)
        d=ImageDraw.Draw(im);y=baseline*scale
        for x in range(0,im.width,14):d.line((x,y,min(x+8,im.width),y),fill=GREY,width=2)
        trail(im,run,v['start'],row['t'],scale)
        plate_overlay(im,row,scale,painter,score=False)
        painter.label(im,f'{state}\n기준점에서 {distance:.1f} cm',(10,10),size=22)
        painter.label(im,f'{target:.3f}초',(10,im.height-10),anchor='lb',size=20)
        if i==3:painter.label(im,'1회',(im.width-10,10),anchor='rt',size=29)
        record={**frame_meta(run,row,t),'requested_t':target,'distance_cm':distance,'signal_y':float(np.interp(target,s.t,s.y))}
        assert record['requested_t']==targets[i]
        assert np.isclose(record['distance_cm'],-(record['signal_y']-tr.ref_y)*100,atol=1e-10)
        panels.append((im,scale,title));records.append(record)
    return painter.combine('rep_states.png',panels,{'rep':chosen+1,'trip':asdict(tr),'frames':records,
                'baseline_pixel_y':baseline,'baseline_source_t':nearest_row(run,v['start'],True)['t'],
                'trail_start':v['start'],'assertions':'Trip times, final-signal distances, nearest decoded frame within half a frame'})

def comparison_shot(args,runs,painter):
    import numpy as np
    table=candidates(args)
    clocks={r['key']:json.loads(r['gt_events']) for r in rows_csv(PIPE/'results/midterm/gt_clock.csv')}
    possibilities=[]
    for key in ['deadlift/deadlift_30.mp4','benchpress/bench press_40.mp4']:
        run=runs[Path(key).stem]
        timeline=rows_csv(args.v40/f'arms/B_generic/timelines/all_test/{key}.csv')
        predicted=[float(r['time_s']) for r in timeline if r['count_event'].lower() in ('true','1')]
        missing=[t for t in clocks[key] if not any(abs(t-p)<=.5 for p in predicted)]
        cached=args.v40/'poses/B_generic'/f'{key}.npz'
        source='B_generic' if cached.is_file() else '파이프라인 MediaPipe 포즈 (--pose)'
        source_data=np.load(cached,allow_pickle=False) if cached.is_file() else None
        previous=None
        for row in run['rows']:
            box=direct_box(row)
            if box is None or box[4]<.85 or not row.get('kp') or not missing:continue
            event=min(missing,key=lambda t:abs(t-row['t']))
            if abs(event-row['t'])>.65:continue
            if source_data is not None:
                index=int(np.argmin(abs(source_data['timestamps']-row['t'])))
                raw=source_data['poses'][index]
                kp=np.full((33,3),np.nan)
                ids=[0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
                kp[ids]=raw
                if np.nanmax(abs(kp[:,:2]))<=2:kp[:,:2]*=[run['result']['width'],run['result']['height']]
            else:kp=np.asarray(row['kp'],float)
            visibility=kp[[13,14,15,16,25,26],2]
            low=float(np.nanmin(visibility))
            if not low<.5:continue
            jump=float(np.nanmax(np.linalg.norm(kp[:,:2]-previous[:,:2],axis=1))/(box[3]-box[1])) if previous is not None else None
            previous=kp
            possibilities.append((low,-box[4],abs(event-row['t']),run,row,kp,key,event,source,cached,jump))
        if source_data is not None:source_data.close()
    if not possibilities:raise RuntimeError('No low-visibility/high-detection frame near a saved pose error')
    low,_,_,run,row,kp,key,event,source,cached,jump=min(possibilities,key=lambda x:x[:3])
    original,t=get_frame(run['path'],row['seq'])
    left,scale,_=painter.panel(original.copy(),'관절: 가려지고 흔들림',width=1000)
    draw_skeleton(left,kp,scale)
    weak=int(np.nanargmin(kp[[13,14,15,16,25,26],2]))
    ids=[13,14,15,16,25,26];names=['왼팔꿈치','오른팔꿈치','왼손목','오른손목','왼무릎','오른무릎']
    painter.label(left,f'{names[weak]} 관측 {low:.2f}',(left.width-10,10),anchor='rt',size=23)
    right,scale,_=painter.panel(original.copy(),'원판: 계속 보임',width=1000)
    _,reps,values=final_and_reps(run)
    rid=min(range(len(reps)),key=lambda i:abs(reps[i].end_t-event))
    start=values[rid]['start'] if values[rid]['start'] is not None else reps[rid].entry_t
    trail(right,run,start,row['t'],scale)
    plate_overlay(right,row,scale,painter)
    for im in [left,right]:painter.label(im,f'{t:.3f}초',(im.width-10,im.height-10),anchor='rb',size=20)
    selected=next(r for r in table if r['key']==key)
    return painter.combine('pose_vs_plate.png',[(left,scale,'관절: 가려지고 흔들림'),(right,scale,'원판: 계속 보임')],
               {'frames':[frame_meta(run,row,t)]*2,'candidates':table,'selected':selected,
                'error_t':event,'error_kind':'빠짐','joint_source':source,'B_generic_pose_cache':str(cached),
                'pose_timeline':str(args.v40/f'arms/B_generic/timelines/all_test/{key}.csv'),
                'min_visibility':low,'weak_joint':names[weak],'joint_visibility':kp[[13,14,15,16,25,26],2].tolist(),
                'plate_score':direct_box(row)[4],'rep':rid+1,'trail_start':start,
                'note':'동일 시각의 관절 관측과 원판 관측 비교. 관절 가림이 포즈 계수 누락의 원인이라는 인과 검증은 아님.'})


def lifter_shot(args,runs,painter):
    import numpy as np
    from core.recog.pose import near_wrist,select_lifter,person_box
    choices=[]
    for run in runs.values():
        if run['name']=='stu9_10':continue  # both detected poses are helpers, not the lying lifter
        by={r['t']:r for r in run['rows']}
        for record in run['poses']:
            row=by[record['t']];box=direct_box(row)
            if box is None or box[4]<.7 or len(record['candidates'])<2 or not row.get('kp'):continue
            cs=[np.asarray(x,float) for x in record['candidates']]
            wrists=[];distances=[]
            center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2]);height=box[3]-box[1]
            for q in cs:
                wrist=near_wrist(q[:,[0,1,3]],box[:4],frame_h=run['result']['height'],frame_w=run['result']['width'])
                wrists.append(wrist)
                if wrist is not None:distance=float(np.linalg.norm(np.array(wrist)-center)/height)
                else:
                    p=person_box(q)
                    distance=float(np.hypot(max(p[0]-center[0],0,center[0]-p[2]),max(p[1]-center[1],0,center[1]-p[3]))/height) if p else math.inf
                distances.append(distance)
            idx=select_lifter(cs,box[:4],None,run['exercise'],image_size=(run['result']['width'],run['result']['height']))
            if idx is None or wrists[idx] is None:continue
            # Require a clear separation; reject duplicate estimates of one person.
            others=[d for i,d in enumerate(distances) if i!=idx]
            if min(others)-distances[idx]<.6:continue
            actual=min(range(len(cs)),key=lambda i:np.mean(abs(cs[i][:,:2]-np.array(row['kp'])[:,:2])))
            assert idx==actual
            choices.append((min(others)-distances[idx],box[4],run,row,cs,wrists,distances,idx))
    if not choices:raise RuntimeError('No distinct multi-person frame in the processed team/cohort videos')
    _,_,run,row,cs,wrists,distances,idx=max(choices,key=lambda x:x[:2])
    im,t=get_frame(run['path'],row['seq']);im,scale,title=painter.panel(im,'원판에 가장 가까운 손목 = 리프터',width=1600)
    from PIL import ImageDraw
    for i,q in enumerate(cs):draw_skeleton(im,q,scale,grey=i!=idx)
    plate_overlay(im,row,scale,painter,score=False)
    center=np.array(row['plate']['center'])*scale
    draw=ImageDraw.Draw(im)
    for i,w in enumerate(wrists):
        if w is None:continue
        draw.line([tuple(np.array(w)*scale),tuple(center)],fill=TEAL if i==idx else GREY,width=3)
    painter.label(im,f'리프터\n원판 높이 {distances[idx]:.2f}배',(10,im.height-10),anchor='lb',size=22)
    other=next(i for i in range(len(cs)) if i!=idx)
    painter.label(im,f'원판 높이 {distances[other]:.2f}배 · 제외',(im.width-10,10),anchor='rt',size=22)
    return painter.combine('lifter_select.png',[(im,scale,title)],{'frames':[frame_meta(run,row,t)],'people':len(cs),
                'selected_index':idx,'distances':distances,'wrists':wrists,'limit':4.0,
                'exclusion':('선택되지 않은 사람은 손목 거리가 4배를 넘어 제외됨' if all(d>4 for j,d in enumerate(distances) if j!=idx) else '한계 4배 안의 후보도 더 가까운 손목이 있으면 선택하지 않음'),
                'selection_assertion':'select_lifter index equals saved pipeline lifter'})


def angle_shot(args,runs,painter):
    from PIL import Image,ImageDraw
    panels=[];records=[]
    for name,title in [('user','내 영상'),('reference','기준 영상')]:
        run=runs[name]
        eligible=[r for r in run['rows'] if direct_box(r) is not None
                  and direct_box(r)[0]>=8 and direct_box(r)[1]>=8
                  and direct_box(r)[2]<=run['result']['width']-8
                  and direct_box(r)[3]<=run['result']['height']-8]
        if name=='reference':
            eligible=[r for r in eligible if direct_box(r)[4]>=.85]
            def padding(row):
                q=direct_box(row);w,h=q[2]-q[0],q[3]-q[1]
                a=(q[0]+q[2])/2-1.5*w;b=(q[1]+q[3])/2-1.5*h
                empty=(max(0,-a)+max(0,a+3*w-run['result']['width']))/(3*w)
                empty+=(max(0,-b)+max(0,b+3*h-run['result']['height']))/(3*h)
                return (empty,-q[4])
            row=min(eligible,key=padding)
        else:row=max(eligible,key=lambda r:direct_box(r)[4])
        box=direct_box(row)
        im,t=get_frame(run['path'],row['seq'])
        x0,y0,x1,y1=box[:4];w,h=x1-x0,y1-y0
        cropw,croph=round(3*w),round(3*h)
        cx,cy=(x0+x1)/2,(y0+y1)/2
        a=round(cx-cropw/2);b=round(cy-croph/2)
        crop=Image.new('RGBA',(cropw,croph),'white');crop.paste(im,(-a,-b))
        scale=min(800/crop.width,850/crop.height)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
        bx=[(x0-a)*scale,(y0-b)*scale,(x1-a)*scale,(y1-b)*scale]
        d=ImageDraw.Draw(crop);d.rectangle(bx,outline=TEAL,width=3)
        # Measurement marks stay in the photo's crop; these are not plot axes.
        yy=bx[3]+20;xx=bx[2]+20
        d.line((bx[0],yy,bx[2],yy),fill=WHITE,width=3)
        for x in [bx[0],bx[2]]:d.line((x,yy-6,x,yy+6),fill=WHITE,width=3)
        d.line((xx,bx[1],xx,bx[3]),fill=WHITE,width=3)
        for y in [bx[1],bx[3]]:d.line((xx-6,y,xx+6,y),fill=WHITE,width=3)
        ratio=w/h;angle=math.degrees(math.acos(min(1,max(0,ratio))))
        painter.label(crop,f'가로/세로 {ratio:.2f} → 약 {angle:.0f}°',(10,10),size=23)
        painter.label(crop,f'가로 {w:.1f} px\n세로 {h:.1f} px',(10,crop.height-10),anchor='lb',size=21)
        panels.append((crop,scale,title));records.append({**frame_meta(run,row,t),'ratio':ratio,'angle_degrees':angle,
                      'detector_box':box[:4],'score':box[4],'crop':[a,b,a+cropw,b+croph]})
    return painter.combine('angle_plate.png',panels,{'frames':records,'formula':'acos(clamp(width/height,0,1)); degrees'})

def browser_python():
    options=[Path(sys.executable),Path(sys.executable).parents[3]/'bin/python',Path.home()/'miniconda3/bin/python']
    for executable in dict.fromkeys(options):
        if executable.exists() and subprocess.run([str(executable),'-c','import playwright.sync_api'],
                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            return executable
    raise RuntimeError('Playwright Python and its installed Chromium are required')


def report(args,metadata,runs,checks,browser):
    from PIL import Image
    table=candidates(args)
    sources={k:{'video':r['path'],'video_sha256':digest(r['path']),
              'model_sha256':digest(args.model),'providers':r['result']['detector_providers'],
              'reps':sum(s['reps'] for s in r['result']['sets'])} for k,r in runs.items()}
    dump(args.output/'shot_sources.json',{'images':metadata,'runs':sources,'browser':browser,'tests':checks})
    text='''# 중간 발표 캡처 보고서

작업 브랜치: integ/midterm-shots. integ/midterm-eval(67e7173)에서 만들고
origin/plate-pipeline(186fab0)을 충돌 없이 병합해 데모 v3를 가져왔다.
이번 작업에서는 PNG 9개와 shots.py만 커밋한다. 병합으로 가져온 기존 데모 파일은 별도다.
PLAN.md·인식/판정 코어·기존 평가 표는 수정하지 않았다.
제출 PNG에는 새 그래프를 만들지 않았다. 영상 위 선은 실제 원판 중심 경로 또는 길이 표시이며 축·곡선·막대 그림이 아니다.
데모의 기존 진행 바는 페이지를 그대로 캡처했다.

## 이미지별 크기·입력·시각

| 이미지 | 크기(px) | 영상·프레임 시각 |
|---|---|---|
'''
    for name,value in metadata.items():
        frames=value.get('frames',[])
        desc='; '.join(f"{Path(f['video']).name} {f['frame_t']:.6f}초 (프레임 {f['seq']})" for f in frames)
        if not frames:
            t=value['times'];desc=f"내 영상 {t['user']:.6f}초 / 기준 {t['ref']:.6f}초, rep 1·바닥"
        text+=f"| {name} | {value['size'][0]}×{value['size'][1]} | {desc} |\n"
    text+='''
## 1. 원판 검출

team9의 세 종목에서 진행 중이고 원판 상자가 화면 안에 들어오는 프레임을 골랐다.
기준 stu10_70 입력은 team9 파일과 SHA-256이 같다. 점수 0.85 이상에서 가장 먼 점에 가까운 직접 검출을 택했다.
상자는 rows.dets[pick]의 검출기 원본 실수 좌표를 사용했다. KLT 상자를 검출기 출력으로 바꾸어 쓰지 않았다.
'''
    for f in metadata['plate_detect.png']['frames']:
        text+=f"- {f['video']}: {f['frame_t']:.6f}초, 점수 {f['score']:.9f}, 상자 {f['detector_box']}.\n"
    state=metadata['rep_states.png'];tr=state['trip']
    text+=f'''
## 2. 상태 전환

stu10_70의 **{state['rep']}번째 반복**, 빈칸 없는 rep을 사용했다.
Trip entry/far/end = **{tr['entry_t']:.12f} / {tr['far_t']:.12f} / {tr['end_t']:.12f}초**,
A = {tr['amp']*100:.12f} cm, ref_y={tr['ref_y']:.12f} m.

| 칸 | Trip으로 정한 시각(초) | 실제 영상 프레임(초) | 기준점에서 거리(cm) |
|---|---:|---:|---:|
'''
    for i,f in enumerate(state['frames'],1):
        text+=f"| {i} | {f['requested_t']:.12f} | {f['frame_t']:.12f} | {f['distance_cm']:.12f} |\n"
    text+=f'''
- 첫 칸은 entry−0.3초에서 최종 신호를 선형 조회했고 나머지는 해당 Trip 시각의 최종 신호 표본을 직접 읽었다.
  거리 = −(최종 신호 y−Trip.ref_y)×100. 그림의 현재 거리를 6·13·A·복귀 임계값으로 강제로 바꾸지 않았다.
- Trip 시각은 최종 신호 재표본화 격자여서 원본 프레임과 완전히 같지 않다.
  각 칸은 요청 시각을 유지하고 가장 가까운 실제 디코드 프레임을 썼다. 프레임 오차 ≤반 프레임 assert 통과.
  요청 entry/far/end와 Trip float의 정확 일치, 표시 거리와 최종 신호 값의 일치를 assert로 확인했다.
- 점선의 원본 높이는 rep 값 start에 가장 가까운 측정 행({state['baseline_source_t']:.6f}초)의 중심 y={state['baseline_pixel_y']:.4f}px.
  원판 경로는 rep 값 start={state['trail_start']:.12f}초부터 현재 원본 프레임까지만 사용했다. lost 사이를 잇지 않는다.
- 네 번째 칸은 복귀 조건으로 1회를 센 직후라 상태는 다시 대기, 오른쪽 위 계수는 1회다.

## 3. 포즈와 원판

후보 전체 {len(table)}개: 동일 61개에서 원판 완료 횟수=정답, B_generic 횟수≠정답.

| 영상 | 정답 | 포즈 | 원판 |
|---|---:|---:|---:|
'''
    for r in table:text+=f"| {r['key']} | {r['truth']} | {r['pose']} | {r['plate']} |\n"
    comp=metadata['pose_vs_plate.png'];f=comp['frames'][0]
    text+=f'''
- 선택: {comp['selected']['key']}, {f['frame_t']:.6f}초, 프레임 {f['seq']}.
- B_generic 누락 완료 시각 {comp['error_t']:.12f}초의 ±0.65초 안에서,
  원판 직접 검출 ≥0.85이고 손목·팔꿈치·무릎 중 visibility<0.5인 후보를 비교했다.
  비교 영상 deadlift_30와 bench press_40 중 낮은 visibility, 높은 검출 점수 순으로 선택했다.
- {comp['weak_joint']} visibility={comp['min_visibility']:.6f}; 관절 [왼팔꿈치, 오른팔꿈치, 왼손목, 오른손목, 왼무릎, 오른무릎]
  visibility={comp['joint_visibility']}. 원판 점수={comp['plate_score']:.9f}.
- 관절 출처: **{comp['joint_source']}**.
  B_generic 예측/구간 CSV는 존재하지만 관절 NPZ `{comp['B_generic_pose_cache']}`는 로컬에 없어 --pose 관측을 사용했다.
  이 골격을 B_generic 입력 자체라고 주장하지 않는다. 관절 가림이 계수 누락의 원인이라는 인과 검증도 아니다.
- 포즈 구간 예측은 `{comp['pose_timeline']}`에서 그대로 읽었으며 포즈 분류를 다시 추론/디코딩하지 않았다.
- 오른쪽 경로는 선택된 rep {comp['rep']}의 start={comp['trail_start']:.12f}초 이후 실제 원판 중심이다.

## 4. 리프터 고르기
'''
    lifter=metadata['lifter_select.png'];f=lifter['frames'][0]
    text+=f'''
- 영상 {f['video']}, {f['frame_t']:.6f}초, 프레임 {f['seq']}, 사람 {lifter['people']}명.
- 선택 index={lifter['selected_index']}, 사람별 거리={lifter['distances']} (원판 높이 단위).
- select_lifter와 같은 visibility≥0.5·화면 안 손목 중 원판 중심과 가장 가까운 손목을 사용하고,
  손목이 없을 때만 person_box와의 거리를 쓴다. 한계는 4배. 이번 선택은 저장된 파이프라인 리프터 index와 같은지 assert 통과.
- 제외 사유: {lifter['exclusion']}. 실제 거리와 한계를 구분해 표시했다.

## 5. 가로·세로 비율과 각도

주변을 검출 box 가로·세로 각각 3배 크기로 잘랐고 프레임 경계 밖은 흰색 여백으로 남겼다. 원판 box 자체는 경계에서 8px 이상 안쪽인 직접 검출을 골랐다.
확대 이미지만 확대를 허용했다. 다른 프레임은 늘리거나 자르지 않았다.
각도는 acos(clamp(가로/세로,0,1))를 도로 바꿨다. 타원 투영 가정에 따른 추정이며 보정된 자세의 정답이 아니다.
'''
    for f in metadata['angle_plate.png']['frames']:
        text+=f"- {f['video']}, {f['frame_t']:.6f}초: 비율 {f['ratio']:.12f}, 각도 {f['angle_degrees']:.9f}°, 점수 {f['score']:.6f}, crop={f['crop']}.\n"
    text+=f'''
## 6~8. 데모 페이지

- `plate_pipeline/docs/midterm/squat_compare_demo_v3.html` 원본을 로컬 HTTP로 열었다. 페이지 내용·분석 데이터·CSS를 수정하지 않았다.
- 창 1920×1080, device_scale_factor=2. rep 1 → 세 번째 포지션(바닥), 측면·겹쳐 보기.
- #s-report.on 뒤 2.5초를 기다렸고 캡처 전에 두 영상의 seeking=false, readyState≥2를 확인했다.
- webm 임시 사본: **{'사용' if browser['webm'] else '사용하지 않음 (Chromium이 원본 H.264를 열었음)'}**.
- 콘솔 오류·pageerror: **{browser['console_errors']}**. 하나라도 있으면 스크립트가 실패하도록 했다.
- #sgrid / #three 요소만 캡처했다. #tline은 요소만 캡처해 주위 CSS 24px(2배 캡처에서 48px)의 흰 여백을 더했다. 이웃 조작 버튼의 글자를 중간에서 자르지 않기 위해서다.
  원래 {browser['demo_timeline.png']['original_capture_size']}px에서 가로 3200px 제한에 맞춰 비율을 유지해 축소했다.
- #fb-list를 demo_findings.png로 별도 저장했다.
- 데모의 6회 사용자 분석은 페이지에 들어 있던 분석이다. 이번 원판 모델 단독 처리와 혼동하지 않는다.
  이번 실행의 사용자 영상 원판 횟수는 {sources['user']['reps']}회(입력 정답 6회)다.
  원본 데모 제작에서는 일부 연결 클립에 초기 원판 box를 주었지만, 이번 캡처 작업은 모델·포즈 조건을 바꾸지 않았다.

## 검증·첨부

- pytest: {checks}.
- PIL과 Noto Sans CJK KR로 모든 추가 한국어 라벨을 그렸다. OpenCV putText는 사용하지 않았다.
- 원판 박스 #17A08F 3px, 리프터 흰 선·청록 점, 다른 사람 회색. 라벨은 #0E1116 75% 불투명 둥근 상자와 흰 글자.
- 합성 칸 사이 흰 여백 16px, 모든 PNG 폭 ≤3200px. 원본 프레임은 비율을 유지해 축소만 했으며 각도 확대만 예외다.
- 세부 프레임·Trip·입력 SHA-256·실제 provider: `{args.output/'shot_sources.json'}`.
- PNG 전부(9장): `{args.output/'midterm_shots.zip'}`.

재현 명령:

```bash
python plate_pipeline/docs/midterm/src/shots.py --user {args.user} --ref {args.ref}
```
'''
    (args.output/'REPORT.md').write_text(text)


def main():
    from PIL import Image
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--user',type=Path,required=True);p.add_argument('--ref',type=Path,required=True)
    p.add_argument('--local-root',type=Path,default=Path.home()/'capstone_exercise')
    p.add_argument('--model',type=Path);p.add_argument('--v40',type=Path)
    p.add_argument('--output',type=Path,default=PIPE/'eval_out/midterm_shots')
    p.add_argument('--shots',type=Path,default=PIPE/'docs/midterm/shots')
    p.add_argument('--refresh',action='store_true',help='re-run current-code recognition instead of matching signed cache')
    p.add_argument('--reuse-browser',action='store_true',help='redraw annotations without repeating a verified demo capture')
    p.add_argument('--skip-tests',action='store_true',help='for visual iteration only; default runs both requested suites')
    args=p.parse_args()
    args.model=args.model or args.local_root/'plate_pipeline/models/plate_tiny.onnx'
    args.v40=args.v40 or args.local_root/'new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42'
    args.output.mkdir(parents=True,exist_ok=True);args.shots.mkdir(parents=True,exist_ok=True)
    setup_core();runs=prepare(args);painter=Painter(args)
    metadata={}
    metadata['plate_detect.png']=detection_shot(args,runs,painter)
    metadata['rep_states.png']=state_shot(args,runs['reference'],painter)
    metadata['pose_vs_plate.png']=comparison_shot(args,runs,painter)
    metadata['lifter_select.png']=lifter_shot(args,runs,painter)
    metadata['angle_plate.png']=angle_shot(args,runs,painter)
    bpy=browser_python()
    if not args.reuse_browser:
        job={'root':str(ROOT),'output':str(args.output),'shots':str(args.shots),'user':str(args.user),'ref':str(args.ref)}
        dump(args.output/'browser_job.json',job)
        subprocess.run([str(bpy),str(Path(__file__).resolve()),'--browser-job',str(args.output/'browser_job.json')],check=True)
    browser=json.loads((args.output/'browser_metadata.json').read_text())
    assert browser['console_errors']==[]
    assert browser['page_sha256']==digest(PIPE/'docs/midterm/squat_compare_demo_v3.html')
    for name in ['demo_side_by_side.png','demo_3d.png','demo_timeline.png','demo_findings.png']:metadata[name]=browser[name]
    checks={}
    if not args.skip_tests:
        driver=subprocess.check_output([str(bpy),'-c','import pathlib,playwright;print(pathlib.Path(playwright.__file__).parent/"driver")'],text=True).strip()
        env={**os.environ,'PATH':driver+os.pathsep+os.environ.get('PATH','')}
        for label,cwd,path in [('원판',PIPE,'tests'),('피드백',ROOT,'tests/fb')]:
            completed=subprocess.run([sys.executable,'-m','pytest','-q',path],cwd=cwd,env=env,text=True,
                                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (args.output/f'{label}_pytest.txt').write_text(completed.stdout)
            print(completed.stdout,flush=True);assert completed.returncode==0,label
            checks[label]=completed.stdout.strip().splitlines()[-1]
    for name in metadata:
        with Image.open(args.shots/name) as im:assert im.width<=3200 and list(im.size)==metadata[name]['size']
    report(args,metadata,runs,checks,browser)
    with zipfile.ZipFile(args.output/'midterm_shots.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in metadata:z.write(args.shots/name,name)
    print('캡처 9장, 보고서, ZIP 완료:',args.shots,args.output,flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--browser-job':browser_capture(json.loads(Path(sys.argv[2]).read_text()))
    else:main()
