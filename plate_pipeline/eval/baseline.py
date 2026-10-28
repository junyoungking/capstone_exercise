"""CPU pose baseline; --videos accepts alternate dataset/team roots, no new split."""
from pathlib import Path
import argparse
from contextlib import contextmanager, ExitStack
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from eval.baseline_metrics import measure, median, ratio, POSITIONS, ANGLES, JOINTS, rep_mask
from eval.midterm.reproduce import aggregate


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_root():
    common=subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=ROOT,text=True).strip()
    return (ROOT/common).resolve().parent


@contextmanager
def timing():
    """Delegate unchanged production calls; measure disjoint stages externally."""
    import cv2
    import unified_feedback_v4 as v4
    from core.io import video
    from core.recog.plate import OnnxPlateDetector
    from core.recog.track import PlateTracker
    totals=dict(decode=0.,plate_detection=0.,klt=0.,pose=0.)
    def timed(fn,key):
        def wrapped(*args,**kwargs):
            start=time.perf_counter()
            try:return fn(*args,**kwargs)
            finally:totals[key]+=time.perf_counter()-start
        return wrapped
    original_frames=video.frames
    def frames(*args,**kwargs):
        iterator=iter(original_frames(*args,**kwargs))
        while True:
            start=time.perf_counter()
            try:item=next(iterator)
            except StopIteration:
                totals['decode']+=time.perf_counter()-start
                return
            totals['decode']+=time.perf_counter()-start
            yield item
    capture_type=cv2.VideoCapture
    class Capture:
        def __init__(self,*args,**kwargs):self.capture=capture_type(*args,**kwargs)
        def __getattr__(self,name):return getattr(self.capture,name)
        def read(self):return timed(self.capture.read,'decode')()
    with ExitStack() as stack:
        stack.enter_context(patch.object(video,'frames',frames))
        stack.enter_context(patch.object(cv2,'VideoCapture',Capture))
        stack.enter_context(patch.object(OnnxPlateDetector,'__call__',timed(OnnxPlateDetector.__call__,'plate_detection')))
        stack.enter_context(patch.object(PlateTracker,'step',timed(PlateTracker.step,'klt')))
        stack.enter_context(patch.object(v4,'extract_all_landmarks',timed(v4.extract_all_landmarks,'pose')))
        yield totals


def contact_sheet(path,pose,segments,rows,dest):
    import cv2
    from core.recog.pose import POSE_CONNECTIONS
    from PIL import Image,ImageDraw
    indices=np.flatnonzero(rep_mask(segments))
    label='rep interval'
    if not len(indices):indices=np.arange(pose['total']);label='NO COMPLETED REP'
    if not len(indices):raise ValueError('empty video')
    chosen=indices[np.rint(np.linspace(0,len(indices)-1,6)).astype(int)]
    sheet=Image.new('RGB',(960,600),(24,27,32));capture=cv2.VideoCapture(str(path))
    try:
        for n,i in enumerate(chosen):
            capture.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,img=capture.read()
            if not ok:raise ValueError(f'cannot decode contact frame {i}: {path}')
            plate=rows[i].get('plate',{})
            if plate.get('used'):
                x0,y0,x1,y1=map(round,plate['box']);cv2.rectangle(img,(x0,y0),(x1,y1),(255,150,30),2)
            if pose['valid'][i]:
                points=pose['raw_lms'][i].copy();points[:,:2]*=[pose['w'],pose['h']]
                for a,b in POSE_CONNECTIONS:
                    if np.isfinite(points[[a,b],:]).all() and min(points[a,3],points[b,3])>=.5:
                        cv2.line(img,tuple(np.rint(points[a,:2]).astype(int)),tuple(np.rint(points[b,:2]).astype(int)),(60,255,160),3)
            thumb=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));thumb.thumbnail((320,270))
            x,y=n%3*320,n//3*300;sheet.paste(thumb,(x+(320-thumb.width)//2,y+30))
            ImageDraw.Draw(sheet).text((x+4,y+5),f'f{int(i)} {label} pose={bool(pose["valid"][i])}',fill='white')
    finally:capture.release()
    dest.parent.mkdir(parents=True,exist_ok=True);sheet.save(dest)
    return list(map(int,chosen))


def count_result(record,path,result,decoder):
    import av
    with av.open(str(path)) as c:ts=[float(f.time) for f in c.decode(video=0) if f.time is not None]
    origin=ts[0];gt=[ts[int(rep[2])]-origin for rep in record['reps']]
    pred=[v['back']-origin for st in result['sets'] for v in st['rep_values'] if v['back'] is not None]
    tp,fp,fn,delays=decoder.match_events(gt,pred);err=len(pred)-len(gt)
    return dict(key=record['key'],type=record['type'],gt_count=len(gt),pred_count=len(pred),
                abs_error=abs(err),exact=err==0,obo=abs(err)<=1,tp=tp,fp=fp,fn=fn,
                gt_events=gt,pred_events=pred,delays=delays)


def compatible_original_cache(old, new, *, allow_preparation_migration=False):
    if not allow_preparation_migration:
        return False
    ignored={'evaluator_sha256','prepared_sha256'}
    return ({k:v for k,v in old.items() if k not in ignored}
            == {k:v for k,v in new.items() if k not in ignored})


def run_one(item,args,decoder):
    from server.jobs import process_video
    from feedback.pose import extract_pose
    from core.config import PoseCfg
    pose_cfg=PoseCfg(backend=args.backend, mode=args.mode, det_frequency=args.det_frequency,
                     bench_rep_guard=not args.no_bench_rep_guard)
    from feedback.plate import prepare_videos, plate_rep_mask
    key,path,exercise,record=item;dest=args.out/'raw'/key;dest.mkdir(parents=True,exist_ok=True)
    external=args.pose_results/key/'pose.npz' if args.pose_results else None
    prepared=None
    if args.prepared_reference:
        reference=json.loads((args.prepared_reference/'raw'/key/'evaluation.json').read_text())
        if reference['fingerprint']['video_sha256'] != sha(path):
            raise ValueError('Prepared reference original video changed: '+key)
        prepared=Path(reference['pose_video'])
        if not prepared.is_file():raise FileNotFoundError(prepared)
    fingerprint=dict(video_sha256=sha(path),plate_model_sha256=sha(args.model),
                     pose_model_sha256=sha(ROOT/'pose_landmarker_lite.task'),
                     external_pose_sha256=sha(external) if external else None,
                     evaluator_sha256=sha(__file__),metrics_sha256=sha(Path(__file__).with_name('baseline_metrics.py')),
                     base='8e5cdb4',selection='plate-wrist-v3',exercise=exercise,
                     prepared_sha256=sha(prepared) if prepared else None,
                     backend=args.backend,mode=args.mode,det_frequency=args.det_frequency,
                     bench_rep_guard=pose_cfg.bench_rep_guard,ort_threads=args.ort_threads,
                     producer_sha256=sha(ROOT/'feedback/pose.py'),
                     adapter_sha256=sha(ROOT/'plate_pipeline/core/recog/pose.py'),
                     config_sha256=sha(ROOT/'plate_pipeline/core/config.py'),
                     quality_sha256=sha(ROOT/'feedback/quality.py'),
                     feedback_metrics_sha256=sha(ROOT/'feedback/metrics.py'))
    saved=dest/'evaluation.json'
    if args.resume and saved.exists():
        cached=json.loads(saved.read_text())
        if cached.get('cache_fingerprint',cached['fingerprint'])==fingerprint:
            print('resume',key,flush=True);return cached
        # An evaluator-only preparation fix may keep already measured ORIGINAL
        # inputs. Retain their actual measurement fingerprint; never relabel a
        # different CFR encoding or reuse its timing/pose/plate output.
        if (prepared is not None and prepared.resolve()==path.resolve()
                and Path(cached['pose_video']).resolve()==path.resolve()
                and sha(path)==fingerprint['prepared_sha256']
                and compatible_original_cache(cached['fingerprint'],fingerprint,
                                             allow_preparation_migration=args.reuse_verified_originals)):
            cached['cache_fingerprint']=fingerprint
            cached['prepared_input']=dict(path=str(prepared),sha256=sha(prepared))
            cached['reuse_provenance']='Measured original input unchanged; only evaluator preparation path changed; actual measurement fingerprint and timing retained.'
            write(saved,cached)
            print('resume verified original',key,flush=True);return cached
    start=time.perf_counter()
    settings={'user':{'video':str(path)},'ref':{'video':str(path)}}
    if prepared is None:
        prepare_videos(settings,dest)
        pose_path=Path(settings['user']['video'])
    else:
        pose_path=prepared
    prepare_s=time.perf_counter()-start
    processing_start=time.perf_counter()
    with timing() as times:
        result,rows,_=process_video(path,'bench' if exercise=='benchpress' else exercise,
                                    model=args.model,device='cpu',pose=False,out_dir=dest/'plate')
        plate_done=time.perf_counter()
        count_source=result
        if pose_path!=path:
            result,rows,_=process_video(pose_path,'bench' if exercise=='benchpress' else exercise,
                                        model=args.model,device='cpu',pose=False,out_dir=dest/'pose_plate')
        if external:
            with np.load(external,allow_pickle=False) as data:pose={k:data[k] for k in data.files}
            for k in ('fps','w','h','total'):pose[k]=pose[k].item()
        else:pose=extract_pose(pose_path,dest/'pose.npz',exercise,no_cache=True,plate_rows=rows,
                               pose_cfg=pose_cfg,rep_mask=plate_rep_mask(rows,result))
        pose['has_fingers'] = args.backend == 'mediapipe'
        inference_done=time.perf_counter()
    if 'inference_timing' in pose:
        times.update(pose['inference_timing'])
    else:
        times['person_detection']=0.
    stats,segments=measure(pose,rows,result,pose_path,exercise)
    image=args.out/'images'/f'{key.replace("/","__")}.jpg'
    image_frames=contact_sheet(pose_path,pose,segments,rows,image)
    counts=count_result(record,path,count_source,decoder) if record else None
    processing=inference_done-processing_start
    times['rest']=max(0.,processing-sum(times.values()));times['total']=processing
    times['plate_pass']=plate_done-processing_start;times['fps']=len(rows)/processing
    times['prepare']=prepare_s
    if external:times['fps']=None;times['pose']=None
    data=dict(key=key,exercise=exercise,video=str(path),pose_video=str(pose_path),fingerprint=fingerprint,
              prepared_input=dict(path=str(pose_path),sha256=sha(pose_path)),frames=len(rows),counts=counts,
              **stats,tracking=pose.get('tracking'),score_distribution=pose.get('score_distribution'),timing=times,wall_s=time.perf_counter()-start,image=str(image.relative_to(args.out)),
              image_frames=image_frames,lifter_review='pending',segment_exclusions=segments['excluded_reps'])
    write(dest/'segments.json',segments);write(saved,data)
    print(f'{key}: reps={len(segments["reps"])} coverage={stats["observation"]["coverage"]} {processing:.1f}s',flush=True)
    return data


def summarize(data,v40):
    output={}
    for exercise in ('all','squat','benchpress','deadlift'):
        videos=[v for v in data if exercise=='all' or v['exercise']==exercise]
        count_rows=[v['counts'] for v in videos if v['counts'] is not None]
        counts=aggregate(v40,pd.DataFrame(count_rows),[x for r in count_rows for x in r['delays']]) if count_rows else None
        variation=[]
        for pos in POSITIONS:
            for metric in ANGLES:
                cells=[c for v in videos for c in v['variation'] if c['position']==pos and c['metric']==metric]
                variation.append(dict(position=pos,metric=metric,videos=sum(c['sd'] is not None for c in cells),
                                      sd=median([c['sd'] for c in cells]),pair_median=median([c['pair_median'] for c in cells])))
        judged=sum(v['false_difference']['judged'] for v in videos);different=sum(v['false_difference']['different'] for v in videos)
        frames=sum(v['observation']['rep_frames'] for v in videos);selected=sum(v['observation']['pose_frames'] for v in videos)
        joints={}
        for joint in JOINTS:
            n=sum(v['observation']['joints'][joint]['visible'] for v in videos)
            d=sum(v['observation']['joints'][joint]['total'] for v in videos)
            joints[joint]=dict(visible=n,total=d,rate=ratio(n,d))
        times={k:sum(v['timing'].get(k,0.) for v in videos if v['timing'].get(k,0.) is not None)
               for k in ('decode','plate_detection','person_detection','klt','pose','rest','total')}
        times['fps']=sum(v['frames'] for v in videos)/times['total'] if videos and all(v['timing']['fps'] is not None for v in videos) else None
        output[exercise]=dict(videos=len(videos),counts=counts,variation=variation,
                              false_difference=dict(judged=judged,different=different,rate=ratio(different,judged)),
                              coverage=ratio(selected,frames),rep_frames=frames,pose_frames=selected,
                              jumps=sum(v['observation']['jumps'] for v in videos),joints=joints,timing=times)
    return output


def render(report):
    def fmt(v):return 'N/A' if v is None else f'{v:.3f}'
    lines=[f'# {report.get("backend", "mediapipe")} {report.get("mode", "")} baseline','',
           'M1b based on plate-pipeline `69102a0`. PLAN.md and main unchanged.',
           '',f'Command: `{report["command"]}`',f'Invocation wall time: {report["wall_s"]:.1f} s',
           '', 'CPU: '+report['environment']['cpu'], 'Versions: `'+json.dumps(report['environment']['versions'])+'`',
           '',f'Midterm per-video reproduction: **{report["midterm_match"]}** ({report["count_videos"]}/61 videos).',
           '', 'Counts use original midterm match_events (0.5 s) and train.summarize AST; completed back events only. F1 pools TP/FP/FN.',
           '', '|scope|videos|MAE|exact|OBO|event F1|','|---|---:|---:|---:|---:|---:|']
    for ex,s in report['summary'].items():
        c=s['counts']
        if c:lines.append(f'|{ex}|{c["videos"]}|'+ '|'.join(fmt(c[k]) for k in ('count_mae','count_exact','obo','event_f1'))+'|')
    lines+=['','## Pose observations','',
            'Denominator: unique decoded frames in completed plate rep intervals. Missing pose is failure. Joint visibility >=0.5 on raw landmarks; pooled joint x frame counts. Jump flag: consecutive-frame torso center displacement > current used plate height; a flag is not confirmed identity ground truth.',
            '', '|scope|videos|pose/rep frames|coverage|jump flags|different/judged|rate|','|---|---:|---:|---:|---:|---:|---:|']
    for ex,s in report['summary'].items():
        f=s['false_difference'];lines.append(f'|{ex}|{s["videos"]}|{s["pose_frames"]}/{s["rep_frames"]}|{fmt(s["coverage"])}|{s["jumps"]}|{f["different"]}/{f["judged"]}|{fmt(f["rate"])}|')
    lines+=['','Odd/even uses native S3b split/reindex and display metrics, >=4 measurable completed reps. Only judged cells enter denominator. Insufficient/advisory/disabled cells stay unjudged. This within-video diagnostic is not independent false-positive validation.',
            '', '|scope|'+ '|'.join(JOINTS)+'|','|---|'+ '|'.join(['---:']*len(JOINTS))+'|']
    for ex,s in report['summary'].items():lines.append('|'+ex+'|'+'|'.join(fmt(s['joints'][j]['rate']) for j in JOINTS)+'|')
    lines+=['','## Rep variation (degrees)','',
            'Per-video sample SD (ddof=1) and median absolute difference over unique unordered pairs; summary = median of video statistics per position/metric. Same S3a available-cell population including excluded reps; n<2 is N/A. Unchanged smoothing/side/raw visibility/screen gates/five-frame medians. Bench angles suppressed by existing structural occlusion remain N/A.',
            '', '|scope|position|metric|videos with >=2|SD median|pair-difference median|','|---|---|---|---:|---:|---:|']
    for ex,s in report['summary'].items():
        for c in s['variation']:lines.append(f'|{ex}|{c["position"]}|{c["metric"]}|{c["videos"]}|{fmt(c["sd"])}|{fmt(c["pair_median"])}|')
    lines+=['','## CPU speed','',
            'Serial videos, CPUExecutionProvider; ORT thread count is recorded in environment (default 1), OpenCV 1 thread. Original PTS plate pass for counts, then current feedback prepared CFR plate/pose passes where conversion is required; otherwise two passes: plate PyAV then feedback OpenCV pose every frame (RTMPose: top-down Halpe26, tracked reuses the selected lifter crop between detections; MediaPipe comparison: at most 2 poses, .45 confidence; existing EMA). Decode includes both passes; KLT includes tracker association/bookkeeping. Rest includes setup/selection/smoothing/analysis/plate I/O. fps = input frames / combined pass time; excludes CFR preparation and metric/contact-sheet/GT validation. Per-video prepare time is stored separately. RTMPose person detection and pose are measured separately including preprocessing and postprocessing; MediaPipe does not expose separate detection timing. External pose replay has no measured pose fps.',
            '', '|scope|decode s|plate detector s|person detector s|KLT s|pose s|rest s|total s|fps|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ex,s in report['summary'].items():lines.append('|'+ex+'|'+'|'.join(fmt(s['timing'][k]) for k in ('decode','plate_detection','person_detection','klt','pose','rest','total','fps'))+'|')
    lines+=['','## Per video / lifter review','',
            'Six raw skeleton frames evenly sampled over rep intervals; no completed rep gives a labeled whole-video diagnostic sheet. Sparse sheets cannot prove identity at every frame. Review and jump flags are separate.',
            '', '|video|coverage|jumps|fps|review|image|','|---|---:|---:|---:|---|---|']
    for v in report['videos']:lines.append(f'|{v["key"]}|{fmt(v["observation"]["coverage"])}|{v["observation"]["jumps"]}|{fmt(v["timing"]["fps"])}|{v["lifter_review"]}|{v["image"]}|')
    lines+=['','Confirmed wrong lifter videos: '+(', '.join(v['key']+' ('+v['image']+')' for v in report['videos'] if v['lifter_review'].startswith('wrong')) or 'none confirmed; pending is not a pass.'),
            '', '## Reuse','',
            '`--resume` requires matching source/model/evaluator hashes; timings retain original run values. `--pose-results DIR` reads DIR/<key>/pose.npz (normalized MP33 lms/raw_lms, valid, total, fps, w, h). Frame count must match plate rows. Recompute all metrics with the same command. `--videos` accepts alternate dataset/team9 roots containing the frozen cohort, not a new split.',
            '', 'Hashes, denominators, detailed cells and jump frames: baseline.json and raw/*/evaluation.json. Generated media/models/NPZ/eval_out are not committed.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--local-root',type=Path,default=local_root())
    p.add_argument('--model',type=Path);p.add_argument('--videos',type=Path,nargs='+')
    p.add_argument('--backend',choices=('rtmpose','mediapipe'),default='rtmpose')
    p.add_argument('--mode',choices=('lightweight','balanced','tracked','tiny_m'),default='tracked')
    p.add_argument('--det-frequency',type=int,default=10)
    guard=p.add_mutually_exclusive_group()
    guard.add_argument('--no-bench-rep-guard',action='store_true',dest='no_bench_rep_guard')
    guard.add_argument('--bench-rep-guard',action='store_false',dest='no_bench_rep_guard')
    p.set_defaults(no_bench_rep_guard=True)
    p.add_argument('--ort-threads',type=int,default=1)
    p.add_argument('--exercise-filter',choices=('squat','benchpress','deadlift'))
    p.add_argument('--keys',nargs='+',help='Exact baseline keys for independent evaluation shards')
    p.add_argument('--prepared-reference',type=Path,help='Reuse this baseline prepared video byte-for-byte, not its plate/pose inference')
    p.add_argument('--reuse-verified-originals',action='store_true',help='Explicit preparation-only cache migration for unchanged original inputs and numerical producers')
    p.add_argument('--pose-results',type=Path);p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.out=args.out.resolve();args.model=args.model or args.local_root/'plate_pipeline/models/plate_tiny.onnx'
    start=time.perf_counter()
    import cv2,onnxruntime as ort
    cv2.setNumThreads(1);original=ort.InferenceSession
    def session(*a,**kw):
        options=kw.pop('sess_options',None) or ort.SessionOptions();options.intra_op_num_threads=args.ort_threads;options.inter_op_num_threads=1
        return original(*a,sess_options=options,**kw)
    v40=args.local_root/'new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42'
    sys.path.insert(0,str(Path(__file__).parent/'midterm'))
    from eval.midterm.run import module
    from eval.midterm.reproduce import reproduce
    from eval.midterm.cohort import included,clock_gate
    test=included(reproduce(v40,args.out/'reproduction'))
    clock_gate(v40,args.local_root/'new_data/new_dataset',args.out/'reproduction')
    records={r['key']:r for r in json.loads((v40/'records.json').read_text())}
    roots=args.videos or [args.local_root/'new_data/new_dataset',args.local_root/'plate_pipeline/data/team/test9']
    def resolve(key):
        for root in roots:
            if root.is_file() and root.name==Path(key).name:return root.resolve()
            for candidate in (root/key,root/Path(key).name):
                if candidate.is_file():return candidate.resolve()
        raise FileNotFoundError(f'{key}: not found in --videos roots {roots}')
    items=[(key,resolve(key),records[key]['type'],records[key]) for key in test.key]
    for r in csv.DictReader((Path(__file__).parent/'lists/team9.csv').open()):
        ex='benchpress' if r['exercise']=='bench' else r['exercise'];key=ex+'/'+r['path']
        if key not in set(test.key):items.append(('team9/'+key,resolve(key),ex,None))
    if args.exercise_filter:items=[item for item in items if item[2]==args.exercise_filter]
    if args.keys:
        missing=set(args.keys)-{item[0] for item in items}
        if missing:p.error('Unknown baseline keys: '+str(sorted(missing)))
        items=[item for item in items if item[0] in args.keys]
    review_path=args.out/'lifter_review.json';reviews=json.loads(review_path.read_text()) if review_path.exists() else {}
    with patch.object(ort,'InferenceSession',session):data=[run_one(item,args,module(v40/'decode.py')) for item in items]
    for v in data:v['lifter_review']=reviews.get(v['key'],'pending')
    old=pd.read_csv(ROOT/'plate_pipeline/results/midterm/per_video.csv').set_index('key');mismatches=[]
    for v in data:
        c=v['counts']
        if c:
            diff={k:[old.loc[v['key'],k],c[k]] for k in ('pred_count','gt_count','tp','fp','fn','abs_error','exact','obo') if old.loc[v['key'],k]!=c[k]}
            if diff:mismatches.append(dict(key=v['key'],differences=diff))
    cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
    report=dict(backend=args.backend, mode=args.mode, det_frequency=args.det_frequency,
                bench_rep_guard=not args.no_bench_rep_guard, schema_version=3,command='python -m eval.baseline '+' '.join(sys.argv[1:]),wall_s=time.perf_counter()-start,
                environment=dict(cpu=cpu,python=sys.version,versions={k:importlib.metadata.version(k) for k in ('onnxruntime','rtmlib','mediapipe','numpy','av','opencv-python')},
                                 provider='CPUExecutionProvider',ort_threads=args.ort_threads,opencv_threads=1),
                count_videos=sum(v['counts'] is not None for v in data),midterm_match=not mismatches,
                midterm_mismatches=mismatches,summary=summarize(data,v40),videos=data)
    write(args.out/'baseline.json',report);markdown=render(report)
    (args.out/'baseline.md').write_text(markdown);(args.out/'REPORT.md').write_text(markdown)
    with zipfile.ZipFile(args.out/'baseline.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ('baseline.json','baseline.md'):archive.write(args.out/name,name)
        for v in data:archive.write(args.out/v['image'],v['image'])
    print(json.dumps(dict(midterm_match=not mismatches,wall_s=report['wall_s'],out=str(args.out))),flush=True)


if __name__=='__main__':main()
