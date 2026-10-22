#!/usr/bin/env python3
"""Reproducible S3a observations, pooled distance measurement and skeleton sheets.

record runs unchanged plate recognition and stores every raw MediaPipe candidate.
measure uses only exactly-one-candidate frames with a plate in use. No labels are
inferred from candidate count: the resulting distances include detection errors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from feedback.plate import prepare_videos
from core.io.rows import read_rows
from core.recog.pose import person_box, near_wrist, select_lifter, person_center


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def record(video, exercise, out, model=None, box=None):
    from server.jobs import process_video
    from unified_feedback_v4 import create_landmarker, extract_all_landmarks, MODEL_PATH

    out.mkdir(parents=True, exist_ok=True)
    settings={'user':{'video':str(video)},'ref':{'video':str(video)}}
    prepare_videos(settings,out)
    path=Path(settings['user']['video'])
    plate_exercise='bench' if exercise=='benchpress' else exercise
    result,rows,_=process_video(path,plate_exercise,init_box=box,model=model,pose=False,
                               device='cpu',out_dir=out/'plate')
    capture=cv2.VideoCapture(str(path))
    fps=capture.get(cv2.CAP_PROP_FPS)
    width,height=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames=[]; counts=[]
    try:
        with create_landmarker(MODEL_PATH) as landmarker:
            while True:
                ok,image=capture.read()
                if not ok: break
                seq=len(frames)
                candidates=extract_all_landmarks(landmarker,image,int(seq/fps*1000))
                data=np.full((2,33,4),np.nan,np.float32)
                for i,candidate in enumerate(candidates[:2]):data[i]=candidate
                counts.append(min(2,len(candidates)));frames.append(data)
    finally:
        capture.release()
    if len(frames)!=len(rows):raise ValueError(f'rows={len(rows)}, pose={len(frames)}')
    np.savez_compressed(out/'candidates.npz',lms=np.asarray(frames),counts=np.asarray(counts),
                        fps=fps,w=width,h=height,total=len(frames),video=str(path))
    write_json(out/'observation.json',{'original_video':str(video),'video':str(path),'exercise':exercise,
              'frames':len(frames),'fps':fps,'size':[width,height],
              'plate_rep_count':sum(s['reps'] for s in result['sets']),
              'candidate_counts':dict(Counter(counts)), 'base_commit':'f84af9d'})
    print(out.name,'frames',len(frames),'reps',sum(s['reps'] for s in result['sets']),flush=True)


def candidates_at(data, seq):
    return [p.astype(float).copy()*[int(data['w']),int(data['h']),1,1]
            for p in data['lms'][seq,:int(data['counts'][seq])]]


def distance_samples(directory):
    rows=read_rows(directory/'plate/rows.jsonl.gz')
    values=[]
    with np.load(directory/'candidates.npz') as data:
        for seq,row in enumerate(rows):
            plate=row.get('plate',{})
            if int(data['counts'][seq])!=1 or not plate.get('used'):continue
            box=plate['box'];h=box[3]-box[1]
            if h<=0:continue
            p=candidates_at(data,seq)[0]
            bounds=person_box(p)
            if bounds is None:continue
            center=(np.asarray(box[:2])+box[2:])/2
            a,b,c,d=bounds;cx,cy=center
            box_distance=float(np.hypot(max(a-cx,0,cx-c),max(b-cy,0,cy-d)))/h
            wrist=near_wrist(p[:,[0,1,3]],box,vis_thr=.5,frame_h=int(data['h']),frame_w=int(data['w']))
            wrist_distance=float(np.linalg.norm(np.asarray(wrist)-center)/h) if wrist is not None else None
            values.append({'seq':seq,'t':row['t'],'box':box_distance,'wrist':wrist_distance,
                           'ranking':wrist_distance if wrist_distance is not None else box_distance})
    return values


def quantiles(samples):
    return {'n':len(samples), **dict(zip(('p50','p90','p99'),map(float,np.percentile(samples,[50,90,99]))))} if samples else {'n':0,'p50':None,'p90':None,'p99':None}


def measure(root):
    videos={};pooled={};all_samples=[]
    for directory in sorted(root.iterdir()):
        if not (directory/'observation.json').is_file():continue
        meta=json.loads((directory/'observation.json').read_text()); samples=distance_samples(directory)
        videos[directory.name]={'exercise':meta['exercise'],'plate_rep_count':meta['plate_rep_count'],
              'distances':{key:quantiles([s[key] for s in samples if s[key] is not None]) for key in ['box','wrist','ranking']},'samples':samples}
        pooled.setdefault(meta['exercise'],[]).extend(samples); all_samples.extend(samples)
    by_exercise={ex:{key:quantiles([s[key] for s in samples if s[key] is not None]) for key in ['box','wrist','ranking']} for ex,samples in pooled.items()}
    overall={key:quantiles([s[key] for s in all_samples if s[key] is not None]) for key in ['box','wrist','ranking']}
    # One shared limit: conservative envelope of the two requested pooled p99s.
    limit=math.ceil(max(overall[key]['p99'] for key in ['box','wrist']))
    result={'base_commit':'f84af9d','units':'plate box height','population':'exactly one raw MediaPipe candidate and plate.used',
            'percentile_method':'numpy percentile linear; pooled individual frames, not averages of videos',
            'videos':videos,'by_exercise':by_exercise,'overall':overall,'acceptance_limit':limit,
            'limit_rule':'ceil(max(p99(all box distances), p99(all visible-wrist distances)))'}
    from importlib.metadata import version
    result['environment'] = {'python':sys.version, **{package:version(package) for package in
        ('numpy','opencv-python','mediapipe','av','onnxruntime')}}
    write_json(root.parent/'stage0_distances.json',result)
    print(json.dumps({'overall':overall,'by_exercise':by_exercise,'acceptance_limit':limit},indent=2),flush=True)


def selection(directory):
    rows=read_rows(directory/'plate/rows.jsonl.gz')
    meta=json.loads((directory/'observation.json').read_text())
    indices=[];prev=None
    with np.load(directory/'candidates.npz') as data:
        for seq,row in enumerate(rows):
            cands=candidates_at(data,seq)
            box=row.get('plate',{}).get('box') if row.get('plate',{}).get('used') else None
            index=select_lifter(cands,box,prev,meta['exercise'],image_size=tuple(meta['size']))
            indices.append(index)
            if index is not None:
                center=person_center(cands[index])
                if center is not None:prev=center
    return indices


def sheets(root):
    from PIL import Image,ImageDraw,ImageFont
    from core.recog.pose import POSE_CONNECTIONS
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    target=root.parent/'skeletons';target.mkdir(exist_ok=True)
    for directory in sorted(root.iterdir()):
        if not (directory/'observation.json').is_file():continue
        meta=json.loads((directory/'observation.json').read_text()); chosen=selection(directory)
        write_json(directory/'selected_indices.json',chosen)
        rows=read_rows(directory/'plate/rows.jsonl.gz')
        with np.load(directory/'candidates.npz') as data:
            fps=float(data['fps']);seconds=list(range(math.ceil(len(chosen)/fps)))
            columns=5;tw,th=320,260
            sheet=Image.new('RGB',(columns*tw,math.ceil(len(seconds)/columns)*th),(25,28,35))
            capture=cv2.VideoCapture(meta['video'])
            for i,second in enumerate(seconds):
                seq=min(len(chosen)-1,round(second*fps));capture.set(cv2.CAP_PROP_POS_FRAMES,seq)
                ok,image=capture.read()
                if not ok:continue
                index=chosen[seq]
                plate=rows[seq].get('plate',{})
                if plate.get('used'):
                    a,b,c,d=map(round,plate['box']);cv2.rectangle(image,(a,b),(c,d),(255,160,20),2)
                if index is not None:
                    p=candidates_at(data,seq)[index]
                    for a,b in POSE_CONNECTIONS:
                        if min(p[a,3],p[b,3])>=.3:
                            cv2.line(image,tuple(np.rint(p[a,:2]).astype(int)),tuple(np.rint(p[b,:2]).astype(int)),(0,255,140),3)
                    for x,y,z,vis in p:
                        if vis>=.3:cv2.circle(image,(round(x),round(y)),4,(20,240,250),-1)
                thumb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));thumb.thumbnail((tw,th-30))
                x,y=(i%columns)*tw,(i//columns)*th
                sheet.paste(thumb,(x+(tw-thumb.width)//2,y+30))
                ImageDraw.Draw(sheet).text((x+6,y+6),f'{second}s  f{seq}  selected={index} / {int(data["counts"][seq])}',font=font,fill='white')
            capture.release();sheet.save(target/f'{directory.name}.png')
        print('sheet',directory.name,sum(index is not None for index in chosen),len(chosen),flush=True)



def replay_pose(directory):
    """Replay exactly the live selection/smoothing on recorded raw candidates."""
    from feedback.pose import choose_plate_candidate
    from unified_feedback_v4 import LandmarkSmoother, valid_lms
    meta=json.loads((directory/'observation.json').read_text())
    rows=read_rows(directory/'plate/rows.jsonl.gz')
    smoother=LandmarkSmoother();previous=None
    with np.load(directory/'candidates.npz') as data:
        total=len(rows);lms=np.full((total,33,4),np.nan,np.float32);raw=lms.copy()
        valid=np.zeros(total,bool)
        for seq,row in enumerate(rows):
            candidates=[p for p in data['lms'][seq,:int(data['counts'][seq])]]
            selected,previous=choose_plate_candidate(candidates,row,previous,meta['exercise'],tuple(meta['size']))
            if valid_lms(selected):
                valid[seq]=True;raw[seq]=selected;lms[seq]=smoother.update(selected)
        return {'lms':lms,'raw_lms':raw,'valid':valid,'total':total,'fps':float(data['fps']),
                'w':int(data['w']),'h':int(data['h']),'video':str(Path(meta['video']).resolve()),
                'mtime_ns':Path(meta['video']).stat().st_mtime_ns,'cache_hit':False}


def replay(root, phase):
    """Run the real plate CLI with model I/O replaced by the recorded observations.

    Only recognition/MediaPipe I/O is replayed; selection, smoothing, gates,
    segmentation, measurement, comparison and export execute current source.
    This avoids rerunning stochastic recognition between the before/after arms.
    """
    from unittest.mock import patch
    import run_demo
    destination=root.parent/phase;destination.mkdir(exist_ok=True)
    poses={};observations={}
    for directory in sorted(root.iterdir()):
        if not (directory/'observation.json').is_file():continue
        meta=json.loads((directory/'observation.json').read_text())
        poses[directory.name]=replay_pose(directory)
        observations[directory.name]=(meta,read_rows(directory/'plate/rows.jsonl.gz'),
                                  json.loads((directory/'plate/result.json').read_text()))
    jobs=[(name,meta['exercise'],name,name,1,'metrics') for name,(meta,_,_) in observations.items() if name!='public']
    jobs += [('public','squat','public','public',3,'report'),
             ('squat','squat','stu10_70','stu8_67',1,'report'),
             ('bench','benchpress','stu2_10_edited','stu2_10_edited',1,'report'),
             ('dead','deadlift','stu7_1_1','deadlift_08',1,'report')]
    for name,exercise,user,reference,ref_rep,until in jobs:
        dest=destination/name;dest.mkdir(exist_ok=True)
        settings={'exercise':exercise,'user':{'video':poses[user]['video'],'view':'side'},
                  'ref':{'video':poses[reference]['video'],'view':'side','rep':ref_rep}}
        if name=='public':
            for role in ('user','ref'):settings[role]['plate_box']=[710,920,1075,1290]
        write_json(dest/'config.json',settings)
        by_path={poses[key]['video']:key for key in [user,reference]}
        def tracked(settings, exercise, out, **kwargs):
            key=by_path[str(Path(settings['video']).resolve())]
            return observations[key][1:]
        def estimated(video, cache, exercise, **kwargs):
            key=by_path[str(Path(video).resolve())];pose=poses[key].copy()
            np.savez_compressed(cache,**{k:v for k,v in pose.items() if k!='cache_hit'})
            return pose
        with patch('feedback.plate.analyze_video',tracked),patch('feedback.pose.extract_pose',estimated):
            try:
                code=run_demo.main(['--config',str(dest/'config.json'),'--out',str(dest),'--segment','plate','--until',until])
            except SystemExit as error:
                code=error.code
        write_json(dest/'replay.json',{'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'phase':phase,'exit_code':code,'model_io':'fresh stage0 rows and raw candidates replayed; no thresholds replaced',
             'user_observation':user,'ref_observation':reference})
        if code:print('REPLAY ERROR',name,code,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('record','measure','sheets','replay'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--video',type=Path);parser.add_argument('--exercise')
    parser.add_argument('--phase',choices=('before','after'))
    parser.add_argument('--model');parser.add_argument('--box',type=str)
    args=parser.parse_args()
    if args.stage=='record' and (args.video is None or args.exercise is None):
        parser.error('record requires --video and --exercise')
    if args.stage=='replay' and args.phase is None:
        parser.error('replay requires --phase before|after')
    if args.stage=='record':record(args.video,args.exercise,args.out,args.model,
                                  [float(v) for v in args.box.split(',')] if args.box else None)
    elif args.stage=='measure':measure(args.out)
    elif args.stage=='sheets':sheets(args.out)
    else:replay(args.out,args.phase)

if __name__=='__main__':main()
