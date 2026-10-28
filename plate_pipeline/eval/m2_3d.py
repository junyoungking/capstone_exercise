"""M2: selected-lifter VideoPose3D, explicit plate geometry and audit images."""
from pathlib import Path
import argparse,json,time
from dataclasses import asdict
import numpy as np
from eval.baseline import ROOT,sha,write
from core.config import DEFAULT
from core.geometry import plate_geometry,standing_frames,stature
from core.geometry.pose3d import (load_model,selected_coco,lift,smooth,fixed_bones,body_basis,
    scale_floor,to_mp33,angles_at_positions,acceleration,segments_cm,MP_FROM_H36M,PARENTS)
from eval.baseline_metrics import rep_variation
from feedback.pose3d import process
from track3d.run_pose3d import bone_length_cv,frames_for_json



def image(video,pose,segments,frames,dest,rows):
    import cv2
    from PIL import Image,ImageDraw
    from core.recog.pose import POSE_CONNECTIONS
    # Physical plate bottom (max image y), including deadlift START rather than far/top.
    index=pose['total']//2
    if segments['reps']:
        first=segments['reps'][0];a=first['phases']['down'][0];b=first['phases']['up'][1]
        candidates=[(i,rows[i]['plate']['center'][1]) for i in range(a,b) if rows[i].get('plate',{}).get('used')]
        index=max(candidates,key=lambda v:v[1])[0] if candidates else int(first['positions']['BOTTOM'])
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();cap.release()
    if not ok:raise ValueError(f'Cannot read {video} f{index}')
    raw=pose['raw_lms'][index].copy();raw[:,:2]*=[pose['w'],pose['h']]
    for a,b in POSE_CONNECTIONS:
        if np.isfinite(raw[[a,b]]).all() and min(raw[a,3],raw[b,3])>=.5:
            cv2.line(bgr,tuple(np.rint(raw[a,:2]).astype(int)),tuple(np.rint(raw[b,:2]).astype(int)),(60,255,160),3)
    left=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));left.thumbnail((640,600))
    canvas=Image.new('RGB',(1280,680),(23,27,34));canvas.paste(left,((640-left.width)//2,55+(600-left.height)//2))
    draw=ImageDraw.Draw(canvas);draw.text((15,12),f'f{index}: '+('rep 1 lowest plate frame' if segments['reps'] else 'NO COMPLETED REP: diagnostic midpoint'),fill='white')
    draw.text((660,12),'3D side: z forward ->, y up; blue=left, orange=right',fill='white')
    p=frames[index]
    if np.isfinite(p).all():
        # Fixed isotropic scale, fit frame; draw floor separately. Side projection drops lateral x.
        xy=p[:,[2,1]];lo=xy.min(axis=0);hi=xy.max(axis=0)
        factor=min(540/max(hi[0]-lo[0],.2),540/max(hi[1],.2))
        points=np.c_[960+(xy[:,0]-(lo[0]+hi[0])/2)*factor,620-xy[:,1]*factor]
        draw.line((660,620,1260,620),fill=(100,100,100),width=2)
        for j,a in enumerate(PARENTS):
            if a<0:continue
            color=(85,175,255) if j in (4,5,6,11,12,13) else (255,175,80) if j in (1,2,3,14,15,16) else (225,225,225)
            draw.line([tuple(points[a]),tuple(points[j])],fill=color,width=5)
        for j,(x,y) in enumerate(points):draw.ellipse((x-4,y-4,x+4,y+4),fill='white');draw.text((x+5,y),str(j),fill='white')
    else:draw.text((700,200),'3D UNAVAILABLE (long gap / invalid pose)',fill='white')
    dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)
    return index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='M1b n10 baseline root')
    p.add_argument('--bench',type=Path,help='New guard-off bench baseline root; must contain all16')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--diameter-mm',type=float,default=450.)
    p.add_argument('--exercise-filter',choices=('squat','deadlift','benchpress'))
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();model=load_model(a.checkpoint,a.threads);base=json.loads((a.source/'baseline.json').read_text())
    if a.bench:
        bench=json.loads((a.bench/'baseline.json').read_text());assert len(bench['videos'])==16
        replacements={v['key']:v for v in bench['videos']}
    else:replacements={}
    entries=[]
    from core.io.rows import read_rows
    for old in base['videos']:
        if a.exercise_filter and old['exercise']!=a.exercise_filter:continue
        v=replacements.get(old['key'],old);source=a.bench if old['key'] in replacements else a.source
        key=v['key'];dest=a.out/'raw'/key;dest.mkdir(parents=True,exist_ok=True)
        fingerprint=dict(pose_sha256=sha(source/'raw'/key/'pose.npz'),checkpoint_sha256=sha(a.checkpoint),
                         geometry_sha256=sha(ROOT/'plate_pipeline/core/geometry/__init__.py'),
                         lift_sha256=sha(ROOT/'plate_pipeline/core/geometry/pose3d.py'),
                         processor_sha256=sha(ROOT/'feedback/pose3d.py'),
                         producer_sha256=sha(__file__),diameter_mm=a.diameter_mm,threads=a.threads,
                         config=asdict(DEFAULT.geometry),postprocess=asdict(DEFAULT.pose3d))
        saved=dest/'result3d.json'
        if a.resume and saved.exists() and json.loads(saved.read_text()).get('fingerprint')==fingerprint:
            payload=json.loads(saved.read_text());print('resume',key,flush=True)
        else:
            with np.load(source/'raw'/key/'pose.npz',allow_pickle=False) as f:pose={k:f[k] for k in f.files}
            for field in ('w','h','total','fps'):pose[field]=pose[field].item()
            stage=source/'raw'/key/('pose_plate' if (source/'raw'/key/'pose_plate').exists() else 'plate')
            rows=read_rows(stage/'rows.jsonl.gz');result=json.loads((stage/'result.json').read_text())
            segments=json.loads((source/'raw'/key/'segments.json').read_text())
            payload,native,final=process(pose,rows,result,segments,model,a.diameter_mm)
            payload['variation3d']=rep_variation(payload['angles3d'])
            img=a.out/'images'/(key.replace('/','__')+'.jpg')
            frame=image(v['pose_video'],pose,segments,final,img,rows)
            payload.update(key=key,exercise=v['exercise'],source=str(source),fingerprint=fingerprint,
                           image=str(img.relative_to(a.out)),image_frame=frame,review='pending')
            write(saved,payload)
            np.savez_compressed(dest/'diagnostic.npz',native=native,final=final)
            print(key,round(payload['timing']['processing_s'],2),'s',payload['geometry']['stature_cm'],flush=True)
        entries.append({k:v for k,v in payload.items() if k not in ('frames','j3','angles3d')})
    write(a.out/('summary_'+a.exercise_filter+'.json' if a.exercise_filter else 'summary.json'),dict(videos=entries))


if __name__=='__main__':main()
