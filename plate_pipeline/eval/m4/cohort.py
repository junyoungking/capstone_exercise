"""Replay frozen 69 plate observations against the original videos for M4."""
from pathlib import Path
import concurrent.futures, csv, json, sys, time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io import video
from core.io.rows import read_rows, write_rows
from core.recog.ellipse import fit_plate_ellipse, interpolate_ellipses, smooth_ellipses
from core.geometry import plate_geometry, standing_frames, stature
from server.jobs import analyse

HOME=Path('/home/kojy1009/capstone_exercise')
BASE=Path('/home/kojy1009/work/m0-baseline/plate_pipeline/eval_out/baseline_mp/raw')
M2=Path('/home/kojy1009/work/m2-3d/plate_pipeline/eval_out/m2/pose3d/summary.json')
OUT=ROOT/'results/m4';TMP=ROOT/'eval_out/m4'; TMP.mkdir(parents=True,exist_ok=True)


def run(item):
    cv2.setNumThreads(1)
    key=item['key']; ex=item['exercise']; path=HOME/'new_data/new_dataset'/key
    cache=TMP/key/'summary.json'
    # Cached rows preserve fitted measurements for reproducible reporting.
    cached_rows=TMP/key/'rows.jsonl.gz'
    if cache.exists(): return json.loads(cache.read_text())
    if key.startswith('team9/'): path=HOME/'plate_pipeline/data/team/test9'/key.removeprefix('team9/')
    base=BASE/key;rows=read_rows(base/'plate/rows.jsonl.gz');res=json.loads((base/'plate/result.json').read_text())
    pose_base=Path(item['source']); pose_base = pose_base if pose_base.is_absolute() else Path('/home/kojy1009/work/m2-3d/plate_pipeline')/pose_base
    pose_path=pose_base/'raw'/key/'pose.npz'
    if not path.is_file(): return {'key':key,'error':'video missing'}
    info=video.info(path);step=max(1,round(info.fps*.25))
    used_indices=[i for i,r in enumerate(rows) if r.get('plate',{}).get('used')]
    targets=[used_indices[round((len(used_indices)-1)*q)] for q in (.2,.5,.8)] if used_indices else []
    targets=[min(used_indices,key=lambda i: abs(i-round(x/step)*step)) for x in targets];shots={};t0=time.perf_counter()
    for i,(_,im) in enumerate(video.frames(path)):
        if i>=len(rows):break
        p=rows[i].get('plate',{})
        if not p.get('used'):continue
        sampled=i%step==0
        fit=fit_plate_ellipse(im,p['box']) if sampled else None
        p['ellipse']={**fit,'sampled':True} if fit else {'source':'box','quality':0.,'sampled':sampled}
        if i in targets:shots[i]=im.copy()
    interpolate_ellipses(rows);smooth_ellipses(rows)
    elapsed=time.perf_counter()-t0
    use=[r for r in rows if r.get('plate',{}).get('used')]; ell=[r['plate']['ellipse'] for r in use if r['plate'].get('ellipse',{}).get('source')=='ellipse']
    boxes=np.array([r['plate']['box'] if r.get('plate',{}).get('used') else [np.nan]*4 for r in rows]);es=[r.get('plate',{}).get('ellipse') for r in rows]
    boxgeo=plate_geometry(boxes,(info.width,info.height));ge=plate_geometry(boxes,(info.width,info.height),ellipses=es)
    try:
        with np.load(pose_path) as f: raw=f['raw_lms'].copy(); raw[:,:,:2]*=[info.width,info.height]
        stand=standing_frames(raw); hb,_=stature(raw,boxgeo,stand);he,_=stature(raw,ge,stand)
    except Exception: hb=he=None
    data={'key':key,'exercise':ex,'frames':len(rows),'used':len(use),'ellipse_success':len(ell)/len(use) if use else 0,
          'sampled_success':sum(e.get('sampled',False) for e in ell),'elapsed_s':elapsed,
          'major_to_box_h':float(np.median([r['plate']['ellipse']['major']/r['plate']['size_px'][1] for r in use if r['plate'].get('ellipse',{}).get('source')=='ellipse'])) if ell else None,
          'box_ratio':boxgeo['ratio'],'box_angle':boxgeo['angle_deg'],'ellipse_angle':ge['angle_deg'] if ell else None,
          'stature_box_cm':None if hb is None else hb*100,'stature_ellipse_cm':None if he is None else he*100}
    for mode,position in [('box','box'),('ellipse','box'),('ellipse_center','ellipse')]:
        sets,_=analyse(rows,'bench' if ex=='benchpress' else ex,res['fps'],scale_source='box' if mode=='box' else 'ellipse',position_source=position)
        vals=[r for s in sets for r in s['rep_values']]
        data[mode+'_reps']=sum(v['back'] is not None for v in vals)
        data[mode+'_events']=[v['back'] for v in vals if v['back'] is not None]
        data[mode+'_rom_cm']=[v['rom_m']*100 for v in vals if v['rom_m'] is not None]
        data[mode+'_velocity']=[v['v_mean'] for v in vals if v['v_mean'] is not None]
    dst=TMP/key;dst.mkdir(parents=True,exist_ok=True);write_rows(dst/'rows.jsonl.gz',rows)
    panels=[]
    for i in targets:
        if i not in shots:
            panel=np.zeros((400,640,3),np.uint8);cv2.putText(panel,'no used plate frame',(20,180),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        else:
            panel=shots[i];p=rows[i]['plate'];e=p['ellipse'];x0,y0,x1,y1=map(int,p['box']);cv2.rectangle(panel,(x0,y0),(x1,y1),(255,100,30),3)
            if e.get('source')=='ellipse':cv2.ellipse(panel,tuple(map(lambda x:int(round(x)),e['center'])),(int(e['major']/2),int(e['minor']/2)),float(np.degrees(e['angle'])),0,360,(40,245,190),3)
            # Keep the complete frame in each panel (never crop tall phone video).
            scale=min(640/panel.shape[1],400/panel.shape[0]); panel=cv2.resize(panel,(max(1,int(panel.shape[1]*scale)),max(1,int(panel.shape[0]*scale))))
            canvas=np.zeros((400,640,3),np.uint8);y=(400-panel.shape[0])//2;x=(640-panel.shape[1])//2;canvas[y:y+panel.shape[0],x:x+panel.shape[1]]=panel;panel=canvas
            cv2.putText(panel,f'frame {i} {e["source"]}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        panels.append(panel)
    cv2.imwrite(str(OUT/'overlays'/key.replace('/','__').replace('.mp4','.jpg')),np.hstack(panels))
    cache.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(key,round(data['ellipse_success'],2),data['box_reps'],data['ellipse_reps'],flush=True)
    return data

if __name__=='__main__':
    items=json.loads(M2.read_text())['videos']
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:data=list(pool.map(run,items))
    (OUT/'cohort.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print('finished',len(data),flush=True)
