"""Two videos -> measured v3 JSON and a standalone, offline report.

Run from plate_pipeline: python -m server.compare --user a.mp4 --ref b.mp4
--exercise squat --out out. Set PLATE_MODEL and POSE3D_CHECKPOINT when models
are stored outside their standard local locations. No model/video is bundled.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import DEFAULT
from core.geometry import plate_geometry, standing_frames, stature
from core.geometry.pose3d import (load_model, selected_coco, lift, smooth, fixed_bones,
    body_basis, scale_floor, to_mp33, segments_cm, MP_FROM_H36M)
from core.signal.height import height_signal
from core.signal.final import clean_final
from core.judge.session import run_session, REST
from core.measure.times import all_rep_values
from core.compare.summary import summarize
from core.compare.judge import run_compare
from core.text.templates import render, summary_sentence
from feedback.pose3d import clean, set_groups
from server.findings import calculate

J2 = [0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]
SCHEMA_VERSION = 3


def prepare(user, ref, out):
    from feedback.plate import prepare_videos
    from tools.check_video import ffprobe, fix
    settings = {role: dict(video=str(Path(path).resolve())) for role,path in (('user',user),('ref',ref))}
    prepare_videos(settings,out/'prepared')
    for role,item in settings.items():
        source = Path(item['video'])
        # Existing preparation accepts VP9/AV1; this command promises H.264.
        if ffprobe(source)['codec'] != 'h264':
            source = fix(source,out/'prepared'/role)
        dest = out/f'{role}.mp4'
        if source.resolve()!=dest.resolve():
            shutil.copy2(source,dest)
        item['video'] = str(dest)
    return settings


def aligned_3d(pose, rows, result, model, target_height_cm):
    raw = np.asarray(pose['raw_lms'],float).copy()
    raw[:,:,:2] *= [pose['w'],pose['h']]
    stand = standing_frames(raw)
    native,_,_ = lift(model,selected_coco(pose),pose['w'],pose['h'])
    fixed,_ = fixed_bones(smooth(native,pose['fps']),set_groups(rows,result)[0])
    basis,warnings = body_basis(fixed,stand,raw)
    oriented = np.einsum('ij,tkj->tki',basis,fixed)
    # Input height is independent of plate-derived stature. If no supported
    # standing frame exists, use the tallest valid pose as an explicit fallback.
    valid = np.isfinite(oriented).all(axis=(1,2))
    support = stand & valid
    heights = oriented[:,9,1]-oriented[:,[3,6],1].min(axis=1)
    support &= heights>1e-6
    if not support.any() and target_height_cm is not None and np.any(valid & (heights>0)):
        support[np.nanargmax(np.where(valid,heights,np.nan))] = True
        warnings.append('No supported standing frame: tallest valid 3D frame used for input height')
    final,scale,reason = scale_floor(oriented,support,None if target_height_cm is None else target_height_cm/100)
    if reason:
        warnings.append(reason)
    actual_height = float(np.median(heights[support])*scale/.92*100) if support.any() else None
    if actual_height is None and valid.any():
        actual_height = float(np.max(heights[valid])*scale/.92*100)
    return raw,stand,final,actual_height,reason is None,warnings


def build_video(rows, result, pose, frames, raw, stand, diameter, height_cm, scaled):
    t = np.asarray([r['t'] for r in rows])
    fps,w,h = pose['fps'],pose['w'],pose['h']
    ex = result['exercise']
    st,sy,sx = height_signal(rows,diameter)
    fr = clean_final(st,sy,sx,result['fps'],DEFAULT.final)
    sets = run_session(fr.sig.t,fr.sig.y,fr.sig.x,fr.sig.gap_before,ex,'final',cfg=DEFAULT)
    trips = [r for s in sets for r in s.reps]
    vals = all_rep_values(fr.sig,fr.raw_t,trips,REST[ex],DEFAULT.speed)
    good = [(trip,v) for trip,v in zip(trips,vals) if all(getattr(v,k) is not None for k in ('start','far_in','far_out','back','down','up'))]
    if not good:
        raise ValueError('완료 rep이 없습니다: '+result['video'])
    reps = []
    for _,v in good:
        lo,hi = sorted((v.far_in,v.far_out))
        # For deadlift the far point is at the top, so depth means excursion
        # away from the resting height, just as for downward-start exercises.
        reps.append(dict(n=len(reps)+1,start=v.start,far_in=lo,far_out=hi,far=(lo+hi)/2,
                         back=v.back,mid_down=(v.start+lo)/2,mid_up=(hi+v.back)/2,
                         rom_cm=round(v.rom*100,1),v=None if v.v_mean is None else round(v.v_mean,3),blank=v.blank,down=round(v.down,2),
                         pause=round(v.far_pause or 0,2),up=round(v.up,2)))
    heights = np.interp(t,fr.sig.t,fr.sig.y)*100
    knots = sorted((r[k],float(np.interp(r[k],t,heights))) for r in reps for k in ('start','back'))
    baseline = np.interp(t,[x[0] for x in knots],[x[1] for x in knots])
    depth = np.maximum(0,(heights-baseline) if REST[ex]=='bottom' else (baseline-heights))
    boxes = np.array([r['plate']['box'] if r.get('plate',{}).get('used') else [np.nan]*4 for r in rows])
    g = plate_geometry(boxes,(w,h),diameter)
    measured_height,_ = stature(raw,g,stand)
    geo = dict(cam_angle=g['angle_deg'],box_ratio=g['ratio'],dist_m=float(np.median(g['distance_m'])),
               kd=float(np.median(g['kd'])),focal_px=g['focal_px'],cm_per_px=float(np.median(g['plate_m_per_px']))*100,
               stature_cm=height_cm)
    seg = segments_cm(frames) if scaled else {}
    for dst,src in (('thigh_cm','thigh'),('shin_cm','shin'),('torso_cm','torso'),('upper_cm','upper_arm'),('fore_cm','forearm')):
        geo[dst] = seg.get(src)
    checks = dict(stature_plate_cm=None if measured_height is None else measured_height*100,
                  warn=bool(height_cm is not None and measured_height is not None and abs(measured_height*100-height_cm)/height_cm>.15))
    plate = [[round(p['center'][0],1),round(p['center'][1],1),round(sum(p['size_px'])/4,1)] if (p:=r.get('plate',{})).get('used') else None for r in rows]
    # Smoothed selected-lifter 2D; visibility comes from the original observations.
    kp = np.asarray(pose['lms'],float).copy()
    kp[:,:,:2] *= [w,h]
    kp2 = []
    for i,p in enumerate(kp):
        kp2.append([[round(float(p[j,0]),1),round(float(p[j,1]),1),round(float(raw[i,j,3]),2)] if np.isfinite(p[j,:2]).all() else None for j in J2] if pose['valid'][i] else None)
    xs,ys = [],[]
    for q in kp2:
        if q:
            for pt in q:
                if pt and pt[2]>=.3:
                    xs.append(pt[0]);ys.append(pt[1])
    pp = np.array([p for p in plate if p is not None])
    x0,x1 = (np.percentile(xs,1),np.percentile(xs,99)) if xs else (0,w)
    y0,y1 = (np.percentile(ys,.5),np.percentile(ys,99.5)) if ys else (0,h)
    x0,x1 = min(x0,np.min(pp[:,0]-pp[:,2])),max(x1,np.max(pp[:,0]+pp[:,2]))
    y0,y1 = min(y0,np.min(pp[:,1]-pp[:,2])),max(y1,np.max(pp[:,1]+pp[:,2]))
    padx,pady = .1*(x1-x0),.06*(y1-y0)
    crop = [round(max(0,x0-padx)),round(max(0,y0-pady)),round(min(w,x1+padx)),round(min(h,y1+pady))]
    mp = to_mp33(frames)
    names = list(MP_FROM_H36M)
    j3 = [q[names].tolist() if np.isfinite(q[names]).all() else None for q in mp]
    data = dict(fps=fps,width=w,height=h,duration=len(t)/fps,t=t.tolist(),plate=plate,
                depth=np.round(depth,1).tolist(),reps=reps,plate_used=100*sum(p is not None for p in plate)/len(plate),
                crop=crop,geo=geo,geo_check=checks,j2=J2,kp2=kp2,j3names=names,j3=j3,clips=[])
    from server.findings import prep_reps,pos_time
    prep_reps(data)
    for rep in reps:
        rep['mid_down'],rep['mid_up'] = pos_time(rep,'MID_DOWN'),pos_time(rep,'MID_UP')
    return data,[x[0] for x in good],[x[1] for x in good]


def write_report(bundle, dest):
    template = (Path(__file__).parent/'web/template3.html').read_text()
    vendor = '/*\n'+(ROOT/'web/vendor/three.LICENSE.txt').read_text()+'\n*/\n'+(ROOT/'web/vendor/three.min.js').read_text()
    template = template.replace('/*__THREE__*/','<script>'+vendor+'</script>')
    # Avoid ending an inline script through user-controlled text.
    encoded = json.dumps(clean(bundle),ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    (Path(dest)/'report.html').write_text(template.replace('/*__BUNDLE__*/null',encoded))


def compare(user,ref,exercise,out,plate_mm=450.,load_kg=None,height_cm=None,threads=6,model=None,checkpoint=None):
    import cv2
    import onnxruntime as ort
    from feedback.pose import extract_pose
    from feedback.plate import plate_rep_mask
    from server.jobs import process_video
    out = Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    for name,value in (('plate_mm',plate_mm),('load_kg',load_kg),('height_cm',height_cm)):
        if value is not None and (not np.isfinite(value) or value<=0):
            raise ValueError(name+' must be positive and finite')
    started = time.perf_counter()
    model = Path(model or os.environ.get('PLATE_MODEL',ROOT/'plate_pipeline/models/plate_tiny.onnx'))
    checkpoint = Path(checkpoint or os.environ.get('POSE3D_CHECKPOINT',Path.home()/'.cache/videopose3d/pretrained_h36m_detectron_coco.bin'))
    if not model.is_file(): raise FileNotFoundError(f'원판 모델이 없습니다: {model} (PLATE_MODEL로 지정)')
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    cv2.setNumThreads(1)
    original = ort.InferenceSession
    def session(*args,**kwargs):
        opts = kwargs.pop('sess_options',None) or ort.SessionOptions()
        opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
        return original(*args,sess_options=opts,**kwargs)
    settings = prepare(user,ref,out)
    vp3d = load_model(checkpoint,threads)
    b = dict(schema_version=SCHEMA_VERSION,exercise=exercise,inputs=dict(plate_mm=plate_mm,load_kg=load_kg,height_cm=height_cm),
             show_3d=exercise!='benchpress',pose3d_scale='input_height' if height_cm is not None else 'model_native')
    stats,timing = {},{}
    ref_height = height_cm
    with patch.object(ort,'InferenceSession',session):
        for role in ('user','ref'):
            start = time.perf_counter();dest = out/'analysis'/role
            result,rows,_ = process_video(settings[role]['video'],'bench' if exercise=='benchpress' else exercise,
                                          plate_mm=plate_mm,model=str(model),device='cpu',pose=False,out_dir=dest/'plate')
            pose = extract_pose(settings[role]['video'],dest/'pose.npz',exercise,no_cache=True,plate_rows=rows,
                                rep_mask=plate_rep_mask(rows,result),pose_cfg=DEFAULT.pose)
            raw,stand,frames,actual,scaled,warnings = aligned_3d(pose,rows,result,vp3d,ref_height)
            b[role],trips,vals = build_video(rows,result,pose,frames,raw,stand,plate_mm,height_cm,scaled and height_cm is not None)
            b[role]['pose3d_warnings'] = warnings
            b[role]['pose3d_height_cm'] = actual
            if role=='user': ref_height = height_cm if height_cm is not None else actual
            stats[role] = (trips,vals,summarize(result['exercise'],result['fps'],trips,vals))
            timing[role] = time.perf_counter()-start
    b['sentences'] = [dict(kind=s.kind,text=s.text,key=s.key,shown=s.shown) for s in render(run_compare(stats['user'][2],stats['ref'][2]),'bench' if exercise=='benchpress' else exercise)]
    b['set_median'] = {role:{key:round(median,5) for key,attr in (('down','down'),('pause','far_pause'),('up','up')) if np.isfinite(median:=float(np.median([getattr(v,attr) or 0 for v in stats[role][1]])))} for role in ('user','ref')}
    vs = [r['v'] for r in b['user']['reps']]
    valid_speeds = [v for v in vs if v is not None]
    b['v_loss'] = round(100*(max(valid_speeds)-vs[-1])/max(valid_speeds),1) if valid_speeds and max(valid_speeds)>0 and vs[-1] is not None else None
    b['summary'] = summary_sentence('bench' if exercise=='benchpress' else exercise,len(vs),vs,b['v_loss']).text
    b['findings'],b['angle_judgments'] = calculate(b)
    b['kpi'] = b['angle_judgments']['FAR']['knee']
    b['timing'] = dict(**timing,total_s=time.perf_counter()-started,threads=threads,scope='preparation, model loading, inference, analysis and judgments; excludes JSON/HTML serialization')
    b = clean(b)
    from server.schema import validate_result
    validate_result(b)
    (out/'result_v3.json').write_text(json.dumps(b,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    write_report(b,out)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--user',required=True,type=Path);ap.add_argument('--ref',required=True,type=Path)
    ap.add_argument('--exercise',required=True,choices=('squat','benchpress','deadlift'))
    ap.add_argument('--plate-mm',type=float,default=450.);ap.add_argument('--load-kg',type=float);ap.add_argument('--height-cm',type=float)
    ap.add_argument('--out',required=True,type=Path)
    a = ap.parse_args()
    b = compare(**vars(a))
    print(json.dumps(dict(reps={r:len(b[r]['reps']) for r in ('user','ref')},timing=b['timing']),ensure_ascii=False))

if __name__=='__main__': main()
