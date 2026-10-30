"""Two videos -> measured v3 JSON and a standalone, offline report.

Run from plate_pipeline: python -m server.compare --user a.mp4 --ref b.mp4
--exercise squat --out out. Set PLATE_MODEL and POSE3D_CHECKPOINT when models
are stored outside their standard local locations. No model/video is bundled.
"""
import argparse
from dataclasses import replace
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
from core.geometry import (plate_geometry, standing_frames, stature, retake_warning,
    box_angle_reliable, implied_plate_diameter, plate_suggestion)
from core.geometry.pose3d import (load_model, selected_coco, lift, smooth, fixed_bones,
    body_basis, scale_floor, to_mp33, segments_cm, MP_FROM_H36M)
from core.signal.height import corrected_height_signal
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
    settings = {role: dict(video=str(Path(path).resolve())) for role,path in (('user',user),('ref',ref)) if path is not None}
    for role,item in settings.items():
        pair = {'user':item.copy(),'ref':item.copy()}
        prepare_videos(pair,out/'prepared'/role)
        settings[role] = pair['user']
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


def aligned_3d(pose, rows, result, model, target_height_cm, native=None):
    raw = np.asarray(pose['raw_lms'],float).copy()
    raw[:,:,:2] *= [pose['w'],pose['h']]
    stand = standing_frames(raw)
    if native is None:
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


def build_video(rows, result, pose, frames, raw, stand, diameter, height_cm, scaled, scale_source="box", position_source="box", is_reference=False, reuse_video=None, speed_cfg=DEFAULT.speed):
    t = np.asarray([r['t'] for r in rows])
    fps,w,h = pose['fps'],pose['w'],pose['h']
    ex = result['exercise']
    st,sy,sx = corrected_height_signal(rows,diameter,scale_source,position_source)
    fr = clean_final(st,sy,sx,result['fps'],DEFAULT.final)
    sets = run_session(fr.sig.t,fr.sig.y,fr.sig.x,fr.sig.gap_before,ex,'final',cfg=DEFAULT)
    trips = [r for s in sets for r in s.reps]
    vals = all_rep_values(fr.sig,fr.raw_t,trips,REST[ex],speed_cfg)
    good = [(trip,v) for trip,v in zip(trips,vals) if all(getattr(v,k) is not None for k in ('start','far_in','far_out','back','down','up'))]
    if not good and reuse_video is None:
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
    baseline = np.interp(t,[x[0] for x in knots],[x[1] for x in knots]) if knots else np.full_like(heights,heights[0])
    depth = np.maximum(0,(heights-baseline) if REST[ex]=='bottom' else (baseline-heights))
    boxes = np.array([r['plate']['box'] if r.get('plate',{}).get('used') else [np.nan]*4 for r in rows])
    g = plate_geometry(boxes,(w,h),diameter)
    ellipse_g = plate_geometry(boxes,(w,h),diameter,ellipses=[r.get("plate",{}).get("ellipse") for r in rows])
    reliable = box_angle_reliable(boxes,(w,h))
    measured_height,_ = stature(raw,g,stand)
    geo = dict(cam_angle=g['angle_deg'],box_ratio=g['ratio'],dist_m=float(np.median(g['distance_m'])),
               kd=float(np.median(g['kd'])),focal_px=g['focal_px'],cm_per_px=float(np.median(g['plate_m_per_px']))*100,
               stature_cm=height_cm,angle_source="box",angle_reliable=reliable,ellipse_stats=dict(count=ellipse_g["ellipse_count"],
               fraction=ellipse_g["ellipse_used_fraction"],major_px=ellipse_g["ellipse_major_px"],minor_px=ellipse_g["ellipse_minor_px"],tilt_deg=ellipse_g["ellipse_tilt_deg"]),
               retake_warn=reliable and retake_warning(g["angle_deg"]))
    seg = segments_cm(frames) if scaled and reuse_video is None else {}
    for dst,src in (('thigh_cm','thigh'),('shin_cm','shin'),('torso_cm','torso'),('upper_cm','upper_arm'),('fore_cm','forearm')):
        geo[dst] = reuse_video["geo"].get(dst) if reuse_video is not None else seg.get(src)
    basis = "input_height" if height_cm is not None and not is_reference else "prior_170"
    assumed_height = height_cm if basis == "input_height" else 170.
    implied = implied_plate_diameter(raw,boxes,(w,h),stand,assumed_height,reliable)
    bounds = [implied_plate_diameter(raw,boxes,(w,h),stand,x,reliable) for x in (160.,180.)]
    checks = dict(stature_plate_cm=None if measured_height is None else measured_height*100,
                  warn=bool(height_cm is not None and not is_reference and measured_height is not None and abs(measured_height*100-height_cm)/height_cm>.15),
                  plate_mm_implied=implied,plate_mm_range=bounds,plate_mm_basis=basis,
                  plate_suggest=plate_suggestion(implied,diameter))
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
    good_boxes = boxes[np.isfinite(boxes).all(axis=1)]
    x0,x1 = min(x0,good_boxes[:,0].min()),max(x1,good_boxes[:,2].max())
    y0,y1 = min(y0,good_boxes[:,1].min()),max(y1,good_boxes[:,3].max())
    padx,pady = .1*(x1-x0),.06*(y1-y0)
    crop = [int(np.floor(x0-padx)),int(np.floor(y0-pady)),int(np.ceil(x1+padx)),int(np.ceil(y1+pady))]
    names = list(MP_FROM_H36M)
    if reuse_video is None:
        mp = to_mp33(frames)
        j3 = [q[names].tolist() if np.isfinite(q[names]).all() else None for q in mp]
    else:
        names = reuse_video["j3names"]; j3 = reuse_video["j3"]
    data = dict(fps=fps,width=w,height=h,duration=len(t)/fps,t=t.tolist(),plate=plate,
                depth=np.round(depth,1).tolist(),reps=reps,plate_used=100*sum(p is not None for p in plate)/len(plate),
                crop=crop,geo=geo,geo_check=checks,j2=J2,kp2=kp2,j3names=names,j3=j3,clips=[],
                ellipse=[r.get('plate',{}).get('ellipse') for r in rows])
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


def compare(user,ref,exercise,out,plate_mm=450.,load_kg=None,height_cm=None,threads=6,model=None,checkpoint=None,
            progress=None,ref_cache=None,posture_cards=False,scale_source="box",position_source="box",recover_early_back=None):
    """The M3a calculations, with observation callbacks and optional reference reuse.

    progress(stage, fraction, **details) reports work actually completed. Cached
    observations bypass inference, while height-dependent display transforms run.
    """
    import cv2
    import onnxruntime as ort
    from feedback.pose import extract_pose
    from feedback.plate import plate_rep_mask
    from core.io.rows import read_rows
    from core.io import video
    from server.jobs import process_video
    from server.cache import cache_key,available,publish
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    for name,value in (('plate_mm',plate_mm),('load_kg',load_kg),('height_cm',height_cm)):
        if value is not None and (not np.isfinite(value) or value<=0): raise ValueError(name+' must be positive and finite')
    if scale_source not in ('box','ellipse') or position_source not in ('box','ellipse'): raise ValueError('invalid source')
    recover_early_back = DEFAULT.speed.recover_early_back if recover_early_back is None else bool(recover_early_back)
    cfg = replace(DEFAULT, speed=replace(DEFAULT.speed,recover_early_back=recover_early_back))
    # Legacy caches do not contain rim fits; never reuse them for an ellipse run.
    if scale_source == 'ellipse' or position_source == 'ellipse': ref_cache = None
    started=time.perf_counter()
    model=Path(model or os.environ.get('PLATE_MODEL',ROOT/'plate_pipeline/models/plate_tiny.onnx'))
    checkpoint=Path(checkpoint or os.environ.get('POSE3D_CHECKPOINT',Path.home()/'.cache/videopose3d/pretrained_h36m_detectron_coco.bin'))
    if not model.is_file(): raise FileNotFoundError(f'원판 모델이 없습니다: {model} (PLATE_MODEL로 지정)')
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    stages={};active=None;stage_start=0
    def emit(stage,fraction,**details):
        nonlocal active,stage_start
        now=time.perf_counter()
        if active!=stage:
            if active is not None: stages[active]=now-stage_start
            active,stage_start=stage,now
        if progress is not None: progress(stage,float(fraction),**details)
        if fraction==1: stages[stage]=time.perf_counter()-stage_start
    emit('prepare',0)
    cached=None;provenance={};hit=False
    if ref is not None and ref_cache is not None:
        key,provenance=cache_key(ref,exercise,plate_mm,model,checkpoint,recover_early_back)
        cached=Path(ref_cache)/key;hit=available(cached)
    settings=prepare(user,None if hit else ref,out)
    if hit:
        shutil.copy2(cached/'ref.mp4',out/'ref.mp4');settings['ref']=dict(video=str(out/'ref.mp4'))
    roles=list(settings)
    emit('prepare',1,ref_cache_hit=hit)
    cv2.setNumThreads(1)
    original=ort.InferenceSession
    def session(*args,**kwargs):
        opts=kwargs.pop('sess_options',None) or ort.SessionOptions()
        opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
        return original(*args,sess_options=opts,**kwargs)
    b=dict(schema_version=SCHEMA_VERSION,exercise=exercise,inputs=dict(plate_mm=plate_mm,load_kg=load_kg,height_cm=height_cm),
           show_3d=exercise!='benchpress',pose3d_scale='input_height' if height_cm is not None else 'model_native',
           posture_cards=bool(posture_cards),has_ref=ref is not None,ref=None,
           scale_source=scale_source,position_source=position_source,recover_early_back=recover_early_back)
    observations={};timing={};stats={}
    with patch.object(ort,'InferenceSession',session):
        for role in roles:
            stage='plate_'+role;emit(stage,0)
            dest=out/'analysis'/role;dest.mkdir(parents=True,exist_ok=True)
            total=video.info(settings[role]['video']).frames
            if total<=0: total=sum(1 for _ in video.frames(settings[role]['video']))
            if role=='ref' and hit:
                shutil.copytree(cached/'plate',dest/'plate',dirs_exist_ok=True)
                result=json.loads((dest/'plate/result.json').read_text());rows=read_rows(dest/'plate/rows.jsonl.gz')
            else:
                def frame(seq,t,img,st,stage=stage,total=total):
                    emit(stage,min((seq+1)/max(1,total),1),processed=seq+1,total=total)
                result,rows,_=process_video(settings[role]['video'],'bench' if exercise=='benchpress' else exercise,
                    plate_mm=plate_mm,model=str(model),device='cpu',pose=False,out_dir=dest/'plate',on_frame=frame,
                    scale_source=scale_source,position_source=position_source,cfg=cfg)
            if not any(r.get('plate',{}).get('ellipse') for r in rows):
                from core.recog.ellipse import annotate_video
                annotate_video(settings[role]['video'],rows)
            observations[role]=dict(result=result,rows=rows)
            emit(stage,1,processed=len(rows),total=len(rows),cached=role=='ref' and hit)
        emit('pose',0)
        total=sum(len(observations[r]['rows']) for r in roles);done=0
        for role in roles:
            o=observations[role];dest=out/'analysis'/role
            if role=='ref' and hit:
                shutil.copy2(cached/'pose.npz',dest/'pose.npz')
                with np.load(dest/'pose.npz',allow_pickle=False) as f: pose={k:f[k] for k in f.files}
                for k in ('fps','w','h','total'): pose[k]=pose[k].item()
            else:
                def frame(n,_,offset=done):emit('pose',(offset+n)/total,processed=offset+n,total=total)
                pose=extract_pose(settings[role]['video'],dest/'pose.npz',exercise,no_cache=True,plate_rows=o['rows'],
                    rep_mask=plate_rep_mask(o['rows'],o['result']),pose_cfg=DEFAULT.pose,on_frame=frame)
            o['pose']=pose;done+=len(o['rows']);emit('pose',done/total,processed=done,total=total,ref_cached=hit)
    emit('3d',0)
    # Loading the exact same model and doing the same postprocessing as M3a.
    vp3d=load_model(checkpoint,threads);ref_height=height_cm
    for index,role in enumerate(roles):
        start=time.perf_counter();o=observations[role];dest=out/'analysis'/role;pose=o['pose']
        if role=='ref' and hit: native=np.load(cached/'native.npy',allow_pickle=False)
        else: native,_,_=lift(vp3d,selected_coco(pose),pose['w'],pose['h'])
        np.save(dest/'native.npy',native)
        raw,stand,frames,actual,scaled,warnings=aligned_3d(pose,o['rows'],o['result'],vp3d,ref_height,native=native)
        b[role],trips,vals=build_video(o['rows'],o['result'],pose,frames,raw,stand,plate_mm,height_cm,scaled and height_cm is not None,scale_source,position_source,is_reference=(role=='ref'),speed_cfg=cfg.speed)
        b[role]['pose3d_warnings']=warnings;b[role]['pose3d_height_cm']=actual
        if role=='user': ref_height=height_cm if height_cm is not None else actual
        stats[role]=(trips,vals,summarize(o['result']['exercise'],o['result']['fps'],trips,vals))
        timing[role]=time.perf_counter()-start
        if role=='ref' and cached is not None and not hit:
            # RTMPose may have populated its model cache during the first run.
            key,provenance=cache_key(ref,exercise,plate_mm,model,checkpoint,recover_early_back)
            publish(Path(ref_cache)/key,dest,out/'ref.mp4',provenance)
        emit('3d',(index+1)/len(roles),processed=index+1,total=len(roles),ref_cached=hit)
    emit('judge',0)
    finalize_bundle(b,stats)
    b['ref_cache']=dict(hit=hit,**provenance)
    emit('judge',1)
    emit('report',0)
    b['timing']=dict(**timing,total_s=time.perf_counter()-started,threads=threads,stages=stages)
    b=clean(b)
    from server.schema import validate_result
    validate_result(b)
    (out/'result_v3.json').write_text(json.dumps(b,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    write_report(b,out);emit('report',1)
    # Persist final stage durations without fabricating a report duration.
    b['timing']['stages']=dict(stages);b['timing']['total_s']=time.perf_counter()-started
    (out/'result_v3.json').write_text(json.dumps(b,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    write_report(b,out)
    return b


def finalize_bundle(b, stats):
    exercise=b['exercise']; posture_cards=b.get('posture_cards',False); ref=b.get('ref')
    roles=list(stats)
    ex='bench' if exercise=='benchpress' else exercise
    b['sentences']=[dict(kind=s.kind,text=s.text,key=s.key,shown=s.shown) for s in render(run_compare(stats['user'][2],stats['ref'][2] if ref is not None else None),ex)]
    b['set_median']={role:{key:round(median,5) for key,attr in (('down','down'),('pause','far_pause'),('up','up'))
        if stats[role][1] and np.isfinite(median:=float(np.median([getattr(v,attr) or 0 for v in stats[role][1]])))} for role in roles}
    vs=[r['v'] for r in b['user']['reps']];valid_speeds=[v for v in vs if v is not None]
    b['v_loss']=round(100*(max(valid_speeds)-vs[-1])/max(valid_speeds),1) if valid_speeds and max(valid_speeds)>0 and vs[-1] is not None else None
    b['summary']=summary_sentence(ex,len(vs),vs,b['v_loss']).text
    b['findings'],b['angle_judgments']=calculate(b,posture_cards=posture_cards)
    b['kpi']=b['angle_judgments']['FAR']['knee']
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--user',required=True,type=Path);ap.add_argument('--ref',type=Path)
    ap.add_argument('--exercise',required=True,choices=('squat','benchpress','deadlift'))
    ap.add_argument('--plate-mm',type=float,default=450.);ap.add_argument('--load-kg',type=float);ap.add_argument('--height-cm',type=float)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--posture-cards',action='store_true')
    ap.add_argument('--scale-source',choices=('box','ellipse'),default='box')
    ap.add_argument('--position-source',choices=('box','ellipse'),default='box')
    ap.add_argument('--recover-early-back',action=argparse.BooleanOptionalAction,default=None)
    a = ap.parse_args()
    b = compare(**vars(a))
    print(json.dumps(dict(reps={r:len(b[r]['reps']) for r in ('user','ref') if b[r] is not None},timing=b['timing']),ensure_ascii=False))

if __name__=='__main__': main()
