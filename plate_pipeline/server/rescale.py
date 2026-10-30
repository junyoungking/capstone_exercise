"""Recompute a saved session from immutable observations, without model inference."""
import json,time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
from core.io.rows import read_rows
from core.geometry import standing_frames
from core.config import DEFAULT
from core.compare.summary import summarize
from server.compare import build_video,finalize_bundle,clean
from server.schema import validate_result


def rescale_bundle(session_dir, previous, plate_mm_user=None, plate_mm_ref=None):
    start=time.perf_counter();root=Path(session_dir);b=deepcopy(previous)
    roles=['user']+(['ref'] if b.get('ref') is not None else [])
    if plate_mm_ref is not None and 'ref' not in roles:raise ValueError('기준 영상이 없습니다.')
    inputs=b['inputs'];old_user=inputs['plate_mm'];old_ref=inputs.get('plate_mm_ref',old_user)
    diameters={'user':old_user if plate_mm_user is None else plate_mm_user,'ref':old_ref if plate_mm_ref is None else plate_mm_ref}
    inputs['plate_mm']=diameters['user']
    if 'ref' in roles:inputs['plate_mm_ref']=diameters['ref']
    stats={}
    for role in roles:
        old=b[role];dest=root/'analysis'/role
        rows=read_rows(dest/'plate/rows.jsonl.gz'); result=json.loads((dest/'plate/result.json').read_text())
        # Existing box runs kept ellipse diagnostics in result_v3, not raw rows.
        for row,ellipse in zip(rows,old.get('ellipse',[])):
            if ellipse and row.get('plate',{}).get('used'):row['plate']['ellipse']=ellipse
        with np.load(dest/'pose.npz',allow_pickle=False) as f:pose={k:f[k] for k in f.files}
        for k in ('fps','w','h','total'):pose[k]=pose[k].item()
        raw=np.asarray(pose['raw_lms'],float).copy();raw[:,:,:2]*=[pose['w'],pose['h']]
        stand=standing_frames(raw)
        video,trips,vals=build_video(rows,result,pose,None,raw,stand,diameters[role],inputs.get('height_cm'),False,
            b.get('scale_source','box'),b.get('position_source','box'),is_reference=role=='ref',reuse_video=old,
            speed_cfg=replace(DEFAULT.speed,recover_early_back=b.get('recover_early_back',DEFAULT.speed.recover_early_back)))
        for key in ('pose3d_warnings','pose3d_height_cm'):
            if key in old:video[key]=old[key]
        b[role]=video
        stats[role]=(trips,vals,summarize(result['exercise'],result['fps'],trips,vals))
    b['allow_empty_reps']=True
    finalize_bundle(b,stats)
    b.setdefault('timing',{})['rescale_s']=time.perf_counter()-start
    b=clean(b);validate_result(b)
    return b
