"""Optional M2 export. These 3D diagnostics never enter feedback judgments."""
import time
from dataclasses import asdict
import numpy as np
from feedback import plate  # Establish the pipeline core package path.
from core.config import DEFAULT
from core.geometry import plate_geometry,standing_frames,stature
from core.geometry.pose3d import (selected_coco,lift,smooth,fixed_bones,body_basis,
    scale_floor,to_mp33,angles_at_positions,acceleration,segments_cm,MP_FROM_H36M,PARENTS)
from track3d.run_pose3d import bone_length_cv,frames_for_json

def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,np.floating,np.bool_)):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def set_groups(rows,result):
    from feedback.plate import nearest_seq
    groups=np.zeros(len(rows),int)
    bounds=[]
    for st in result['sets']:
        values=[v for v in st['rep_values'] if v['start'] is not None and v['back'] is not None]
        if values:bounds.append((nearest_seq(rows,min(v['start'] for v in values)),nearest_seq(rows,max(v['back'] for v in values))))
    if bounds:
        centres=np.asarray([(a+b)/2 for a,b in bounds])
        groups=np.argmin(abs(np.arange(len(rows))[:,None]-centres),axis=1)
        for i,(a,b) in enumerate(bounds):groups[a:b+1]=i
    return groups,bounds


def process(pose,rows,result,segments,model,diameter_mm=450.):
    started=time.perf_counter()
    raw=np.asarray(pose['raw_lms'],float).copy();raw[:,:,:2]*=[pose['w'],pose['h']]
    boxes=np.asarray([r['plate']['box'] if r.get('plate',{}).get('used') else [np.nan]*4 for r in rows])
    geo=plate_geometry(boxes,(pose['w'],pose['h']),diameter_mm)
    stand=standing_frames(raw);height,stand_used=stature(raw,geo,stand)
    t=time.perf_counter();native,interpolated,nulls=lift(model,selected_coco(pose),pose['w'],pose['h']);inference=time.perf_counter()-t
    filtered=smooth(native,pose['fps']);groups,bounds=set_groups(rows,result)
    fixed,targets=fixed_bones(filtered,groups)
    basis,reasons=body_basis(fixed,stand_used,raw)
    oriented=np.einsum('ij,tkj->tki',basis,fixed)
    final,scale,scale_reason=scale_floor(oriented,stand_used,height,DEFAULT.geometry.nose_stature_ratio)
    if scale_reason:reasons.append(scale_reason)
    if not stand_used.any():reasons.append('No supported upright straight-knee standing frame for plate stature')
    angle_metrics=angles_at_positions(final,segments)
    # Same physical scale on pre/post sequences; floor shifts excluded from this comparison.
    acc_before=acceleration(native*scale,pose['fps']);acc_after=acceleration(filtered*scale,pose['fps'])
    geometry=dict(diameter_mm=diameter_mm,angle_deg=geo['angle_deg'],ratio=geo['ratio'],focal_px=geo['focal_px'],
                  distance_m=float(np.median(geo['distance_m'])),kd=float(np.median(geo['kd'])),
                  stature_cm=None if height is None else height*100,segments_cm=segments_cm(final),
                  standing_frames=int(stand_used.sum()),standing_indices=np.flatnonzero(stand_used),
                  observed_plate_frames=int(geo['observed'].sum()),
                  scale=scale,scaled_to_stature=scale_reason is None,assumptions=asdict(DEFAULT.geometry))
    elapsed=time.perf_counter()-started
    mp=to_mp33(final);j3names=list(MP_FROM_H36M)
    payload=dict(schema_version=1,fps=pose['fps'],total=pose['total'],units='metres',coordinate_system='x=lateral, y=up, z=forward',
                 joints='h36m17',parents=PARENTS,frames=frames_for_json(final),
                 mp33_mapping=MP_FROM_H36M,nose_mapping='H36M 9 neck/nose proxy (no independent COCO nose)',
                 j3names=j3names,j3=[[frame[j] for j in j3names] if np.isfinite(frame[:,0]).any() else None for frame in mp],
                 geometry=geometry,angles3d=angle_metrics,
                 bone_cv_before=bone_length_cv(filtered),bone_cv_after=bone_length_cv(fixed),
                 per_set_bone_cv={int(g):dict(before=bone_length_cv(filtered[groups==g]),after=bone_length_cv(fixed[groups==g])) for g in np.unique(groups)},
                 per_set_bone_targets=targets,set_frame_bounds=bounds,
                 acceleration_before=acc_before,acceleration_after=acc_after,
                 interpolation=dict(filled=np.flatnonzero(interpolated),null=np.flatnonzero(nulls),max_gap_frames=10),
                 basis=basis,warnings=reasons,config=asdict(DEFAULT.pose3d),
                 timing=dict(inference_s=inference,geometry_postprocess_s=elapsed-inference,processing_s=elapsed))
    return clean(payload),native,final
