"""Lift the selected MP33/COCO stream, without a second person detector."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
from core.config import DEFAULT
from core.geometry import angle

PARENTS=(-1,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15)
COCO_FROM_MP=(0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28)
# H36M has no independent COCO nose: joint 9 (neck/nose) is an explicit proxy.
MP_FROM_H36M={0:9,11:11,12:14,13:12,14:15,15:13,16:16,23:4,24:1,25:5,26:2,27:6,28:3}


def selected_coco(pose):
    raw=np.asarray(pose['raw_lms'],float)
    out=raw[:,COCO_FROM_MP,:2]*[pose['w'],pose['h']]
    # As in track3d, a frame is missing as a whole. No extra confidence pruning.
    out[~np.asarray(pose['valid'],bool)]=np.nan
    return out.astype(np.float32)


def load_model(checkpoint, threads=1):
    import torch
    from track3d.run_pose3d import load_videopose_model
    from feedback import config
    checkpoint=Path(checkpoint)
    if checkpoint.name!=config.POSE3D_CHECKPOINT_NAME:
        raise ValueError('Expected pretrained_h36m_detectron_coco.bin')
    model,path=load_videopose_model(checkpoint.parent)
    torch.set_num_threads(threads)
    assert model.receptive_field()==243
    return model


def lift(model,points,width,height,cfg=None):
    import torch
    from track3d.run_pose3d import interpolate_short_gaps,valid_runs_after_interpolation,normalize_keypoints
    cfg=cfg or DEFAULT.pose3d
    prepared,interpolated,nulls=interpolate_short_gaps(points,cfg.max_gap_frames)
    output=np.full((len(points),17,3),np.nan,np.float32)
    pad=(model.receptive_field()-1)//2
    with torch.inference_mode():
        for run in valid_runs_after_interpolation(prepared):
            norm=normalize_keypoints(prepared[run],width,height)
            padded=np.pad(norm,((pad,pad),(0,0),(0,0)),mode='edge')
            for start in range(0,len(run),cfg.chunk_frames):
                n=min(cfg.chunk_frames,len(run)-start)
                inputs=torch.from_numpy(padded[None,start:start+n+2*pad].copy())
                output[run[start:start+n]]=model(inputs).cpu().numpy()[0]
    return output,interpolated,nulls


def smooth(frames,fps,cfg=None):
    from scipy.ndimage import gaussian_filter1d
    from track3d.run_pose3d import contiguous_runs
    cfg=cfg or DEFAULT.pose3d
    out=np.asarray(frames,float).copy()
    sigma=max(1.,fps*cfg.smooth_sigma_s)
    # No smoothing/interpolation across long missing runs.
    for run in contiguous_runs(np.flatnonzero(np.isfinite(out).all(axis=(1,2)))):
        out[run]=gaussian_filter1d(out[run],sigma,axis=0,mode='nearest',radius=int(np.ceil(3*sigma)))
    return out


def fixed_bones(frames,groups=None):
    """Fix all 16 tree edges per set; preserve the original edge directions."""
    p=np.asarray(frames,float);out=p.copy()
    groups=np.zeros(len(p),int) if groups is None else np.asarray(groups)
    targets={}
    for group in np.unique(groups):
        mask=groups==group
        lengths={j:np.linalg.norm(p[mask,j]-p[mask,a],axis=1) for j,a in enumerate(PARENTS) if a>=0}
        med={j:float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else float('nan') for j,v in lengths.items()}
        targets[int(group)]=med
        for j,a in enumerate(PARENTS):
            if a<0:continue
            d=p[mask,j]-p[mask,a];n=np.linalg.norm(d,axis=1,keepdims=True)
            with np.errstate(invalid='ignore',divide='ignore'):
                direction=d/n
            # A zero-length bone has no direction; leave the entire frame unavailable.
            out[mask,j]=out[mask,a]+direction*med[j]
    invalid=~np.isfinite(out).all(axis=(1,2));out[invalid]=np.nan
    return out,targets


def body_basis(frames,stand,raw_pixels):
    valid=np.isfinite(frames).all(axis=(1,2));use=valid&stand
    reasons=[]
    if use.any():
        up=np.median(frames[use,9]-(frames[use,3]+frames[use,6])/2,axis=0)
    else:
        up=np.array([0.,-1.,0.]);reasons.append('No standing 3D frame: camera-up fallback')
    if not valid.any():return np.eye(3),['No finite 3D frames']
    norm=np.linalg.norm(up)
    if norm<1e-8:up=np.array([0.,-1.,0.]);reasons.append('Degenerate standing up vector')
    else:up=up/norm
    heading=use if use.any() else valid
    lat=np.median(frames[heading,1]-frames[heading,4],axis=0)
    lat-=up*np.dot(lat,up)
    if np.linalg.norm(lat)<1e-8:
        lat=np.array([1.,0,0]);lat-=up*np.dot(lat,up);reasons.append('Degenerate hip axis')
    lat/=np.linalg.norm(lat);fwd=np.cross(lat,up)
    # Halpe heel -> big toe vector, projected into the original image plane.
    toes=(raw_pixels[:,[31,32],:2]-raw_pixels[:,[29,30],:2]).mean(axis=1)
    supported=(raw_pixels[:,[29,30,31,32],3]>=.5).all(axis=1)&heading&np.isfinite(toes).all(axis=1)
    sign=float(np.median(toes[supported]@fwd[:2])) if supported.any() else 0.
    if sign<0:lat=-lat;fwd=-fwd
    if abs(sign)<1e-3:reasons.append('Toe heading sign unobservable/ambiguous')
    return np.stack([lat,up,fwd]),reasons


def scale_floor(frames,stand,stature_m,nose_ratio=.92):
    q=np.asarray(frames,float).copy()
    use=stand&np.isfinite(q).all(axis=(1,2))
    nose_height=q[:,9,1]-q[:,[3,6],1].min(axis=1)
    use &= nose_height>1e-6
    scale=1.
    reason=None
    if stature_m is not None and use.any():
        scale=stature_m*nose_ratio/float(np.median(nose_height[use]))
    else:reason='Stature/standing unavailable: native VideoPose3D metre scale retained'
    q*=scale
    q[:,:,1]-=q[:,[3,6],1].min(axis=1)[:,None]
    valid=np.isfinite(q).all(axis=(1,2))
    if valid.any():
        anchor=np.median(q[use if use.any() else valid][:,[3,6]].mean(axis=1),axis=0)
        q[:,:,[0,2]]-=anchor[[0,2]]
    return q,scale,reason


def to_mp33(frames):
    out=np.full((len(frames),33,3),np.nan)
    for mp,h in MP_FROM_H36M.items():out[:,mp]=frames[:,h]
    return out


def angles_at_positions(frames,segments):
    p=frames;reps=[]
    for rep in segments['reps']:
        positions={}
        side=rep['side'];h,k,a,s=(4,5,6,11) if side=='left' else (1,2,3,14)
        for name,i in rep['positions'].items():
            vals={key:None for key in ('knee_angle','hip_angle','trunk_lean')}
            if np.isfinite(p[i]).all():
                vec=(p[i,11]+p[i,14])/2-(p[i,1]+p[i,4])/2
                vals=dict(knee_angle=float(angle(p[i,h],p[i,k],p[i,a])),
                          hip_angle=float(angle(p[i,s],p[i,h],p[i,k])),
                          trunk_lean=float(np.degrees(np.arctan2(np.linalg.norm(vec[[0,2]]),vec[1]))))
            positions[name]=dict(frame=i,values=vals)
        reps.append(dict(rep=rep['rep'],positions=positions))
    return dict(reps=reps)


def acceleration(frames,fps):
    # Unit: m/s^2, no cross-gap differences and no floor-lock distortion here.
    accel=np.linalg.norm(np.diff(frames,n=2,axis=0)*fps**2,axis=2)
    good=accel[np.isfinite(accel)]
    return float(np.median(good)) if len(good) else None


def segments_cm(frames):
    pairs=dict(thigh=((1,2),(4,5)),shin=((2,3),(5,6)),
               torso=((1,14),(4,11)),upper_arm=((11,12),(14,15)),forearm=((12,13),(15,16)))
    result={}
    for name,edges in pairs.items():
        values=np.stack([np.linalg.norm(frames[:,a]-frames[:,b],axis=1) for a,b in edges],axis=1)
        good=values[np.isfinite(values)]
        result[name]=float(np.median(good)*100) if good.size else None
    return result
