"""Plate-plane scale and explicit assumed bar-depth correction, independent of judging."""
import numpy as np
from core.config import DEFAULT


def plate_geometry(boxes, image_size, diameter_mm=450., cfg=None):
    cfg = cfg or DEFAULT.geometry
    if diameter_mm <= 0:
        raise ValueError('Plate diameter must be positive')
    boxes = np.asarray(boxes, float)
    widths, heights = boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]
    good = np.isfinite(boxes).all(axis=1) & (widths > 0) & (heights > 0)
    if not good.any():
        raise ValueError('No positive finite plate boxes')
    ratio = float(np.median(widths[good]/heights[good]))
    angle = float(np.arccos(np.clip(ratio, 0, 1)))
    focal = cfg.focal_long_side * max(image_size)
    h = np.interp(np.arange(len(boxes)), np.flatnonzero(good), heights[good])
    distance = focal * (diameter_mm/1000) / h
    kd = (distance + cfg.bar_depth_m*np.cos(angle)) / distance
    return dict(angle_deg=float(np.degrees(angle)), ratio=ratio, focal_px=focal,
                distance_m=distance, kd=kd, plate_m_per_px=(diameter_mm/1000)/h,
                lifter_m_per_px=(diameter_mm/1000)/h*kd, observed=good)


def angle(a,b,c):
    u,v=np.asarray(a)-b,np.asarray(c)-b
    norms=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.degrees(np.arccos(np.clip(np.sum(u*v,axis=-1)/norms,-1,1)))


def standing_frames(raw):
    """Conservative 2D standing evidence, never a standing spotter re-selection."""
    p=np.asarray(raw)
    ids=[0,11,12,23,24,25,26,27,28]
    supported=np.isfinite(p[:,ids,:]).all(axis=(1,2)) & (p[:,ids,3]>=.5).all(axis=1)
    shoulders=p[:,[11,12],:2].mean(axis=1);hips=p[:,[23,24],:2].mean(axis=1)
    trunk=shoulders-hips
    upright=(trunk[:,1]<0)&(np.abs(trunk[:,0])<np.abs(trunk[:,1])*np.tan(np.radians(30)))
    knees=np.stack([angle(p[:,h,:2],p[:,k,:2],p[:,a,:2]) for h,k,a in ((23,25,27),(24,26,28))],axis=1)
    return supported & upright & (knees>160).all(axis=1)


def stature(raw, geometry, stand, cfg=None):
    cfg=cfg or DEFAULT.geometry
    feet=np.asarray(raw)[:,[27,28,29,30,31,32]]
    visible=(feet[:,:,3]>=.5)&np.isfinite(feet[:,:,:2]).all(axis=2)
    floor=np.max(np.where(visible,feet[:,:,1],-np.inf),axis=1)
    nose=raw[:,0,1]
    height=(floor-nose)*geometry['lifter_m_per_px']/cfg.nose_stature_ratio
    use=stand & np.isfinite(height) & (height>0)
    return (float(np.median(height[use])) if use.any() else None),use
