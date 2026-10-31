"""Plate-plane scale and explicit assumed bar-depth correction, independent of judging."""
import numpy as np
from core.config import DEFAULT


def retake_warning(angle_deg):
    return bool(angle_deg > 60.)


def plate_geometry(boxes, image_size, diameter_mm=450., cfg=None, ellipses=None):
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
    ellipse_valid = [(i,e) for i,e in enumerate(ellipses or []) if e and e.get("source") == "ellipse"]
    source = "ellipse" if ellipse_valid else "box"
    if ellipse_valid:
        ratios = [e.get("smooth_minor",e["minor"])/e.get("smooth_major",e["major"]) for _,e in ellipse_valid]
        ratio = float(np.median(ratios))
        angle = float(np.arccos(np.clip(ratio,0,1)))
        for i,e in ellipse_valid: h[i] = e.get("smooth_major",e["major"])
    distance = focal * (diameter_mm/1000) / h
    kd = (distance + cfg.bar_depth_m*np.cos(angle)) / distance
    return dict(angle_deg=float(np.degrees(angle)), ratio=ratio, focal_px=focal,
                distance_m=distance, kd=kd, plate_m_per_px=(diameter_mm/1000)/h,
                lifter_m_per_px=(diameter_mm/1000)/h*kd, observed=good, angle_source=source,
                ellipse_count=len(ellipse_valid), ellipse_used_fraction=len(ellipse_valid)/int(good.sum()),
                ellipse_major_px=float(np.median([e.get("smooth_major",e["major"]) for _,e in ellipse_valid])) if ellipse_valid else None,
                ellipse_minor_px=float(np.median([e.get("smooth_minor",e["minor"]) for _,e in ellipse_valid])) if ellipse_valid else None,
                ellipse_tilt_deg=float(np.degrees(.5*np.arctan2(np.median([np.sin(2*e.get("smooth_angle",e["angle"])) for _,e in ellipse_valid]),np.median([np.cos(2*e.get("smooth_angle",e["angle"])) for _,e in ellipse_valid])))) if ellipse_valid else None)


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


def box_touches_edge(box, image_size):
    """The shared one-pixel frame-edge rule for a display-coordinate box."""
    x0,y0,x1,y1=map(float,box);w,h=image_size
    return bool(x0<=1 or y0<=1 or x1>=w-1 or y1>=h-1)


def box_angle_reliable(boxes, image_size):
    """Box width can include the sleeve; a clipped box is not an angle cue."""
    boxes = np.asarray(boxes, float)
    good = np.isfinite(boxes).all(axis=1) & (boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])
    if not good.any(): return False
    b=boxes[good]
    ratio=float(np.median((b[:,2]-b[:,0])/(b[:,3]-b[:,1])))
    touches=any(box_touches_edge(box,image_size) for box in b)
    return bool(ratio < .95 and not touches)


def implied_plate_diameter(raw, boxes, image_size, standing, height_cm=170., reliable=None, cfg=None):
    """Invert M2's nose/floor height equation per standing frame (mm).

    H*0.92/pixel_height = D/pixel_plate_height + bar_depth*cos(angle)/focal.
    Unknown box angles use cos=1; no supported positive solution returns None.
    """
    cfg=cfg or DEFAULT.geometry
    if not np.isfinite(height_cm) or height_cm<=0: raise ValueError('height must be positive')
    p=np.asarray(raw,float);b=np.asarray(boxes,float);stand=np.asarray(standing,bool)
    if len(p)!=len(b) or len(p)!=len(stand): raise ValueError('frame count mismatch')
    good=np.isfinite(b).all(axis=1)&(b[:,2]>b[:,0])&(b[:,3]>b[:,1])
    if not good.any(): return None
    heights=np.interp(np.arange(len(b)),np.flatnonzero(good),(b[:,3]-b[:,1])[good])
    feet=p[:,[27,28,29,30,31,32]]
    visible=(feet[:,:,3]>=.5)&np.isfinite(feet[:,:,:2]).all(axis=2)
    floor=np.max(np.where(visible,feet[:,:,1],-np.inf),axis=1)
    pixel_height=floor-p[:,0,1]
    if reliable is None: reliable=box_angle_reliable(b,image_size)
    ratio=float(np.median((b[good,2]-b[good,0])/(b[good,3]-b[good,1])))
    cosine=float(np.clip(ratio,0,1)) if reliable else 1.
    focal=cfg.focal_long_side*max(image_size)
    with np.errstate(divide='ignore',invalid='ignore'):
        diam=1000*heights*(height_cm/100*cfg.nose_stature_ratio/pixel_height-cfg.bar_depth_m*cosine/focal)
    # An interpolated plate size outside its observations is not evidence for
    # the same standing person. Require a tracked plate in the standing frame.
    width,height=image_size
    complete=(b[:,0]>1)&(b[:,1]>1)&(b[:,2]<width-1)&(b[:,3]<height-1)
    use=stand & good & complete & np.isfinite(diam)&(diam>0)&np.isfinite(pixel_height)&(pixel_height>0)
    return float(np.median(diam[use])) if use.any() else None


PLATE_SUGGEST_FRACTION = .30


def plate_suggestion(implied_mm, used_mm, threshold=PLATE_SUGGEST_FRACTION):
    return bool(implied_mm is not None and np.isfinite(implied_mm) and used_mm>0 and abs(implied_mm-used_mm)/used_mm>threshold)
