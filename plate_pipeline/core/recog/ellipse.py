"""Conservative plate-rim ellipse fitting and temporal filtering.

Ellipses are represented by centre, full major/minor diameters, and the major
axis angle in radians in image coordinates. A rejected fit is never silently
promoted to an observation.
"""
import cv2
import numpy as np


def fit_plate_ellipse(image, box):
    x0, y0, x1, y1 = map(float, box)
    w, h = x1-x0, y1-y0
    if min(w, h) <= 3:
        return None
    cx, cy = (x0+x1)/2, (y0+y1)/2
    size = max(w, h)
    H, W = image.shape[:2]
    # A truncated detection does not establish the unseen part of the rim.
    if x0 <= 1 or y0 <= 1 or x1 >= W-1 or y1 >= H-1:
        return None
    pad = .30*size
    xa, ya = max(0, int(x0-pad)), max(0, int(y0-pad))
    xb, yb = min(W, int(np.ceil(x1+pad))), min(H, int(np.ceil(y1+pad)))
    roi = image[ya:yb, xa:xb]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    # Scaling fixes edge and RANSAC tolerances across video resolutions.
    k = min(1., 320/max(roi.shape[:2]))
    if k < 1:
        gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edge = cv2.Canny(gray, 35, 110)
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    points = np.column_stack(np.nonzero(edge))[:, ::-1].astype(float)
    if len(points) < 20:
        return None
    # Contour proposals plus deterministic RANSAC subsets. Exclude the image
    # boundary itself, which otherwise pretends to be a straight rim segment.
    keep = (points[:,0] > 1) & (points[:,0] < edge.shape[1]-2) & (points[:,1] > 1) & (points[:,1] < edge.shape[0]-2)
    points = points[keep]
    distance = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    sample_theta = np.linspace(0, 2*np.pi, 96, endpoint=False)
    rng = np.random.default_rng(0)
    proposals = [c.reshape(-1,2).astype(np.float32) for c in contours if len(c) >= 20]
    if len(points) >= 10:
        proposals += [points[rng.choice(len(points), 8, replace=False)].astype(np.float32) for _ in range(100)]
    best = None
    for p in proposals:
        try:
            (ex,ey),(a,b),deg = cv2.fitEllipse(p)
        except cv2.error:
            continue
        if a < b:
            a,b = b,a; deg += 90
        angle = np.deg2rad(deg % 180)
        X,Y = xa + ex/k, ya + ey/k
        major,minor = a/k,b/k
        if not np.isfinite([X,Y,major,minor,angle]).all(): continue
        if np.hypot((X-cx)/max(w,1), (Y-cy)/max(h,1)) > .35: continue
        if not .7*size <= major <= 1.4*size or minor/major < .2: continue
        # Fits extending off-screen cannot establish an outer diameter reliably.
        ca,sa = np.cos(angle),np.sin(angle)
        rx = np.hypot(major*ca,minor*sa)/2
        ry = np.hypot(major*sa,minor*ca)/2
        if X-rx < 1 or Y-ry < 1 or X+rx > W-2 or Y+ry > H-2: continue
        # Measure support at evenly spaced points ON the proposed rim. Counting
        # arbitrary edge pixels in a thick normalized annulus admits dense text,
        # sleeve edges and short straight occluders as false ellipses.
        eu = a/2*np.cos(sample_theta); ev = b/2*np.sin(sample_theta)
        xx = np.rint(ex+ca*eu-sa*ev).astype(int); yy = np.rint(ey+sa*eu+ca*ev).astype(int)
        inside = (xx>=1)&(yy>=1)&(xx<edge.shape[1]-1)&(yy<edge.shape[0]-1)
        if inside.mean() < .98: continue
        dist = np.full(len(xx),99.); dist[inside] = distance[yy[inside],xx[inside]]
        hits = dist <= 2.0
        coverage = float(hits.mean())
        sectors = hits.reshape(24,4).any(axis=1).mean()
        if coverage < .62 or sectors < .70: continue
        quality = coverage - .20*np.hypot((X-cx)/w,(Y-cy)/h)
        if best is None or quality > best['quality']:
            best = dict(center=[float(X),float(Y)],major=float(major),minor=float(minor),
                        angle=float(angle),quality=float(quality),coverage=float(coverage),source='ellipse')
    return best


def smooth_ellipses(rows, window_s=.25):
    """Median axes and doubled-angle orientation, only on successful frames."""
    measured = [(r['t'],r['plate']['ellipse']) for r in rows if r.get('plate',{}).get('ellipse',{}).get('source') == 'ellipse']
    if not measured: return
    ts = np.array([a for a,_ in measured]); fits = [b for _,b in measured]
    for r in rows:
        e = r.get('plate',{}).get('ellipse')
        if not e or e.get('source') != 'ellipse': continue
        idx = np.flatnonzero(np.abs(ts-r['t']) <= window_s/2 + 1e-9)
        window = [fits[i] for i in idx]
        e['smooth_major'] = float(np.median([x['major'] for x in window]))
        e['smooth_minor'] = float(np.median([x['minor'] for x in window]))
        a = np.array([x['angle'] for x in window])
        e['smooth_angle'] = float(.5*np.arctan2(np.median(np.sin(2*a)),np.median(np.cos(2*a))) % np.pi)


def interpolate_ellipses(rows, max_gap_s=.35):
    """Fill between two successful samples only, retaining box fallbacks elsewhere."""
    samples = [i for i,r in enumerate(rows) if r.get('plate',{}).get('ellipse',{}).get('source') == 'ellipse']
    for left,right in zip(samples,samples[1:]):
        a,b=rows[left],rows[right]; dt=b['t']-a['t']
        if dt <= 0 or dt > max_gap_s: continue
        ea,eb=a['plate']['ellipse'],b['plate']['ellipse']
        # Interpolate centre offsets from the tracker, not absolute position.
        ca=np.asarray(ea['center'])-a['plate']['center']; cb=np.asarray(eb['center'])-b['plate']['center']
        da=((eb['angle']-ea['angle']+np.pi/2)%np.pi)-np.pi/2
        for i in range(left+1,right):
            row=rows[i];p=row.get('plate',{})
            if not p.get('used') or p.get('ellipse',{}).get('sampled'): continue
            f=(row['t']-a['t'])/dt
            p['ellipse']={'source':'ellipse','sampled':False,'center':(np.asarray(p['center'])+(1-f)*ca+f*cb).tolist(),
                          'major':float((1-f)*ea['major']+f*eb['major']),
                          'minor':float((1-f)*ea['minor']+f*eb['minor']),
                          'angle':float((ea['angle']+f*da)%np.pi),
                          'quality':float(min(ea['quality'],eb['quality'])),
                          'coverage':float(min(ea['coverage'],eb['coverage']))}


def annotate_video(path, rows):
    """Attach optional rim observations without changing the saved box rows."""
    from core.io import video
    step = max(1, round(video.info(path).fps * .25))
    for i,(_,frame) in enumerate(video.frames(path)):
        if i >= len(rows): break
        p = rows[i].get('plate',{})
        if not p.get('used'): continue
        sampled = i % step == 0
        e = fit_plate_ellipse(frame,p['box']) if sampled else None
        p['ellipse'] = {**e,'sampled':True} if e else {'source':'box','quality':0.,'sampled':sampled}
    interpolate_ellipses(rows)
    smooth_ellipses(rows)
