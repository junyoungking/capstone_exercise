"""Physical tracking contracts, using textured moving discs and fake detections."""
import cv2
import numpy as np
import pytest

from core.recog.plate import Detection
from core.recog.track import PlateTracker
from core.signal.height import height_signal
from server import jobs


def box(x=100, y=100, size=80):
    return (x-size/2, y-size/2, x+size/2, y+size/2)


def image(y=100):
    rng = np.random.default_rng(15)
    im = rng.integers(140, 210, (320, 320, 3), dtype=np.uint8)
    cv2.circle(im, (100, y), 40, (20, 20, 20), -1)
    for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
        cv2.circle(im, (int(100+27*np.cos(a)), int(y+27*np.sin(a))), 3, (240,240,240), -1)
    return im


def recognize(monkeypatch, detections, ys=None):
    ys = ys or [100+i for i in range(len(detections))]
    monkeypatch.setattr(jobs.video, 'frames', lambda _: iter((i/30, image(y)) for i,y in enumerate(ys)))
    return jobs.recognize('fake', detector=lambda _,i: detections[i], rim=False)


def test_low_cannot_start_and_high_tier_precedes_low(monkeypatch):
    rows = recognize(monkeypatch, [[Detection(box(), .2)], [Detection(box(y=101), .8)],
                                  [Detection(box(y=102), .1), Detection(box(y=103), .7)],
                                  [Detection(box(y=103), .1)]])
    assert 'plate' not in rows[0]
    assert rows[2]['plate']['src'] == 'det'
    assert rows[3]['plate']['src'] == 'det_low'
    assert rows[3]['plate']['used']


def test_vertical_recovery_rejects_wrong_x_size_and_low():
    tr = PlateTracker()
    gray = np.zeros((320,320), np.uint8)  # full occlusion, no usable KLT points
    tr.step(gray, Detection(box(), .9), t=0)
    for i in range(1, 16):
        assert tr.step(gray, t=i/30).src == 'lost'
    assert tr.step(gray, dets=[Detection(box(y=220), .2), Detection(box(x=230,y=220), .9),
                              Detection(box(y=220,size=120), .9)], t=.51).src == 'lost'
    state = tr.step(gray, dets=[Detection(box(y=250), .9), Detection(box(y=220), .8)], t=.54)
    assert state.src == 'det' and state.center == (100,220)


@pytest.mark.parametrize('scale', [.7, 1.4])
def test_klt_step_scale_is_bounded(monkeypatch, scale):
    gray = cv2.cvtColor(image(), cv2.COLOR_BGR2GRAY)
    tr = PlateTracker()
    tr.step(gray, Detection(box(), .9), t=0)
    def flow(a,b,pts,*args,**kwargs):
        return pts.copy(), np.ones((len(pts),1),np.uint8), None
    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', flow)
    monkeypatch.setattr(cv2, 'estimateAffinePartial2D', lambda a,b,**k:
                        (np.array([[scale,0,100*(1-scale)],[0,scale,100*(1-scale)]]), np.ones((len(a),1),np.uint8)))
    state = tr.step(gray, t=.04)
    assert state.src == 'klt'
    assert state.size[0]/80 == pytest.approx(np.clip(scale,.98,1.02))


def test_drift_marks_only_klt_since_last_measurement(monkeypatch):
    # A detection 16px from the KLT prediction exceeds 7cm (12.44px at this scale).
    ds = [[Detection(box(),.9)], [], [], [Detection(box(x=116,y=103),.2)]]
    rows = recognize(monkeypatch, ds)
    assert rows[0]['plate']['src'] == 'det'
    assert all(r['plate']['src']=='klt_drift' and not r['plate']['used'] for r in rows[1:3])
    assert rows[3]['plate']['src'] == 'det_low'
    assert len(height_signal(rows)[0]) == 2


def test_measured_scale_ignores_klt_sizes_and_uses_nearest():
    rows = [{'t':i/10, 'plate':{'used':True, 'src':'det' if i in (0,80) else 'klt',
                              'center':[100+i,300-i], 'size_px':[90,90]}} for i in range(81)]
    before = height_signal(rows)
    for r in rows:
        if r['plate']['src']=='klt':
            r['plate']['size_px']=[10+r['t']*100, 400-r['t']*40]
    after = height_signal(rows)
    for a,b in zip(before,after):
        np.testing.assert_array_equal(a,b)
    assert after[1][-1] == pytest.approx(.4)


def test_low_cannot_recover_before_corridor():
    tr = PlateTracker(); gray=np.zeros((320,320),np.uint8)
    tr.step(gray, Detection(box(),.9), t=0)
    tr.step(gray, t=.03)
    assert tr.step(gray, Detection(box(),.2), t=.1).src == 'lost'
    assert tr.step(gray, Detection(box(y=220),.9), t=.2).src == 'lost'
    assert tr.step(gray, Detection(box(),.9), t=.25).src == 'det'


def test_measured_window_median_and_nearest():
    # At t=4 only t=2 is in the inclusive +/-2.5s window; at t=6 neither is.
    rows=[]
    for t,src,size in [(0,'rim',60),(2,'det_low',120),(4,'klt',900),(6,'klt',3),(10,'det',240)]:
        rows.append({'t':t,'plate':{'used':True,'src':src,'center':[100,300-t*10],'size_px':[size,size]}})
    _,y,_=height_signal(rows)
    scales=np.array([.45/90,.45/90,.45/120,.45/120,.45/240])
    np.testing.assert_allclose(np.diff(y),np.diff([0,2,4,6,10])*10*(scales[1:]+scales[:-1])/2)


def test_occluded_disc_recovers_at_different_y(monkeypatch):
    frames=[(i/30,image(100) if i==0 else image(220)) for i in range(18)]
    background=np.random.default_rng(2).integers(140,210,(320,320,3),dtype=np.uint8)
    frames[1:17]=[(i/30,background.copy()) for i in range(1,17)]
    monkeypatch.setattr(jobs.video,'frames',lambda _:iter(frames))
    rows=jobs.recognize('fake',detector=lambda _,i:[Detection(box(),.9)] if i==0 else
                        ([Detection(box(y=220),.9),Detection(box(x=250,y=100),.95)] if i==17 else []),rim=False)
    assert rows[-2]['plate']['src']=='lost'
    assert rows[-1]['plate']['src']=='det'
    assert rows[-1]['plate']['center']==[100,220]


def test_detector_includes_exact_low_boundary():
    from core.recog.plate import decode_yolox, yolox_grids
    grid, _ = yolox_grids(32)
    out = np.zeros((len(grid), 6), np.float32)
    out[0, 4:] = [.05, 1.0]
    dets = decode_yolox(out, 32, 1.0)
    assert len(dets) == 1 and dets[0].score == pytest.approx(.05)
