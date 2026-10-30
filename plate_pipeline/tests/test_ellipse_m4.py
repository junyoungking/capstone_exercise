import cv2
import numpy as np
import pytest
from core.recog.ellipse import fit_plate_ellipse, smooth_ellipses
from core.signal.height import ellipse_matrix, path_axes, height_signal, corrected_height_signal
from core.geometry import plate_geometry


def test_synthetic_rim_with_sleeve_and_noise():
    rng = np.random.default_rng(42)
    im = np.full((360,360,3), 150, np.uint8)
    cv2.ellipse(im,(180,180),(90,55),20,0,360,(30,30,30),3)
    cv2.rectangle(im,(175,175),(242,185),(210,210,210),-1)
    im = np.clip(im.astype(float)+rng.normal(0,3,im.shape),0,255).astype(np.uint8)
    e = fit_plate_ellipse(im,(90,90,270,270))
    assert e is not None
    assert abs(e['major']-180)<8 and abs(e['minor']-110)<8
    assert e['coverage']>=.54


def test_offscreen_and_blank_rejected():
    im = np.full((200,200,3),128,np.uint8)
    assert fit_plate_ellipse(im,(20,20,120,120)) is None
    cv2.ellipse(im,(10,100),(60,45),0,0,360,(0,0,0),3)
    assert fit_plate_ellipse(im,(-50,40,70,160)) is None


def test_symmetric_rectification():
    a,b,theta=140.,70.,.4
    m=ellipse_matrix(a,b,theta)
    v=np.array([np.cos(theta),np.sin(theta)])
    w=np.array([-v[1],v[0]])
    assert np.allclose(m,m.T)
    assert np.linalg.norm(m @ (a/2*v)) == pytest.approx(.225)
    assert np.linalg.norm(m @ (b/2*w)) == pytest.approx(.225)


def test_path_axis_and_short_fallback():
    up,side=path_axes(np.array([[0.,0.],[.03,-.2],[.06,-.4]]))
    assert up[1] < 0 and np.dot(up,side) == pytest.approx(0,abs=1e-12)
    up,_=path_axes(np.array([[0.,0.],[.001,-.001]]))
    assert np.allclose(up,[0,-1])


def test_box_signal_unchanged_and_rejected_fallback():
    rows=[{'t':i*.1,'plate':{'used':True,'src':'det','center':[100.,100.-i], 'size_px':[90.,100.], 'box':[55,50-i,145,150-i], 'ellipse':{'source':'box','quality':0}},'cam':[0,0]} for i in range(8)]
    baseline=height_signal(rows)
    for a,b in zip(baseline,corrected_height_signal(rows)): assert np.array_equal(a,b)
    t,y,x=corrected_height_signal(rows,scale_source='ellipse')
    assert np.allclose(y,baseline[1]) and np.allclose(x,baseline[2])


def test_angle_statistics_and_60_degree_threshold():
    e={'source':'ellipse','major':100.,'minor':40.,'angle':0.,'quality':1.}
    g=plate_geometry([[0,0,100,100]],(640,480),ellipses=[e])
    assert g['angle_source']=='ellipse' and g['angle_deg']>60
    assert g['ellipse_used_fraction']==1
    g=plate_geometry([[0,0,100,100]],(640,480),ellipses=[None])
    assert g['angle_source']=='box'
