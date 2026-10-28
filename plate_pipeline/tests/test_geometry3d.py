"""M2 numerical contracts without external models/videos."""
from dataclasses import replace
import numpy as np
import pytest
from core.config import DEFAULT,PoseCfg
from core.geometry import plate_geometry,stature,angle
from core.geometry.pose3d import fixed_bones,body_basis,scale_floor,to_mp33,MP_FROM_H36M,PARENTS,selected_coco,lift


def test_pose_defaults():
    assert (PoseCfg().mode,PoseCfg().det_frequency,PoseCfg().bench_rep_guard)==('tracked',10,False)


def test_plate_proportional_and_lifter_affine_diameter_laws():
    boxes=np.tile([10,20,110,220],(5,1))
    a=plate_geometry(boxes,(1000,600),450);b=plate_geometry(boxes,(1000,600),400)
    assert a['angle_deg']==pytest.approx(60) and a['focal_px']==770
    np.testing.assert_allclose(a['distance_m'],770*.45/200)
    np.testing.assert_allclose(b['distance_m']/a['distance_m'],400/450)
    depths=[np.arange(5)*2*g['plate_m_per_px'] for g in (a,b)]
    np.testing.assert_allclose(depths[1],depths[0]*400/450)
    np.testing.assert_allclose(np.diff(depths[1])*30,np.diff(depths[0])*30*400/450)
    from core.signal.height import height_signal
    rows=[dict(t=i/30,cam=[0,0],plate=dict(used=True,center=[60,100-i*2],size_px=[100,200],src='det')) for i in range(5)]
    t,y450,_=height_signal(rows,450);_,y400,_=height_signal(rows,400)
    np.testing.assert_allclose(y400,y450*400/450)
    np.testing.assert_allclose(np.diff(y400)/np.diff(t),np.diff(y450)/np.diff(t)*400/450)
    np.testing.assert_allclose(a['kd'],(a['distance_m']+.70*.5)/a['distance_m'])
    expected=.400/200+.70*.5/770
    np.testing.assert_allclose(b['lifter_m_per_px'],expected)
    raw=np.zeros((5,33,4));raw[:,:,3]=1;raw[:,0,1]=10;raw[:,27:33,1]=510
    h,used=stature(raw,b,np.ones(5,bool))
    assert used.all() and h==pytest.approx(500*expected/.92)
    assert not np.isclose(h/(500*a['lifter_m_per_px'][0]/.92),400/450)
    with pytest.raises(ValueError):plate_geometry(boxes,(1000,600),0)


def skeleton():
    p=np.zeros((5,17,3))
    for j,a in enumerate(PARENTS):
        if a>=0:p[:,j]=p[:,a]+[.1*j,.2,.03*j]
    return p


def test_fixed_bones_preserve_direction_per_set():
    p=skeleton();p[0]*=.7;p[-1]*=1.4
    out,targets=fixed_bones(p,[0,0,0,1,1])
    for j,a in enumerate(PARENTS):
        if a<0:continue
        for ids in ([0,1,2],[3,4]):
            lengths=np.linalg.norm(out[ids,j]-out[ids,a],axis=1)
            np.testing.assert_allclose(lengths,lengths[0])
        before=p[:,j]-p[:,a];after=out[:,j]-out[:,a]
        np.testing.assert_allclose(before/np.linalg.norm(before,axis=1)[:,None],after/np.linalg.norm(after,axis=1)[:,None])
    assert set(targets)=={0,1}


def test_body_coordinates_angles_are_rotation_invariant():
    p=skeleton();raw=np.zeros((5,33,4));raw[:,:,3]=1;raw[:,[31,32],0]=1
    stand=np.ones(5,bool);basis,_=body_basis(p,stand,raw)
    np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-10)
    t=.6;r=np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]])
    rotated=p@r.T;basis2,_=body_basis(rotated,stand,raw)
    q=p@basis.T;q2=rotated@basis2.T
    for a,b,c in ((1,2,3),(11,4,5)):
        np.testing.assert_allclose(angle(q[:,a],q[:,b],q[:,c]),angle(q2[:,a],q2[:,b],q2[:,c]),atol=1e-6)
    v=q[:,9]-(q[:,1]+q[:,4])/2;v2=q2[:,9]-(q2[:,1]+q2[:,4])/2
    np.testing.assert_allclose(np.arctan2(np.linalg.norm(v[:,[0,2]],axis=1),v[:,1]),np.arctan2(np.linalg.norm(v2[:,[0,2]],axis=1),v2[:,1]),atol=1e-8)
    np.testing.assert_allclose((rotated[:,9]-(rotated[:,3]+rotated[:,6])/2)@basis2[0],0,atol=1e-10)


def test_scale_floor_and_segment_law():
    p=skeleton();p[:,9,1]=1.5;p[:,3,1]=0;p[:,6,1]=.1
    a,s,reason=scale_floor(p,np.ones(5,bool),1.8)
    assert reason is None and s==pytest.approx(1.8*.92/1.5)
    np.testing.assert_allclose(a[:,[3,6],1].min(axis=1),0)
    b,_,_=scale_floor(p,np.ones(5,bool),1.6)
    np.testing.assert_allclose(np.linalg.norm(b[:,1]-b[:,2],axis=1)/np.linalg.norm(a[:,1]-a[:,2],axis=1),1.6/1.8)
    _,s,reason=scale_floor(p,np.zeros(5,bool),None)
    assert s==1 and 'native' in reason


def test_mp33_and_coco_mapping_without_person_reselection():
    p=np.arange(2*17*3).reshape(2,17,3);mp=to_mp33(p)
    for j,h in MP_FROM_H36M.items():np.testing.assert_array_equal(mp[:,j],p[:,h])
    assert np.isnan(mp[:,17:23]).all()
    pose=dict(raw_lms=np.ones((2,33,4)),valid=np.array([True,False]),w=640,h=480)
    coco=selected_coco(pose);np.testing.assert_array_equal(coco[0],np.tile([640,480],(17,1)))
    assert np.isnan(coco[1]).all()


def test_chunked_full_context_and_gap_rules():
    import torch
    from track3d.vendor.videopose3d.model import TemporalModel
    torch.manual_seed(7);torch.set_num_threads(1)
    m=TemporalModel(17,2,17,[3,3],channels=8,dropout=0).eval()
    points=np.random.default_rng(3).random((48,17,2)).astype(np.float32)*100
    a,_,_=lift(m,points,640,480,replace(DEFAULT.pose3d,chunk_frames=9))
    b,_,_=lift(m,points,640,480,replace(DEFAULT.pose3d,chunk_frames=100))
    np.testing.assert_allclose(a,b,atol=1e-6)
    points[3:5]=np.nan;points[20:32]=np.nan
    _,filled,nulls=lift(m,points,640,480)
    assert filled[3:5].all() and nulls[20:32].all() and not filled[20:32].any()
