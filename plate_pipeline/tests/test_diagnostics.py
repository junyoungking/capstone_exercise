import builtins
import csv
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core.recog.plate import Detection, OnnxPlateDetector
from eval import counts
from eval.debug_track import diagnose, timeline, debug_video, count_times
from server import jobs


def ort_stub(monkeypatch,available,active=None):
    calls=[]
    class Session:
        def __init__(self,path,providers):
            self.providers=active if active is not None else providers
            self.disabled=False
            calls.append(self)
            self.requested=providers
        def get_providers(self):
            return self.providers
        def disable_fallback(self):
            self.disabled=True
        def get_inputs(self):
            return [SimpleNamespace(name='images',shape=[1,3,416,416])]
    monkeypatch.setitem(sys.modules,'onnxruntime',SimpleNamespace(
        get_available_providers=lambda:available,InferenceSession=Session))
    return calls


def test_auto_cpu_without_torch(monkeypatch):
    calls=ort_stub(monkeypatch,['CPUExecutionProvider'])
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='torch':
            raise AssertionError('CPU must not import torch')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    det=OnnxPlateDetector('unused')
    assert det.providers==['CPUExecutionProvider']
    assert calls[0].requested==['CPUExecutionProvider']
    with pytest.raises(RuntimeError,match='not available'):
        OnnxPlateDetector('unused',device='cuda')


def test_cuda_must_activate_and_disable_fallback(monkeypatch):
    calls=ort_stub(monkeypatch,['CUDAExecutionProvider','CPUExecutionProvider'])
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    det=OnnxPlateDetector('unused',device='cuda')
    assert calls[0].disabled and 'CUDAExecutionProvider' in det.providers
    cpu=OnnxPlateDetector('unused',device='cpu')
    assert cpu.providers==['CPUExecutionProvider']
    ort_stub(monkeypatch,['CUDAExecutionProvider','CPUExecutionProvider'],active=['CPUExecutionProvider'])
    with pytest.raises(RuntimeError,match='not active'):
        OnnxPlateDetector('unused',device='cuda')
    assert OnnxPlateDetector('unused').providers==['CPUExecutionProvider']


def test_cuda_preload_before_optional_torch(monkeypatch):
    calls=ort_stub(monkeypatch,['CUDAExecutionProvider','CPUExecutionProvider'])
    order=[]
    sys.modules['onnxruntime'].preload_dlls=lambda:order.append('preload')
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='torch':
            order.append('torch')
            raise ImportError('optional torch absent')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert OnnxPlateDetector('unused').providers[0]=='CUDAExecutionProvider'
    assert order==['preload','torch'] and len(calls)==1


def test_candidate_logging_is_sorted_bounded_and_does_not_change_plate(monkeypatch):
    img=np.zeros((100,100,3),np.uint8)
    monkeypatch.setattr(jobs.video,'frames',lambda _:iter([(0,img.copy()),(.1,img.copy())]))
    dets=[Detection((20,20,60,60),.2+i*.05) for i in range(14)]
    picked=jobs.pick_detection(dets,None)
    assert picked is dets[-1]
    rows=jobs.recognize('unused',detector=lambda *_:dets,rim=False)
    assert rows[0]['dets']==[[*map(float,d.box),d.score] for d in reversed(dets[4:])]
    assert rows[0]['pick']==0 and len(rows[0]['dets'])==10
    # Within the high-score tier, closest-center ties retain input order.
    # Logging must not sort or truncate the selection candidates.
    expected=jobs.recognize('unused',detector=lambda img,seq:[picked if seq==0 else dets[2]],rim=False)
    assert [r['plate'] for r in rows]==[r['plate'] for r in expected]
    assert rows[1]['pick'] is None  # selected candidate falls outside the stored top ten
    no_det=jobs.recognize('unused',init_box=picked.box,rim=False)
    assert all('dets' not in r and 'pick' not in r for r in no_det)
    empty=jobs.recognize('unused',detector=lambda *_:[],rim=False)
    assert all(r['dets']==[] and r['pick'] is None for r in empty)


def synthetic_rows():
    rng=np.random.default_rng(4)
    rows=[]
    for k in range(61):
        cy=130+30*np.sin(k/30*2*np.pi)
        height=100 if k<30 else 150
        box=[100,cy-height/2,200,cy+height/2]
        kp=np.zeros((33,3));kp[15]=[150,cy+rng.normal(0,.3),1]
        rows.append(dict(seq=k,t=k/30,
            plate=dict(box=box,center=[150,cy],size_px=[100,height],src='det',used=True),
            dets=[[400,0,500,100,.8],[700,10+k*2,800,110+k*2,.7]],
            kp=kp.tolist(),pose=dict(n=1,ms=1)))
    return rows


def test_diagnostic_metrics_and_timeline(tmp_path):
    rows=synthetic_rows()
    result=dict(height=480,fps=30,exercise='squat',video='synthetic',sets=[])
    stats=diagnose(rows,result)
    assert stats['moving_cands']['n']==1
    assert stats['moving_cands']['intervals']==[[0,2]]
    assert len(stats['jumps'])==1 and stats['jumps'][0]['t']==1
    assert stats['wrist_r']>.9 and stats['wrist_windows'][0]['r']>.9
    assert stats['wrist_seen']==1
    timeline(rows,result,tmp_path/'timeline.png',truth=0,stats=stats)
    assert (tmp_path/'timeline.png').stat().st_size>1000
    legacy=[{k:v for k,v in r.items() if k not in ('dets','pose','kp')} for r in rows]
    old=diagnose(legacy,result)
    assert old['moving_cands']['n']==0 and old['wrist_r'] is None
    timeline(legacy,result,tmp_path/'legacy.png')


def test_debug_video_without_dets_or_pose(tmp_path,monkeypatch):
    import cv2
    rows=synthetic_rows()[:2]
    for r in rows:
        r.pop('dets');r.pop('kp');r.pop('pose')
    monkeypatch.setattr('eval.debug_track.video.info',lambda _:SimpleNamespace(height=480,width=640,fps=30))
    monkeypatch.setattr('eval.debug_track.video.frames',lambda _:iter((r['t'],np.zeros((480,640,3),np.uint8)) for r in rows))
    path=tmp_path/'debug.mp4'
    debug_video(rows,{'sets':[]},'unused',path)
    cap=cv2.VideoCapture(str(path))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_WIDTH)==540
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==2
    cap.release()


def test_video_count_includes_reps_with_missing_speed_endpoint():
    from test_core import squat_like
    t,y,x=squat_like(3)
    rows=[]
    for tt,yy in zip(t,y):
        cy=500-yy*100/.45
        rows.append(dict(t=float(tt),plate=dict(used=True,box=[0,cy-50,100,cy+50],
                         center=[50,cy],size_px=[100,100])))
    sets,_=jobs.analyse(rows,'squat',30)
    assert sum(s['reps'] for s in sets)==3
    sets[0]['rep_values'][0]['back']=None
    assert len(count_times(rows,dict(exercise='squat',fps=30,sets=sets)))==3


def test_counts_root_and_refuses_existing_output(tmp_path,monkeypatch):
    root=tmp_path/'videos';(root/'squat').mkdir(parents=True)
    path=root/'squat'/'clip.mp4';path.touch()
    sets=tmp_path/'sets.csv';sets.write_text('path,exercise,reps\nclip.mp4,squat,2\n')
    calls=[]
    def process(path,exercise,plate_mm,**kw):
        calls.append((path,kw))
        return dict(sets=[{'reps':1},{'reps':1}],recog_fps=12,detector_providers=['CPUExecutionProvider']),[],{}
    monkeypatch.setattr(counts,'process_video',process)
    out=tmp_path/'evaluation'
    args=['--sets',str(sets),'--root',str(root),'--out',str(out),'--device','cpu']
    counts.main(args)
    assert calls[0][0]==str(path) and calls[0][1]['device']=='cpu'
    row=next(csv.DictReader(open(out/'counts.csv')))
    assert row['got']=='2' and row['n_sets']=='2' and row['exact']=='True' and row['right']=='False'
    assert counts.resolve_video('squat/clip.mp4',root)==path
    with pytest.raises(FileExistsError):
        counts.main(args)
    assert len(calls)==1
    (root/'other').mkdir();(root/'other'/'clip.mp4').touch()
    with pytest.raises(ValueError,match='Ambiguous'):
        counts.resolve_video('clip.mp4',root)
