"""v40's causal COCO17/8-channel feature contract and optional action vote."""
from pathlib import Path
import os
import numpy as np

# COCO17 -> MP33 indices, matching the first 17 Halpe joints.
COCO_FROM_MP=(0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28)
ACTIONS=('squat','benchpress','deadlift')


def make_features(pose,timestamps,resets=None):
    """Causal 8 Hz sample/normalization from the v40 data.make_features code."""
    p=np.asarray(pose,dtype=np.float32);raw_t=np.asarray(timestamps,float)
    if p.ndim!=3 or p.shape[1:]!=(17,3) or len(p)!=len(raw_t) or not len(p):raise ValueError('poses must be (T,17,3)')
    if not np.isfinite(raw_t).all() or (np.diff(raw_t)<=0).any():raise ValueError('invalid timestamps')
    hz=8.;dt=1/hz;stop=int(np.ceil((raw_t[-1]-raw_t[0])*hz-1e-9));ts=raw_t[0]+np.arange(stop+1)/hz
    idx=np.maximum(np.searchsorted(raw_t,ts+1e-9,side='right')-1,0);age=ts-raw_t[idx];q=p[idx].copy()
    reliable=np.isfinite(q).all(axis=-1)&(q[:,:,2]>=.3)
    selected=np.isfinite(p).any(axis=(1,2))[idx]
    observed=selected&(age<=dt+1e-6)&(reliable.sum(1)>=6)&reliable[:,[5,6]].any(1)&reliable[:,[11,12]].any(1)
    raw_reset=np.asarray(resets if resets is not None else np.zeros(len(raw_t),bool),bool)
    reset=np.zeros(len(ts),bool);previous=-1
    for i,j in enumerate(idx):reset[i]=raw_reset[previous+1:j+1].any();previous=j
    reset[0]=True;features=np.zeros((len(ts),17,8),np.float32);last_xy=last_root=last_t=None;last_valid=np.zeros(17,bool)
    for i,t in enumerate(ts):
        if reset[i]:last_xy=last_root=last_t=None
        if not observed[i]:last_xy=last_root=last_t=None;continue
        v=reliable[i];hips=q[i,[11,12],:2][v[[11,12]]];shoulders=q[i,[5,6],:2][v[[5,6]]]
        root=hips.mean(0);scale=np.linalg.norm(shoulders.mean(0)-root)
        if not np.isfinite(scale) or scale<1e-5:observed[i]=False;last_xy=last_root=last_t=None;continue
        xy=np.zeros((17,2),np.float32);xy[v]=(q[i,v,:2]-root)/scale
        vel=np.zeros_like(xy);rootvel=np.zeros(2,np.float32)
        if last_xy is not None and 0<t-last_t<=dt+1e-6:
            common=v&last_valid;vel[common]=(xy[common]-last_xy[common])/(t-last_t);rootvel=(root-last_root)/scale/(t-last_t)
        features[i,:,:2]=xy;features[i,:,2]=np.where(v,q[i,:,2],0);features[i,:,3:5]=np.clip(vel,-20,20);features[i,:,5]=v;features[i,:,6:8]=np.clip(rootvel,-20,20)
        last_xy,last_root,last_valid,last_t=xy,root,v,t
    return ts,features,observed,reset


def infer(pose,rows,rep_intervals,checkpoint=None,threads=6):
    checkpoint=checkpoint or os.environ.get('V40_CHECKPOINT')
    if not checkpoint or not Path(checkpoint).is_file():return {'status':'unavailable','reason':'v40 체크포인트 없음','exercise':None,'votes':{}}
    raw=np.asarray(pose.get('raw_lms'));scores=pose.get('raw_scores')
    if scores is None or raw.ndim!=3 or len(raw)!=len(rows):return {'status':'unavailable','reason':'원본 COCO 관절 점수 없음','exercise':None,'votes':{}}
    scores=np.asarray(scores,dtype=np.float32)
    if scores.shape!=(len(raw),17):return {'status':'unavailable','reason':'원본 COCO 점수 형태 불일치','exercise':None,'votes':{}}
    xy=raw[:,COCO_FROM_MP,:2]*np.array([pose['w'],pose['h']]);points=np.concatenate([xy,scores[:,:,None]],axis=2)
    times=np.asarray([r['t'] for r in rows]);valid=np.asarray(pose['valid'],bool);points[~valid]=np.nan
    resets=pose.get('selection_reset')
    if resets is None or np.asarray(resets).shape!=(len(rows),):
        return {'status':'unavailable','reason':'포즈 선택 리셋 기록 없음','exercise':None,'votes':{}}
    resets=np.asarray(resets,bool).copy();resets[0]=True
    ts,x,observed,resets=make_features(points,times,resets)
    ids=[i for i,t in enumerate(ts) if observed[i] and any(a<=t<=b for a,b in rep_intervals)]
    if not ids:return {'status':'unavailable','reason':'rep 구간의 유효 포즈 없음','exercise':None,'votes':{}}
    import torch
    from core.auto.v40_model import STGCN_TCN
    state=torch.load(str(checkpoint),map_location='cpu',weights_only=False);model=STGCN_TCN(base=32);model.load_state_dict(state['model']);model.eval();votes=[]
    previous=torch.get_num_threads();torch.set_num_threads(threads)
    try:
        with torch.no_grad():
            windows=[]
            for i in ids:
                start=max(0,i-31);reset=np.flatnonzero(resets[start:i+1]);start+=int(reset[-1]) if len(reset) else 0
                window=np.zeros((32,17,8),np.float32);window[-(i-start+1):]=x[start:i+1]
                windows.append(window.transpose(2,0,1))
            for start in range(0,len(windows),64):
                a,_=model(torch.from_numpy(np.stack(windows[start:start+64])));votes.extend(a.argmax(1).tolist())
    finally:torch.set_num_threads(previous)
    counts={name:votes.count(i) for i,name in enumerate(ACTIONS)};best=max(counts.values());winners=[k for k,v in counts.items() if v==best]
    return {'status':'ok','exercise':winners[0] if len(winners)==1 else None,'votes':counts}
