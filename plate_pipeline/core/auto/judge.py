"""Evidence-based exercise candidate; absence or disagreement requires confirmation."""
import numpy as np
from core.config import DEFAULT
from core.geometry import box_angle_reliable
from core.signal.height import height_signal
from core.signal.final import clean_final
from core.judge.session import run_session,REST
from core.measure.times import all_rep_values
from core.auto import v40


def _intervals(sig,raw_t):
    out={'top':[],'bottom':[]}
    for exercise in ('squat','deadlift'):
        sets=run_session(sig.t,sig.y,sig.x,sig.gap_before,exercise,'final',cfg=DEFAULT)
        trips=[r for s in sets for r in s.reps]
        vals=all_rep_values(sig,raw_t,trips,REST[exercise],DEFAULT.speed)
        out[REST[exercise]].extend((v.start,v.back) for v in vals if v.start is not None and v.back is not None and v.start<v.back)
    return out


def _body_frame(p,w,h):
    if p.shape!=(33,4):return None
    ids=[11,12,23,24];good=np.isfinite(p[ids]).all(axis=1)&(p[ids,3]>=.5)
    if not (good[:2].any() and good[2:].any()):return None
    q=p[:,:2]*[w,h];shoulder=q[[11,12]][good[:2]].mean(0);hip=q[[23,24]][good[2:]].mean(0)
    dx,dy=np.abs(shoulder-hip)
    if np.hypot(dx,dy)<.02*min(w,h):return None
    if dx>dy*1.25:return 'lying'
    if dy>dx*1.25:return 'standing'
    return None


def _floor_value(p,plate,w,h):
    if p.shape!=(33,4):return None
    pix=p[:,:2]*[w,h];vis=p[:,3]
    # Require at least one supported, in-frame, ordered shoulder/hip/knee/ankle chain.
    ordered=False
    for shoulder,hip,knee,ankle in ((11,23,25,27),(12,24,26,28)):
        ids=[shoulder,hip,knee,ankle]
        if not np.isfinite(p[ids]).all() or np.any(vis[ids]<.5):continue
        if np.any(pix[ids,0]<0) or np.any(pix[ids,0]>=w) or np.any(pix[ids,1]<0) or np.any(pix[ids,1]>.97*h):continue
        tolerance=.05*h
        if pix[shoulder,1]<=pix[hip,1]+tolerance and pix[hip,1]<=pix[knee,1]+tolerance and pix[knee,1]<=pix[ankle,1]+tolerance:
            ordered=True;break
    if not ordered:return None
    foot=np.array([27,28,29,30,31,32]);valid=np.isfinite(p[foot]).all(axis=1)&(vis[foot]>=.5)&(pix[foot,0]>=0)&(pix[foot,0]<w)&(pix[foot,1]>=0)&(pix[foot,1]<=.97*h)
    if not valid.any():return None
    box=plate.get('box');size=plate.get('size_px')
    if box is None or size is None or size[1]<=0:return None
    return float((pix[foot[valid],1].max()-box[3])/size[1])


def assess(rows,pose,checkpoint=None):
    if len(rows)!=len(pose['valid']):raise ValueError('pose and plate frames differ')
    w,h=int(pose['w']),int(pose['h']);t,y,x=height_signal(rows,450.)
    result={'rest':'unknown','floor':'unknown','posture':'unknown','classifier':{'status':'unavailable','exercise':None,'reason':'분류기 입력 없음'},
            'candidate':None,'status':'needs_confirmation','exercise':None,'angle':{'reliable':False,'degrees':None},'rep_intervals':[],
            'selection_ambiguous':None}
    ambiguity=pose.get('selection_ambiguous')
    if ambiguity is not None and np.asarray(ambiguity).shape==(len(rows),):
        result['selection_ambiguous']=bool(np.any(np.asarray(ambiguity,bool)))
    boxes=[r['plate']['box'] for r in rows if r.get('plate',{}).get('used') and r['plate'].get('box')]
    if boxes and box_angle_reliable(boxes,(w,h)):
        a=np.asarray(boxes,float);ratio=np.median((a[:,2]-a[:,0])/(a[:,3]-a[:,1]));result['angle']={'reliable':True,'degrees':float(np.degrees(np.arccos(np.clip(ratio,0,1))))}
    if len(t)<6:return result
    fps=len(rows)/max(rows[-1]['t']-rows[0]['t'],1e-6);fr=clean_final(t,y,x,fps);s=fr.sig
    by_rest=_intervals(s,fr.raw_t);intervals=sorted(by_rest['top']+by_rest['bottom'])
    result['rep_intervals']=[[float(a),float(b)] for a,b in intervals]
    origin=float(np.median(s.y[:min(5,len(s.y))]));where=np.flatnonzero(np.abs(s.y-origin)>=.13)
    if not len(where):return result
    cross=int(where[0]);first=float(s.t[cross]);direction='bottom' if s.y[cross]>origin else 'top'
    raw=np.asarray(pose.get('raw_lms',pose['lms']),float);valid=np.asarray(pose['valid'],bool)
    floor=[];rest_body=[];rep_body=[]
    for i,row in enumerate(rows):
        if not valid[i] or not np.isfinite(raw[i]).all():continue
        tm=float(row['t']);body=_body_frame(raw[i],w,h)
        if tm<first:
            if body:rest_body.append(body)
            p=row.get('plate',{})
            if p.get('used'):
                value=_floor_value(raw[i],p,w,h)
                if value is not None:floor.append(value)
        if body and any(a<=tm<=b for a,b in intervals):rep_body.append(body)
    if len(floor)>=5:
        d=float(np.median(floor));result['floor']='top' if d>.25 else ('bottom' if d>=-.25 else 'unknown')
    if result['floor']==direction and by_rest[direction]:result['rest']=direction
    def vote(values):
        if len(values)<5:return None
        lying=values.count('lying');standing=values.count('standing')
        return 'lying' if lying>standing else 'standing' if standing>lying else None
    a,b=vote(rest_body),vote(rep_body)
    if a and a==b:result['posture']=a
    if result['rest']=='bottom':result['candidate']='deadlift'
    elif result['rest']=='top' and result['posture']=='lying':result['candidate']='benchpress'
    elif result['rest']=='top' and result['posture']=='standing':result['candidate']='squat'
    try:result['classifier']=v40.infer(pose,rows,intervals,checkpoint)
    except Exception as exc:
        result['classifier']={'status':'unavailable','exercise':None,'reason':f'v40 입력 또는 체크포인트 오류: {type(exc).__name__}'}
    if (result['candidate'] is not None and result['classifier'].get('exercise')==result['candidate']
            and result['selection_ambiguous'] is False):
        result['status']='confirmed';result['exercise']=result['candidate']
    return result
