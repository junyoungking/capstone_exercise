"""Read-only plate diagnostics: no selection, tracking or rep rules are changed."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.io.rows import read_rows
from core.io import video
from core.recog.pose import near_wrist, POSE_CONNECTIONS
from core.signal.height import height_signal
from core.signal.final import clean_final
from core.judge.session import run_session

COLORS = {'det': '#218a48', 'klt': '#377eb8', 'rim': '#e08214', 'lost': '#888888'}
BGR = {'det': (50,190,50), 'klt': (230,140,50), 'rim': (20,160,240), 'lost': (140,140,140)}


def iou(a,b):
    x0,y0 = max(a[0],b[0]),max(a[1],b[1])
    x1,y1 = min(a[2],b[2]),min(a[3],b[3])
    intersection = max(0,x1-x0)*max(0,y1-y0)
    union = max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.0


def moving_candidates(rows):
    """Greedy one-to-one IoU>=.3 links, <=5 missing decoded frames.

    Candidates overlapping the current tracked box at IoU>=.3 are excluded,
    including a lost box (which is still the current target hypothesis). These
    are diagnostic tracklets, not verified identities or a replacement tracker.
    """
    tracks = []
    for index,r in enumerate(rows):
        seq = r.get('seq',index)
        plate = r.get('plate',{}).get('box')
        cands = [d for d in r.get('dets',[]) if plate is None or iou(d,plate) < .3]
        pairs = []
        for k,tr in enumerate(tracks):
            if 0 < seq-tr['seq'] <= 6:
                for j,d in enumerate(cands):
                    overlap = iou(tr['box'],d)
                    if overlap >= .3:
                        pairs.append((-overlap,k,j))
        used_tracks, used_dets = set(),set()
        for _,k,j in sorted(pairs):
            if k in used_tracks or j in used_dets:
                continue
            tr = tracks[k]; d = cands[j]
            tr.update(seq=seq,box=d[:4],t1=r['t'])
            tr['ys'].append((d[1]+d[3])/2); tr['hs'].append(d[3]-d[1])
            used_tracks.add(k); used_dets.add(j)
        for j,d in enumerate(cands):
            if j not in used_dets:
                tracks.append(dict(seq=seq,box=d[:4],t0=r['t'],t1=r['t'],
                                   ys=[(d[1]+d[3])/2],hs=[d[3]-d[1]]))
    moving = []
    for tr in tracks:
        height = float(np.median(tr['hs']))
        if tr['t1']-tr['t0'] >= .5-1e-9 and height > 0 and np.ptp(tr['ys']) > .3*height:
            moving.append(dict(t0=tr['t0'],t1=tr['t1'],observations=len(tr['ys']),
                               motion_over_height=float(np.ptp(tr['ys'])/height)))
    return dict(n=len(moving), intervals=[[r['t0'],r['t1']] for r in moving], tracks=moving)


def _used(rows):
    return [r for r in rows if r.get('plate',{}).get('used')
            and r['plate']['box'][3] > r['plate']['box'][1]]


def _center(p):
    b = p['box']
    return np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])


def _height(p):
    return p['box'][3]-p['box'][1]


def wrist_pairs(rows,frame_h,frame_w=None):
    pairs=[]
    for r in _used(rows):
        wrist=near_wrist(r.get('kp'),r['plate']['box'],frame_h=frame_h,frame_w=frame_w)
        if wrist is not None:
            pairs.append([r['t'],float(_center(r['plate'])[1]),wrist[1],_height(r['plate'])])
    return np.asarray(pairs,float).reshape(-1,4)


def _correlation(pairs):
    if len(pairs) < 2 or np.std(pairs[:,1]) < 1e-9 or np.std(pairs[:,2]) < 1e-9:
        return None
    return float(np.corrcoef(pairs[:,1],pairs[:,2])[0,1])


def diagnose(rows, result=None, frame_h=None):
    result=result or {}
    if frame_h is None:
        frame_h=result.get('height')
    use=_used(rows)
    heights=np.array([_height(r['plate']) for r in use])
    centers=np.array([_center(r['plate'])[1] for r in use])
    y_range=float(np.diff(np.percentile(centers,[5,95]))[0]/np.median(heights)) if use else None
    jumps=[]
    for a,b in zip(use[:-1],use[1:]):
        pa,pb=a['plate'],b['plate']
        scale=_height(pb)/_height(pa)
        shift=float(np.linalg.norm(_center(pb)-_center(pa))/_height(pa))
        if abs(scale-1) > .25 or shift > .3:
            jumps.append(dict(t=b['t'],prev_t=a['t'],height_ratio=scale,center_over_height=shift))
    sources=Counter(r.get('plate',{}).get('src','none') for r in rows)
    pairs=wrist_pairs(rows,frame_h,result.get('width'))
    windows=[]
    if rows:
        for t in range(int(rows[0]['t']//2)*2,int(rows[-1]['t']//2)*2+1,2):
            p=pairs[(pairs[:,0]>=t)&(pairs[:,0]<t+2)]
            eligible=len(p)>=10 and np.ptp(p[:,1]) > .1*np.median(p[:,3])
            windows.append(dict(t0=t,t1=t+2,pairs=len(p),r=_correlation(p) if eligible else None))
    attempted=sum('pose' in r or 'kp' in r for r in use)
    return dict(used_ratio=len(use)/len(rows) if rows else 0.0,
                src_share={k:v/len(rows) for k,v in sorted(sources.items())} if rows else {},
                y_range=y_range,jumps=jumps,moving_cands=moving_candidates(rows),
                wrist_r=_correlation(pairs),wrist_windows=windows,
                wrist_seen=len(pairs)/attempted if attempted else None,
                wrist_pairs=len(pairs),wrist_pose_plate_frames=attempted)


def _signals(rows,result):
    t,y,x=height_signal(rows,result.get('plate_mm',450.0))
    fps=result.get('fps') or ((len(rows)-1)/(rows[-1]['t']-rows[0]['t']) if len(rows)>1 else 30)
    final=clean_final(t,y,x,fps)
    return t,y,final.sig


def rep_intervals(result):
    reps=[]
    for s in result.get('sets',[]):
        for r in s.get('rep_values',[]):
            if r.get('start') is not None and r.get('back') is not None:
                reps.append((s['set'],r['n'],r['start'],r['back']))
    return reps


def count_times(rows,result):
    """Actual FSM count times, including reps whose speed endpoints are blank.

    A missing rep_values.back must not hide a counted rep in the video overlay.
    Replaying the unchanged final session is read-only and uses the result FPS.
    """
    if result.get('exercise') not in ('squat','bench','deadlift'):
        return sorted(back for _,_,_,back in rep_intervals(result))
    _,_,s=_signals(rows,result)
    sessions=run_session(s.t,s.y,s.x,s.gap_before,result['exercise'])
    return sorted(r.end_t for st in sessions for r in st.reps)


def timeline(rows,result,png,truth=None,stats=None):
    stats=stats if stats is not None else diagnose(rows,result)
    t=np.array([r['t'] for r in rows])
    fig,axes=plt.subplots(4,1,figsize=(15,12),sharex=True)
    for src,color in COLORS.items():
        for ax,field in zip(axes[:2],['center','height']):
            values=[(float(_center(r['plate'])[1]) if field=='center' else _height(r['plate']))
                    if r.get('plate',{}).get('src')==src else np.nan for r in rows]
            ax.plot(t,values,color=color,label=src,lw=1.5,marker='.',markersize=2)
    candidates=[(r['t'],d) for r in rows for d in r.get('dets',[])]
    if candidates:
        times=np.array([a for a,_ in candidates]);ds=np.array([d for _,d in candidates])
        colors=np.tile([.3,.3,.3,1.0],(len(ds),1));colors[:,3]=.1+.8*np.clip(ds[:,4],0,1)
        axes[0].scatter(times,(ds[:,1]+ds[:,3])/2,s=5,c=colors,label='candidates')
        axes[1].scatter(times,ds[:,3]-ds[:,1],s=5,c=colors,label='candidates')
    axes[0].set_ylabel('center y (px)'); axes[0].invert_yaxis()
    axes[1].set_ylabel('box height (px)')
    rt,ry,signal=_signals(rows,result)
    axes[2].plot(rt,ry*100,label='raw',alpha=.65,marker='.',ms=2)
    axes[2].plot(signal.t,signal.y*100,label='clean_final')
    for set_id,n,start,back in rep_intervals(result):
        axes[2].axvspan(start,back,color='#8dd3c7',alpha=.22)
        axes[2].text((start+back)/2,.97,f'{set_id}:{n}',transform=axes[2].get_xaxis_transform(),ha='center',va='top',fontsize=8)
    for st in result.get('sets',[]):
        for rep in st.get('rep_values',[]):
            if rep.get('start') is not None and rep.get('back') is None:
                axes[2].text(rep['start'],.97,f"{st['set']}:{rep['n']}*",
                             transform=axes[2].get_xaxis_transform(),ha='left',va='top',fontsize=8)
    # Actual session open/close times, not the speed endpoints of its first/last rep.
    if result.get('exercise') in ('squat','bench','deadlift'):
        sessions=run_session(signal.t,signal.y,signal.x,signal.gap_before,result['exercise'])
        boundaries=[v for st in sessions for v in (st.open_t,st.close_t) if v is not None]
    else:
        boundaries=[s[k] for s in result.get('sets',[]) for k in ('t0','t1') if s.get(k) is not None]
    for boundary in boundaries:
        axes[2].axvline(boundary,color='black',ls='--',alpha=.35)
    axes[2].set_ylabel('height (cm)')
    pairs=wrist_pairs(rows,result.get('height'),result.get('width'))
    if len(pairs):
        axes[3].plot(pairs[:,0],pairs[:,1],'.-',ms=2,label='plate center y')
        axes[3].plot(pairs[:,0],pairs[:,2],'.-',ms=2,label='near wrist y')
    else:
        axes[3].text(.02,.5,'No paired visible wrist observations',transform=axes[3].transAxes)
    axes[3].set_ylabel('y (px)');axes[3].invert_yaxis();axes[3].set_xlabel('time (s)')
    for ax in axes:
        handles,_=ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right',ncol=5,fontsize=8)
        ax.grid(alpha=.15)
    predicted=sum(s['reps'] for s in result.get('sets',[]))
    shares=' '.join(f'{k}:{v:.1%}' for k,v in stats['src_share'].items())
    fig.suptitle(f"{result.get('video','unknown')} | {result.get('exercise','unknown')} | truth={truth} predicted={predicted}\nused={stats['used_ratio']:.1%} src={shares}")
    fig.tight_layout(rect=(0,0,1,.95))
    Path(png).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(png,dpi=120);plt.close(fig)


def debug_video(rows,result,path,out):
    info=video.info(path)
    height=max(2,int(round(info.height*540/info.width/2))*2)
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),info.fps,(540,height))
    if not writer.isOpened():
        raise OSError(f'Cannot write {out}')
    by_seq={r.get('seq',i):r for i,r in enumerate(rows)}
    back_times=count_times(rows,result)
    try:
        for seq,(t,img) in enumerate(video.frames(path)):
            row=by_seq.get(seq,{})
            sx,sy=540/img.shape[1],height/img.shape[0]
            img=cv2.resize(img,(540,height))
            def box(b,color,thickness):
                a=(int(b[0]*sx),int(b[1]*sy));z=(int(b[2]*sx),int(b[3]*sy))
                cv2.rectangle(img,a,z,color,thickness)
                return a
            for j,d in enumerate(row.get('dets',[])):
                a=box(d,(180,180,180),1)
                cv2.putText(img,f"{j}:{d[4]:.2f}",a,0,.38,(220,220,220),1)
            p=row.get('plate')
            src=p.get('src','none') if p else 'none'
            if p:
                box(p['box'],BGR.get(src,(255,255,255)),3)
            kp=row.get('kp')
            if kp:
                for a,b in POSE_CONNECTIONS:
                    if min(kp[a][2],kp[b][2]) >= .3:
                        cv2.line(img,(int(kp[a][0]*sx),int(kp[a][1]*sy)),
                                 (int(kp[b][0]*sx),int(kp[b][1]*sy)),(255,220,0),2)
            count=int(np.searchsorted(back_times,t,side='right'))
            text=f't={t:.2f} count={count} src={src}'
            cv2.putText(img,text,(6,22),0,.52,(0,0,0),3)
            cv2.putText(img,text,(6,22),0,.52,(255,255,255),1)
            writer.write(img)
    finally:
        writer.release()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rows');ap.add_argument('--result');ap.add_argument('--png',required=True)
    ap.add_argument('--video');ap.add_argument('--mp4');ap.add_argument('--truth',type=int)
    a=ap.parse_args()
    if a.mp4 and not a.video:
        ap.error('--mp4 requires --video')
    result=json.loads(Path(a.result).read_text()) if a.result else {}
    if a.video:
        info=video.info(a.video)
        result.setdefault('height',info.height);result.setdefault('width',info.width)
        result.setdefault('video',Path(a.video).name)
    rows=read_rows(a.rows)
    stats=diagnose(rows,result)
    timeline(rows,result,a.png,a.truth,stats)
    if a.mp4:
        Path(a.mp4).parent.mkdir(parents=True,exist_ok=True)
        debug_video(rows,result,a.video,a.mp4)
    print(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
