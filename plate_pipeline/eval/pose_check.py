"""Inspect pose coverage and foot/plate offsets without affecting rep analysis."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from core.config import DEFAULT
from core.io.rows import read_rows
from core.io import video
from core.recog.pose import POSE_CONNECTIONS
from core.signal.height import height_signal


def floor_intervals(rows, duration=.5, tolerance=.03):
    """Observed stationary intervals lasting >=0.5s with <=3cm height range.

    Heights use the existing 450mm plate scale. Missing plate frames break a
    window; no interpolation or extension through tracking gaps is performed.
    Each output reports the median foot-bottom offset over its pose observations.
    """
    t,y,_ = height_signal(rows)
    if len(t) < 2:
        return []
    use = [r for r in rows if r.get('plate', {}).get('used')]
    all_t = np.array([r['t'] for r in rows])
    diffs = np.diff(all_t)
    positive = diffs[diffs > 0]
    cadence = float(np.median(positive)) if len(positive) else 0.0
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > 1.5*cadence)+1, len(t)]
    intervals = []
    for a,b in zip(cuts[:-1], cuts[1:]):
        i = a
        spans = []
        while i < b:
            j = int(np.searchsorted(t, t[i]+duration-1e-9))
            if j >= b:
                break
            if np.ptp(y[i:j+1]) > tolerance:
                i += 1
                continue
            lo,hi = y[i:j+1].min(), y[i:j+1].max()
            e = j+1
            while e < b and max(hi,y[e])-min(lo,y[e]) <= tolerance:
                lo,hi = min(lo,y[e]),max(hi,y[e])
                e += 1
            spans.append((i,e))
            i = e
        for s,e in spans:
            offsets = []
            for r in use[s:e]:
                p = r['plate']; h = p['box'][3]-p['box'][1]
                if 'person' in r and h > 0:
                    offsets.append((r['person']['box'][3]-p['box'][3])/h)
            intervals.append(dict(t0=round(float(t[s]),4), t1=round(float(t[e-1]),4),
                                  n=len(offsets), median=float(np.median(offsets)) if offsets else None))
    return intervals


def summarize(rows):
    ran = [r for r in rows if 'pose' in r]
    found = sum('person' in r for r in ran)
    bins = []
    if rows:
        start = int(np.floor(rows[0]['t']/2))*2
        for t in range(start, int(rows[-1]['t']//2)*2+1, 2):
            chunk = [r for r in rows if t <= r['t'] < t+2]
            attempted = [r for r in chunk if 'pose' in r]
            n = sum('person' in r for r in attempted)
            bins.append(dict(t0=t, t1=t+2, frames=len(chunk), pose_frames=len(attempted),
                             lifter_frames=n, ratio=n/len(attempted) if attempted else None))
    return dict(frames=len(rows), pose_frames=len(ran),
                candidates=dict(sorted(Counter(r['pose']['n'] for r in ran).items())),
                lifter_frames=found, lifter_ratio=found/len(ran) if ran else None,
                lifter_all_frames_ratio=found/len(rows) if rows else None,
                bins=bins, floor=floor_intervals(rows),
                pose_ms_median=float(np.median([r['pose']['ms'] for r in ran])) if ran else None)


def write_sheet(rows, video_path, out):
    if not rows:
        raise ValueError('Cannot draw a sheet from empty rows')
    # Choose the first pose-run frame at/after each 0.5s tick. Never carry kp
    # into a frame on which pose was skipped (or choose a different person).
    eligible = [i for i,r in enumerate(rows) if 'pose' in r] or list(range(len(rows)))
    times = np.array([rows[i]['t'] for i in eligible])
    selected = set()
    for t in np.arange(rows[0]['t'], rows[-1]['t']+1e-9, .5):
        j = int(np.searchsorted(times, t))
        if j < len(times):
            selected.add(eligible[j])
    by_seq = {rows[i].get('seq',i): rows[i] for i in selected}
    thumbs = []
    for seq, (t,img) in enumerate(video.frames(video_path)):
        if seq not in by_seq:
            continue
        r = by_seq[seq]
        p = r.get('plate')
        if p:
            b = [int(round(z)) for z in p['box']]
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0,255,0) if p.get('used') else (0,0,255), 3)
        kp = r.get('kp')
        if kp:
            for a,b in POSE_CONNECTIONS:
                if min(kp[a][2],kp[b][2]) >= DEFAULT.pose.vis_thr:
                    cv2.line(img, tuple(int(round(z)) for z in kp[a][:2]),
                             tuple(int(round(z)) for z in kp[b][:2]), (255,200,0), 3)
        h,w = img.shape[:2]
        thumb = cv2.resize(img, (320, max(1,round(h*320/w))))
        thumb = cv2.copyMakeBorder(thumb, 26, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(thumb, f"t={t:.2f} lifter={'yes' if kp else 'no'} side={r.get('side','-')}",
                    (4,18), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
        thumbs.append(thumb)
    if len(thumbs) != len(selected):
        raise ValueError('Video frames do not match rows: missing selected frames')
    cols = min(4,len(thumbs))
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    sheet = np.vstack([np.hstack(thumbs[i:i+cols]) for i in range(0,len(thumbs),cols)])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), sheet):
        raise OSError(f'Cannot write sheet: {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rows')
    ap.add_argument('--video')
    ap.add_argument('--sheet', metavar='OUT.jpg')
    args = ap.parse_args()
    if args.sheet and not args.video:
        ap.error('--sheet requires --video')
    rows = read_rows(args.rows)
    # JSON output includes the entire 2-second coverage and stationary interval lists.
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    if args.sheet:
        write_sheet(rows, args.video, args.sheet)
        print(f'Sheet: {args.sheet}')


if __name__ == '__main__':
    main()
