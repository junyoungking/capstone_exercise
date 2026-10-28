"""Audit reference eligibility and plate payload equality without rerunning inference."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np
from eval.baseline import ROOT


def read(path):return json.loads(Path(path).read_text())


def eligibility(source):
    from feedback.metrics import build_metrics
    from feedback.set_judge import _eligible
    entries=[]
    for video in read(source/'baseline.json')['videos']:
        if video['exercise'] not in ('benchpress','deadlift'):continue
        raw=source/'raw'/video['key']
        with np.load(raw/'pose.npz',allow_pickle=False) as f:pose={k:f[k] for k in f.files}
        segments=read(raw/'segments.json')
        metrics=[build_metrics(dict(pose,has_fingers=value),segments) for value in (True,False)]
        entries.append(dict(key=video['key'],exercise=video['exercise'],
                            before=sorted(_eligible(metrics[0])),after=sorted(_eligible(metrics[1]))))
    return entries


def invariance(base,candidate):
    a={v['key']:v for v in read(base/'baseline.json')['videos']}
    b={v['key']:v for v in read(candidate/'baseline.json')['videos']}
    evidence=[]
    for key,v in b.items():
        old=a[key]
        checks=dict(counts=old['counts']==v['counts'],frames=old['image_frames']==v['image_frames'])
        expected=hashlib.sha256(Path(old['pose_video']).read_bytes()).hexdigest()
        actual=hashlib.sha256(Path(v['pose_video']).read_bytes()).hexdigest()
        checks['prepared_video_sha256']=expected==actual
        checks['recorded_prepared_sha256']=('prepared_input' not in v or v['prepared_input']['sha256']==actual)
        for field in ('video_sha256','plate_model_sha256'):
            checks[field]=old['fingerprint'][field]==v['fingerprint'][field]
        hashes={}
        for stage in ('plate','pose_plate'):
            x,y=base/'raw'/key/stage,candidate/'raw'/key/stage
            if not x.exists() and not y.exists():continue
            if not x.exists() or not y.exists():checks[stage]=False;continue
            for name in ('rows.jsonl.gz','result.json'):
                if name.endswith('.gz'):
                    left,right=[gzip.decompress((p/name).read_bytes()) for p in (x,y)]
                else:
                    left,right=[read(p/name) for p in (x,y)]
                    for obj in (left,right):
                        for field in ('recog_s','recog_fps'):obj.pop(field,None)
                    left,right=[json.dumps(v,ensure_ascii=False).encode() for v in (left,right)]
                label=stage+'/'+name;checks[label]=left==right
                hashes[label]=[hashlib.sha256(v).hexdigest() for v in (left,right)]
        evidence.append(dict(key=key,checks=checks,hashes=hashes))
    return dict(videos=len(b),count_videos=sum(v['counts'] is not None for v in b.values()),
                exact=all(all(v['checks'].values()) for v in evidence),evidence=evidence)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source',type=Path)
    source.add_argument('--m1-root',type=Path,help='Requalify both M1 modes on identical saved poses')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base',type=Path)
    a=p.parse_args()
    if a.m1_root:
        if a.base:p.error('--base requires --source')
        result={mode:eligibility(a.m1_root/mode) for mode in ('lightweight','balanced')}
    else:
        result=invariance(a.base,a.source) if a.base else eligibility(a.source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if a.base and not result['exact']:raise SystemExit('Plate result changed')


if __name__=='__main__':main()
