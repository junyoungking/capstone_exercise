"""Frozen v40 test73 plate evaluation. All generated raw artifacts stay outside results/."""
from pathlib import Path
import sys,json,csv,os,importlib.util
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'plate_pipeline'))
sys.path.insert(0,str(Path(__file__).parent))
from reproduce import reproduce,aggregate


def module(path):
    spec=importlib.util.spec_from_file_location('v40_decode',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def run_one(item):
    record,dataset,model,out,v40=item
    import cv2,onnxruntime as ort,av
    cv2.setNumThreads(1)
    # Execution parallelism only; detector/model/thresholds remain unchanged.
    original=getattr(ort,"_midterm_original_session",ort.InferenceSession)
    ort._midterm_original_session=original
    def session(*args,**kw):
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        return original(*args,sess_options=options,**kw)
    ort.InferenceSession=session
    from server.jobs import process_video
    from core.io.rows import read_rows
    from eval.debug_track import timeline
    key=record['key'];dest=Path(out)/key;dest.mkdir(parents=True,exist_ok=True)
    path=Path(dataset)/key
    if not (dest/'result.json').exists():
        res,rows,_=process_video(path,'bench' if record['type']=='benchpress' else record['type'],model=model,
                                out_dir=dest,device='auto',pose=False)
    else:
        res=json.loads((dest/'result.json').read_text());rows=read_rows(dest/'rows.jsonl.gz')
    # Native unrounded decoded PTS, same frame->timestamp semantics as v40 make_pack.
    with av.open(str(path)) as c:timestamps=[float(f.time) for f in c.decode(video=0) if f.time is not None]
    origin=timestamps[0]
    native_gt=[timestamps[int(rep[2])] for rep in record['reps']]
    gt=[t-origin for t in native_gt]
    native_pred=[rep['back'] for st in res['sets'] for rep in st['rep_values'] if rep['back'] is not None]
    pred=[t-origin for t in native_pred]
    incomplete=sum(rep['back'] is None for st in res['sets'] for rep in st['rep_values'])
    dec=module(Path(v40)/'decode.py');tp,fp,fn,delays=dec.match_events(gt,pred)
    measured=[p for r in rows if (p:=r.get('plate',{})).get('used') and p.get('src') in ('det','det_low','rim')]
    ratios=[(p['box'][2]-p['box'][0])/(p['box'][3]-p['box'][1]) for p in measured if p['box'][3]>p['box'][1]]
    ratio=float(np.median(ratios)) if ratios else None
    view='없음' if ratio is None else '옆' if ratio>=.75 else '비스듬' if ratio>=.4 else '정면'
    tracked=sum(r.get('plate',{}).get('used',False) for r in rows)/len(rows) if rows else 0
    err=len(pred)-len(gt)
    cause='일치' if err==0 else ('원판 측정 없음' if not ratios else '원판 추적 누락/초기화 확인 필요' if tracked<.6 else
            '정면·비스듬 조건에서 신호/가림 확인 필요' if view!='옆' else '검출 대상·세트/rep 분할 확인 필요')
    # Counted incomplete trips are kept in result, but not completion events.
    d={'key':key,'type':record['type'],'gt_count':len(gt),'pred_count':len(pred),'signed_error':err,'abs_error':abs(err),
       'exact':err==0,'obo':abs(err)<=1,'tp':tp,'fp':fp,'fn':fn,'tracked_fraction':tracked,'plate_ratio':ratio,'view':view,
       'counted_trips':sum(st['reps'] for st in res['sets']),'incomplete_counted_trips':incomplete,'cause':cause,'delays':delays}
    res['midterm_evaluation']={**d,'gt_events':gt,'pred_events':pred,'native_gt_events':native_gt,'native_pred_events':native_pred,'clock_origin_s':origin,'event_definition':'back: completed return to resting position'}
    (dest/'result.json').write_text(json.dumps(res,ensure_ascii=False,indent=1))
    timeline(rows,res,dest/'timeline.png',truth=len(gt))
    print(f"{key}: {len(pred)}/{len(gt)} {view} ({tracked:.0%})",flush=True)
    return d


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    p=argparse.ArgumentParser();p.add_argument('--local-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'out/midterm');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();v40=a.local_root/'new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42'
    results=ROOT/'plate_pipeline/results/midterm';test=reproduce(v40,results)
    from cohort import included,clock_gate,summarize_cohort
    clock_gate(v40,a.local_root/'new_data/new_dataset',results)
    test=included(test)
    records={r['key']:r for r in json.loads((v40/'records.json').read_text())}
    for key in test.key:
        if not (a.local_root/'new_data/new_dataset'/key).is_file():raise FileNotFoundError(key)
    items=[(records[key],str(a.local_root/'new_data/new_dataset'),str(a.model),str(a.out),str(v40)) for key in test.key]
    with ProcessPoolExecutor(a.workers) as pool:data=list(pool.map(run_one,items))
    summarize_cohort(v40,results,a.out)

if __name__=='__main__':main()
