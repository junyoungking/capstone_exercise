"""Serial full plate+pose timing of one exact 1800-frame / 30-fps clip."""
from pathlib import Path
import argparse
from dataclasses import replace
import hashlib
import json
import time
from unittest.mock import patch
from eval.baseline import ROOT, timing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--model',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import cv2,onnxruntime as ort
    cv2.setNumThreads(1);original=ort.InferenceSession
    def session(*args,**kwargs):
        options=kwargs.pop('sess_options',None) or ort.SessionOptions()
        options.intra_op_num_threads=a.threads;options.inter_op_num_threads=1
        return original(*args,sess_options=options,**kwargs)
    from core.config import DEFAULT
    from server.jobs import process_video
    from feedback.pose import extract_pose
    from feedback.plate import plate_rep_mask
    from eval.baseline_metrics import measure
    results=[]
    for name,mode,n in [('balanced','balanced',5),('n3','tracked',3),('n5','tracked',5),
                        ('n10','tracked',10),('tiny_m','tiny_m',5),('lightweight','lightweight',5)]:
        dest=a.out/name;dest.mkdir(exist_ok=True)
        start=time.perf_counter()
        with patch.object(ort,'InferenceSession',session),timing() as times:
            result,rows,_=process_video(a.video,'bench',model=a.model,device='cpu',pose=False,out_dir=dest/'plate')
            pose=extract_pose(a.video,dest/'pose.npz','benchpress',no_cache=True,plate_rows=rows,
                              rep_mask=plate_rep_mask(rows,result),pose_cfg=replace(DEFAULT.pose,mode=mode,det_frequency=n))
            processing=time.perf_counter()-start
        assert pose['total']==1800 and abs(pose['fps']-30)<1e-6
        measure(pose,rows,result,a.video,'benchpress')
        elapsed=time.perf_counter()-start
        times.update(pose['inference_timing'])
        times['rest']=processing-sum(times.values())
        entry=dict(name=name,mode=mode,det_frequency=n,threads=a.threads,frames=1800,fps=30,
                   video=str(a.video.resolve()),video_sha256=hashlib.sha256(a.video.read_bytes()).hexdigest(),
                   processing_s=processing,end_to_end_s=elapsed,processing_fps=1800/processing,timing=times)
        results.append(entry)
        (a.out/'benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(entry),flush=True)


if __name__=='__main__':main()
