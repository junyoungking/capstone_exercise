"""Standalone M2 ORT6 + torch6 complete processing of the exact M1b 60s clip."""
from pathlib import Path
import argparse,time,json
from unittest.mock import patch
import numpy as np
from eval.baseline import write,sha,timing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import onnxruntime as ort,cv2
    cv2.setNumThreads(1);original=ort.InferenceSession
    def session(*args,**kwargs):
        opts=kwargs.pop('sess_options',None) or ort.SessionOptions();opts.intra_op_num_threads=6;opts.inter_op_num_threads=1
        return original(*args,sess_options=opts,**kwargs)
    from server.jobs import process_video
    from feedback.pose import extract_pose
    from feedback.plate import plate_rep_mask,build_plate_segments
    from feedback.pose3d import process
    from core.geometry.pose3d import load_model
    from core.config import DEFAULT
    from eval.baseline_metrics import measure
    started=time.perf_counter()
    with patch.object(ort,'InferenceSession',session),timing() as times:
        result,rows,_=process_video(a.video,'bench',model=a.model,device='cpu',pose=False,out_dir=a.out/'plate')
        pose=extract_pose(a.video,a.out/'pose.npz','benchpress',no_cache=True,plate_rows=rows,
                          rep_mask=plate_rep_mask(rows,result),pose_cfg=DEFAULT.pose)
        two_d=time.perf_counter()-started
    assert pose['total']==1800 and pose['fps']==30
    model=load_model(a.checkpoint,threads=6)
    segments=build_plate_segments(pose,dict(video=str(a.video),view='side'),'benchpress',rows,result)
    payload,_,_=process(pose,rows,result,segments,model)
    processing=time.perf_counter()-started
    measure(pose,rows,result,a.video,'benchpress')
    elapsed=time.perf_counter()-started
    times.update(pose['inference_timing'])
    entry=dict(name='M2 tracked N10 guard off + 3D',video=str(a.video),video_sha256=sha(a.video),
               frames=1800,fps=30,ort_threads=6,torch_threads=6,pose_config=vars(DEFAULT.pose),
               two_d_s=two_d,three_d=payload['timing'],processing_s=processing,end_to_end_s=elapsed,
               processing_fps=1800/processing,stage_timing=times,checkpoint_sha256=sha(a.checkpoint))
    write(a.out/'benchmark.json',entry);write(a.out/'result3d.json',payload)
    print(json.dumps(entry),flush=True)


if __name__=='__main__':main()
