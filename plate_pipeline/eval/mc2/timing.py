"""Single-process CPU benchmark: same 61.5 s input, same six ONNX threads."""
from dataclasses import replace
from pathlib import Path
import json,sys,time
import cv2
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.config import DEFAULT
from server.jobs import process_video

if __name__=='__main__':
 video=Path('/home/kojy1009/Downloads/squat_user_set.mp4');model=Path('/home/kojy1009/capstone_exercise/plate_pipeline/models/plate_tiny.onnx')
 cv2.setNumThreads(1);original=ort.InferenceSession
 def session(*a,**kw):
  options=ort.SessionOptions();options.intra_op_num_threads=6;options.inter_op_num_threads=1
  return original(*a,sess_options=options,**kw)
 ort.InferenceSession=session
 out=[]
 for enabled in (False,True):
  cfg=replace(DEFAULT,track=replace(DEFAULT.track,plate_pick=enabled));start=time.perf_counter()
  result,rows,_=process_video(video,'squat',model=str(model),device='cpu',cfg=cfg)
  out.append({'plate_pick':enabled,'wall_s':time.perf_counter()-start,'frames':len(rows),'recognition_s':result['recog_s'],'completed_reps':sum(v['back'] is not None for s in result['sets'] for v in s['rep_values'])})
  print(out[-1],flush=True)
 dest=ROOT/'results/mc2/timing.json';dest.write_text(json.dumps(out,indent=2))
