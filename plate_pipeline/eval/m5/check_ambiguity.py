"""Validate the neutral-policy ambiguity flag on an actual multi-person bench video."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
import cv2,onnxruntime as ort
from core.io.rows import read_rows
from core.auto.judge import assess
from feedback.pose import extract_pose
cv2.setNumThreads(1);original=ort.InferenceSession
def session(*args,**kwargs):
 opts=kwargs.pop('sess_options',None) or ort.SessionOptions();opts.intra_op_num_threads=6;opts.inter_op_num_threads=1
 return original(*args,sess_options=opts,**kwargs)
ort.InferenceSession=session
key='benchpress/stu10_11.mp4';base=next(x for x in json.loads((ROOT/'results/mc/baseline.json').read_text()) if x['key']==key)
rows=read_rows(ROOT/'eval_out/m5/plate_rows/midterm/benchpress__stu10_11.mp4.jsonl.gz');out=ROOT/'eval_out/m5/ambiguity_stu10.npz'
pose=extract_pose(Path(base['video']),out,'squat',no_cache=True,plate_rows=rows)
value={'key':key,'frames':len(rows),'ambiguous_frames':int(pose['selection_ambiguous'].sum()),'judgment':assess(rows,pose)}
(ROOT/'results/m5/ambiguity_check.json').write_text(json.dumps(value,ensure_ascii=False,indent=2));print(json.dumps(value,ensure_ascii=False),flush=True)
