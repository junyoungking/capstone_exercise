"""61.5 s automatic path plus confirmation, excluding human decision time."""
from pathlib import Path
import json,time,sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root));sys.path.insert(0,str(root.parent))
from server.compare import compare
video='/home/kojy1009/Downloads/squat_user_set.mp4';model='/home/kojy1009/capstone_exercise/plate_pipeline/models/plate_tiny.onnx';out=root/'eval_out/m5/timing/auto'
t=time.perf_counter();pending=compare(video,None,'auto',out,threads=6,model=model);first=time.perf_counter()-t
assert pending.get('needs_confirmation')
t=time.perf_counter();result=compare(video,None,'squat',out,threads=6,model=model,resume_auto=True);resume=time.perf_counter()-t
value={'auto_to_confirmation_s':first,'resume_s':resume,'total_compute_s':first+resume,'reps':len(result['user']['reps']),'classifier_status':pending['auto_result']['user']['classifier']['status']}
(root/'results/m5/timing_auto.json').write_text(json.dumps(value,ensure_ascii=False,indent=2));print(json.dumps(value,ensure_ascii=False),flush=True)
