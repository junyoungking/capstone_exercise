"""Show earlier Mc2 choice alongside the Mc2b final choice."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io.rows import read_rows
from eval.mc2b.evaluate import raw_path
from eval.mc2b.images import panel

if __name__=='__main__':
 base={r['key']:r for r in json.loads((ROOT/'results/mc/baseline.json').read_text())}
 new={r['key']:r for r in json.loads((ROOT/'results/mc2b/evaluation.json').read_text())}
 prev=Path('/home/kojy1009/work/mc-count/plate_pipeline/eval_out/mc2/confirmed/rows');out=ROOT/'results/mc2b/images';out.mkdir(parents=True,exist_ok=True)
 for key,name in [('deadlift/GB_S07_R12.mp4','edge_compare.jpg'),('squat/stu6_11_2.mp4','stored_plate.jpg'),('benchpress/stu10_11.mp4','bench_regression.jpg')]:
  item=base[key];filename=key.replace('/','__')+'.jsonl.gz';oldpath=prev/item['dataset']/filename
  old=read_rows(oldpath) if oldpath.exists() else read_rows(raw_path(item['result_path'])/'rows.jsonl.gz')
  newpath=ROOT/'eval_out/mc2b/rows'/item['dataset']/filename
  now=read_rows(newpath) if new[key]['selection_changed'] else read_rows(raw_path(item['result_path'])/'rows.jsonl.gz')
  cap=cv2.VideoCapture(item['video']);a=panel(cap,old,'Mc2');b=panel(cap,now,'Mc2b');cap.release()
  cv2.imwrite(str(out/name),np.hstack([a,b]),[int(cv2.IMWRITE_JPEG_QUALITY),88])
