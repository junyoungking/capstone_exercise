"""Representative real frame for every unresolved automatic decision."""
from pathlib import Path
import json,sys
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io.rows import read_rows
from eval.mc2b.evaluate import raw_path
base={r['key']:r for r in json.loads((ROOT/'results/mc/baseline.json').read_text())}
plate={r['key']:r for r in json.loads((ROOT/'eval_out/m5/eval0.json').read_text())}
evaldata=json.loads((ROOT/'results/m5/auto_evaluation.json').read_text());out=ROOT/'eval_out/m5/auto_frames';out.mkdir(parents=True,exist_ok=True);n=0
for a in evaldata:
 if a.get('judgment',{}).get('status')=='confirmed' and a.get('judgment',{}).get('exercise')==a['truth']:continue
 b=base[a['key']];m=plate[a['key']];p=(raw_path(b['result_path'])/'rows.jsonl.gz') if m['start_same'] else ROOT/'eval_out/m5/plate_rows'/b['dataset']/(b['key'].replace('/','__')+'.jsonl.gz')
 rows=read_rows(p);row=rows[len(rows)//2];cap=cv2.VideoCapture(b['video']);cap.set(cv2.CAP_PROP_POS_MSEC,row['t']*1000);ok,im=cap.read();cap.release()
 if not ok:continue
 box=row.get('plate',{}).get('box')
 if box:
  x0,y0,x1,y1=map(int,box);cv2.rectangle(im,(x0,y0),(x1,y1),(0,0,255),max(2,im.shape[1]//400))
 cv2.putText(im,f"{a['key']}  {row['t']:.2f}s",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1,cv2.LINE_AA)
 cv2.imwrite(str(out/(a['key'].replace('/','__')+'.jpg')),im,[int(cv2.IMWRITE_JPEG_QUALITY),80]);n+=1
print(n,'images',out)
