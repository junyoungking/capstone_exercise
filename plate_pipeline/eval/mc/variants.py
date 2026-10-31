"""Frozen observations, feature off/on; no detector replay required."""
from dataclasses import replace
from pathlib import Path
import gzip,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.config import DEFAULT
from server.jobs import analyse
from eval.mc.baseline import matching,read_analysis
OUT=ROOT/'results/mc'

def score(row,sets):
 pred=[z['back']-row['origin'] for s in sets for z in s['rep_values'] if z['back'] is not None]
 if row['gt'] is None:return {'pred':pred,'tp':None,'fp':None,'fn':None}
 pairs,fn,fp=matching(row['gt'],pred)
 return {'pred':pred,'tp':len(pairs),'fp':[pred[i] for i in fp],'fn':[row['gt'][i] for i in fn]}

def main():
 rows=json.loads((OUT/'baseline.json').read_text());out=[]
 for row in rows:
  res,records=read_analysis(Path(row['result_path']));vals={}
  for on in (False,True):
   cfg=replace(DEFAULT,speed=replace(DEFAULT.speed,recover_early_back=on))
   sets,sig=analyse(records,res['exercise'],res['fps'],res['plate_mm'],cfg=cfg)
   if not on:
    assert sets==res['sets'],row['key']
    assert json.dumps(sets,ensure_ascii=False,separators=(',',':')).encode()==json.dumps(res['sets'],ensure_ascii=False,separators=(',',':')).encode(),row['key']
   vals['on' if on else 'off']=score(row,sets)
  out.append({'key':row['key'],'dataset':row['dataset'],'exercise':row['exercise'],'scope':row['scope'],
              'gt':row['gt'],'gt_count':row['gt_count'],'off':vals['off'],'on':vals['on'],
              'off_sets_bytes_equal':True,'rows_sha256':hashlib.sha256(gzip.decompress((Path(row['result_path'])/'rows.jsonl.gz').read_bytes())).hexdigest()})
  if vals['off']['pred']!=vals['on']['pred']:print(row['key'],vals['off']['pred'],'=>',vals['on']['pred'],flush=True)
 (OUT/'variants.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 for dataset in ('midterm','team9','user_clip','edited'):
  d=[x for x in out if x['dataset']==dataset and x['gt'] is not None]
  if not d:continue
  for mode in ('off','on'):
   tp=sum(x[mode]['tp'] for x in d);fp=sum(len(x[mode]['fp']) for x in d);fn=sum(len(x[mode]['fn']) for x in d)
   print(dataset,mode,'TP/FP/FN',tp,fp,fn,'precision',tp/(tp+fp) if tp+fp else None,'recall',tp/(tp+fn) if tp+fn else None,flush=True)
if __name__=='__main__':main()
