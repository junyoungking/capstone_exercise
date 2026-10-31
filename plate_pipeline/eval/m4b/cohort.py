"""Replay M2's fixed 69-video rows and selected poses; no inference."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io.rows import read_rows
from core.geometry import plate_geometry,standing_frames,stature,box_angle_reliable,implied_plate_diameter,plate_suggestion
from server.jobs import analyse
M2=Path('/home/kojy1009/work/m2-3d/plate_pipeline/eval_out/m2/pose3d/summary.json')
BASE=Path('/home/kojy1009/work/m0-baseline/plate_pipeline/eval_out/baseline_mp/raw')
OUT=ROOT/'results/m4b'

def group(item):
 key=item['key'];name=Path(key).name
 if key.startswith('team9/'):return 'team9'
 if item['exercise']=='benchpress':return 'benchpress'
 if item['exercise']=='squat':return 'squat'
 if name.startswith(('GB_','LLK_','PB_')):return 'GB_LLK_PB'
 return 'other_deadlift'

def run(item):
 key=item['key'];source=Path(item['source']);source=source if source.is_absolute() else Path('/home/kojy1009/work/m2-3d/plate_pipeline')/source
 pose_path=source/'raw'/key/'pose.npz';base=BASE/key;rows=read_rows(base/'plate/rows.jsonl.gz');res=json.loads((base/'plate/result.json').read_text())
 with np.load(pose_path) as f:
  raw=f['raw_lms'].copy();w=int(f['w']);h=int(f['h']);raw[:,:,:2]*=[w,h]
 boxes=np.array([r['plate']['box'] if r.get('plate',{}).get('used') else [np.nan]*4 for r in rows]);stand=standing_frames(raw);reliable=box_angle_reliable(boxes,(w,h));g=plate_geometry(boxes,(w,h));pre,_=stature(raw,g,stand)
 d={str(H):implied_plate_diameter(raw,boxes,(w,h),stand,H,reliable) for H in (160,170,180)}
 implied=d['170']; suggested=plate_suggestion(implied,450);chosen=round(implied/10)*10 if suggested else 450
 stats={};t0=time.perf_counter()
 for role,mm in [('before',450),('after',chosen)]:
  sets,_=analyse(rows,res['exercise'],res['fps'],mm);vals=[v for st in sets for v in st['rep_values']]
  stats[role]={'count':sum(v['back'] is not None for v in vals),'events':[v['back'] for v in vals if v['back'] is not None], 'rom_cm':[v['rom_m']*100 for v in vals if v['rom_m'] is not None], 'velocity':[v['v_mean'] for v in vals if v['v_mean'] is not None]}
 after,_=stature(raw,plate_geometry(boxes,(w,h),chosen),stand)
 return {'key':key,'exercise':item['exercise'],'group':group(item),'standing_frames':int(stand.sum()),'box_ratio':g['ratio'],'angle_reliable':reliable,'assumed_cos_one':not reliable,'implied_mm':implied,'range_mm':[d['160'],d['180']],'by_height':d,'cos_one_mm':implied_plate_diameter(raw,boxes,(w,h),stand,170,False),
          'cos_box_ratio_mm':implied_plate_diameter(raw,boxes,(w,h),stand,170,True),'suggested':suggested,'chosen_mm':chosen,'stature_before_cm':None if pre is None else 100*pre,'stature_after_cm':None if after is None else 100*after,'stats':stats,'eval_s':time.perf_counter()-t0}

if __name__=='__main__':
 entries=json.loads(M2.read_text())['videos'];result=[run(x) for x in entries];OUT.mkdir(parents=True,exist_ok=True);(OUT/'cohort.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 for grp in ('GB_LLK_PB','other_deadlift','squat','team9','benchpress'):
  a=[x for x in result if x['group']==grp];v=[x['implied_mm'] for x in a if x['implied_mm'] is not None];print(grp,len(a),'est',len(v),'median',round(float(np.median(v)),1) if v else None,'suggested',sum(x['suggested'] for x in a), flush=True)
 for threshold in (.2,.25,.3,.35,.4,.45,.5):
  print(threshold,[(grp,sum(plate_suggestion(x['implied_mm'],450,threshold) for x in result if x['group']==grp)) for grp in ('GB_LLK_PB','other_deadlift','squat','team9','benchpress')])
