"""Paired synthetic rep counts, speed error and endpoint times with recovery off/on."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import json,sys,zlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.config import DEFAULT
from core.judge.sets import counted
from core.measure.times import all_rep_values
from sim.gen import make_recording
from sim.evaluate import run_final,time_match
from eval.mc.baseline import matching

def job(arg):
 ex,fps,noise,seed=arg;rng=np.random.default_rng(seed);rec=make_recording(ex,rng,fps,noise);out={}
 for enabled in (False,True):
  cfg=replace(DEFAULT,speed=replace(DEFAULT.speed,recover_early_back=enabled));fr,trips=run_final(rec,cfg);reps=counted(trips);vals=all_rep_values(fr.sig,fr.raw_t,reps,rec.rest,cfg.speed)
  gt=[r.t1 for r in rec.reps if r.t1 is not None];pred=[v.back for v in vals if v.back is not None]
  pairs,fn,fp=matching(gt,pred);errors=[v.v_mean-r.v_true for v,r in zip(vals,rec.reps) if v.v_mean is not None and r.v_true and len(vals)==len(rec.reps)]
  out['on' if enabled else 'off']={'n':len(reps),'exact':len(reps)==len(rec.reps),'far_ok':time_match(reps,rec.reps),'tp':len(pairs),'fp':len(fp),'fn':len(fn),'errors':errors,'available_speeds':sum(v.v_mean is not None for v in vals)}
 return {'exercise':ex,'fps':fps,'noise_m':noise,'seed':seed,**out}

if __name__=='__main__':
 jobs=[(ex,fps,noise,zlib.crc32(f'{ex}|{fps}|{noise}|{i}'.encode())) for ex in ('squat','bench','deadlift') for fps in (15,30,60) for noise in (.003,.01,.02,.03,.04) for i in range(8)]
 with ProcessPoolExecutor(max_workers=4) as pool:data=list(pool.map(job,jobs))
 summary={}
 for mode in ('off','on'):
  es=np.array([e for d in data for e in d[mode]['errors']]);tp=sum(d[mode]['tp'] for d in data);fp=sum(d[mode]['fp'] for d in data);fn=sum(d[mode]['fn'] for d in data)
  summary[mode]={'sets':len(data),'exact':sum(d[mode]['exact'] for d in data),'far_ok':sum(d[mode]['far_ok'] for d in data),'tp':tp,'fp':fp,'fn':fn,'speed_n':len(es),'speed_rmse':float(np.sqrt(np.mean(es**2))),'speed_mae':float(np.mean(abs(es))),'available_speeds':sum(d[mode]['available_speeds'] for d in data)}
 assert all(d['off']['n']==d['on']['n'] and d['off']['far_ok']==d['on']['far_ok'] for d in data)
 p=ROOT/'results/mc/sim.json';p.write_text(json.dumps({'summary':summary,'conditions':data},indent=2))
 print(summary,flush=True)
