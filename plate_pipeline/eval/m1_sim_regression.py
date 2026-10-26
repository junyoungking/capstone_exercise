"""Snapshot deterministic count/tempo/speed core cases for cross-worktree comparison.

With PYTHONPATH pointing at either M0/plate_pipeline or M1/plate_pipeline,
execute this same script by absolute path. Compare the two JSON files bytewise.
Outputs stay under eval_out; tracked simulation result files are not overwritten.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import core.config as core_config
from sim.gen import make_recording
from sim.evaluate import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cases=[]
    for exercise in ('squat','bench','deadlift'):
        for fps in (15,30,60):
            for noise in (.003,.01,.02,.03,.04):
                for seed in (0,1):
                    recording=make_recording(exercise,np.random.default_rng(seed),fps,noise)
                    result=evaluate(recording,speed=True)
                    cases.append(dict(exercise=exercise,fps=fps,noise=noise,seed=seed,
                                      **asdict(result)))
    encoded=json.dumps(cases,sort_keys=True,allow_nan=False)+'\n'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(encoded)
    print(json.dumps(dict(conditions=len(cases),sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                          core_source=str(Path(core_config.__file__).resolve()))))


if __name__=='__main__':main()
