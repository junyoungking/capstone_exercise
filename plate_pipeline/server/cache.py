"""Content-addressed reference observations, independent of viewer height/load."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from core.config import DEFAULT


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def cache_key(video,exercise,plate_mm,model,checkpoint):
    root=Path(__file__).resolve().parents[2]
    sources=list((root/'plate_pipeline/core').rglob('*.py'))
    sources += [root/p for p in ('feedback/pose.py','feedback/plate.py','tools/check_video.py','plate_pipeline/server/jobs.py','plate_pipeline/server/compare.py')]
    models=[Path(model),Path(checkpoint)]+sorted((Path.home()/'.cache/rtmlib/hub/checkpoints').glob('*.onnx'))
    config=dict(version=1,exercise=exercise,plate_mm=plate_mm,config=asdict(DEFAULT),
                models={str(p):file_hash(p) for p in models},sources={str(p.relative_to(root)):file_hash(p) for p in sources})
    settings_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    sha=file_hash(video)
    return hashlib.sha256((sha+settings_hash).encode()).hexdigest(),dict(video_sha256=sha,settings_sha256=settings_hash)


def available(path):
    return path is not None and all((path/name).is_file() for name in ('complete.json','ref.mp4','plate/result.json','plate/rows.jsonl.gz','pose.npz','native.npy'))


def publish(dest,analysis,video,provenance):
    """Publish only complete observations; interrupted writes cannot be hits."""
    tmp=dest.with_name(dest.name+'.pending')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copytree(analysis/'plate',tmp/'plate')
    for name in ('pose.npz','native.npy'): shutil.copy2(analysis/name,tmp/name)
    shutil.copy2(video,tmp/'ref.mp4')
    (tmp/'complete.json').write_text(json.dumps(provenance,sort_keys=True))
    if dest.exists(): shutil.rmtree(dest)
    tmp.rename(dest)
