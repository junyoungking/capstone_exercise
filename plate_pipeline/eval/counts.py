"""Rep counts on your own sets (구현 설계 tab, 3단계 완료 기준: 20 sets right; later 57 of 60).

python -m eval.counts --sets sets.csv --model models/plate_tiny.onnx --out eval_out/counts

sets.csv columns: path,exercise,reps[,plate_mm][,box]
  reps = the true count (counted by eye; partial / failed attempts not counted)
  box  = "x0 y0 x1 y1" of the plate in the first frame, only needed without --model
Each set is processed like an upload; a set counts as right when the recording gives exactly one set with
that many reps. Writes counts.csv, one report folder per video and prints the share right and missed /
extra reps per 100.
"""
import argparse
import csv
import json
from pathlib import Path

from server.jobs import process_video
from eval.debug_track import diagnose, timeline, debug_video


def score(rows):
    right = sum(r["right"] for r in rows)
    true = sum(r["reps"] for r in rows)
    miss = sum(max(0, r["reps"] - r["got"]) for r in rows)
    extra = sum(max(0, r["got"] - r["reps"]) for r in rows)
    return dict(sets=len(rows), right=right, share=right / max(len(rows), 1),
                missed_per_100=100 * miss / max(true, 1), extra_per_100=100 * extra / max(true, 1),
                exact=sum(r['got']==r['reps'] for r in rows),
                mae=sum(abs(r['got']-r['reps']) for r in rows)/max(len(rows),1),
                true_sum=true, got_sum=sum(r['got'] for r in rows))


def resolve_video(path, root=None):
    """Prefix relative paths with root. Bare filenames may resolve uniquely below it.

    Team files live in exercise subdirectories; no exercise-to-directory aliases
    are guessed. Ambiguous basenames must be made explicit in the CSV.
    """
    relative=Path(path)
    candidate=(Path(root)/relative if root is not None and not relative.is_absolute() else relative)
    if candidate.is_file():
        return candidate
    if root is not None and relative.name==str(relative):
        hits=sorted(p for p in Path(root).rglob(relative.name) if p.is_file())
        if len(hits)==1:
            return hits[0]
        if len(hits)>1:
            raise ValueError(f'Ambiguous video basename {path}: {hits}')
    raise FileNotFoundError(candidate)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--pose", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", choices=("auto","cuda","cpu"), default="auto")
    ap.add_argument("--out", default="eval_out/counts")
    a = ap.parse_args(argv)
    out=Path(a.out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {out}')
    with open(a.sets,newline='') as f:
        sets=list(csv.DictReader(f))
    if not sets:
        raise ValueError('Empty evaluation list')
    paths=[resolve_video(s['path'],a.root) for s in sets]
    if len({p.stem for p in paths})!=len(paths):
        raise ValueError('Video stems must be unique within an evaluation')
    out.mkdir(parents=True,exist_ok=False)
    rows = []
    for s,path in zip(sets,paths):
        box = tuple(float(v) for v in s["box"].split()) if s.get("box") else None
        name = path.stem
        dest=out/name
        res, frames, _ = process_video(str(path), s["exercise"], float(s.get("plate_mm") or 450), init_box=box,
                                       model=a.model, out_dir=str(dest),pose=a.pose,device=a.device)
        got_sets = [x["reps"] for x in res["sets"]]
        got = sum(got_sets)
        r = dict(video=name, exercise=s["exercise"], reps=int(s["reps"]), got=got, sets=len(got_sets),
                 right=len(got_sets) == 1 and got == int(s["reps"]),
                 n_sets=len(got_sets),err=got-int(s['reps']),exact=got==int(s['reps']))
        stats=diagnose(frames,res)
        r.update(used_ratio=stats['used_ratio'],y_range=stats['y_range'],jumps=len(stats['jumps']),
                 moving_cands=stats['moving_cands']['n'],wrist_r=stats['wrist_r'],fps=res['recog_fps'],
                 detector_providers=json.dumps(res.get('detector_providers',[])))
        dest.mkdir(parents=True,exist_ok=True)
        (dest/'diagnostics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False))
        if a.debug:
            timeline(frames,res,dest/'timeline.png',r['reps'],stats)
            if not r['exact']:
                debug_video(frames,res,str(path),dest/'debug.mp4')
        rows.append(r)
        print(f"{name}: true {r['reps']} got {got_sets} {'OK' if r['right'] else 'X'} "
              f"used={r['used_ratio']:.1%} fps={r['fps']} providers={r['detector_providers']}",flush=True)
    with (out/'counts.csv').open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sc = score(rows)
    print(f"right {sc['right']}/{sc['sets']} ({100 * sc['share']:.0f}%), missed {sc['missed_per_100']:.1f} and "
          f"extra {sc['extra_per_100']:.1f} per 100 reps")
    print(f"exact {sc['exact']}/{sc['sets']}, MAE {sc['mae']:.2f}, sum {sc['got_sum']}/{sc['true_sum']}")
    (out/'summary.json').write_text(json.dumps(sc,indent=2))


if __name__ == "__main__":
    main()
