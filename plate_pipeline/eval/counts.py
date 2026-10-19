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
import os

from server.jobs import process_video


def score(rows):
    right = sum(r["right"] for r in rows)
    true = sum(r["reps"] for r in rows)
    miss = sum(max(0, r["reps"] - r["got"]) for r in rows)
    extra = sum(max(0, r["got"] - r["reps"]) for r in rows)
    return dict(sets=len(rows), right=right, share=right / max(len(rows), 1),
                missed_per_100=100 * miss / max(true, 1), extra_per_100=100 * extra / max(true, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="eval_out/counts")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for s in csv.DictReader(open(a.sets)):
        box = tuple(float(v) for v in s["box"].split()) if s.get("box") else None
        name = os.path.splitext(os.path.basename(s["path"]))[0]
        res, _, _ = process_video(s["path"], s["exercise"], float(s.get("plate_mm") or 450), init_box=box,
                                  model=a.model, out_dir=os.path.join(a.out, name))
        got_sets = [x["reps"] for x in res["sets"]]
        got = sum(got_sets)
        r = dict(video=name, exercise=s["exercise"], reps=int(s["reps"]), got=got, sets=len(got_sets),
                 right=len(got_sets) == 1 and got == int(s["reps"]))
        rows.append(r)
        print(f"{name}: true {r['reps']} got {got_sets} {'OK' if r['right'] else 'X'}")
    with open(os.path.join(a.out, "counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sc = score(rows)
    print(f"right {sc['right']}/{sc['sets']} ({100 * sc['share']:.0f}%), missed {sc['missed_per_100']:.1f} and "
          f"extra {sc['extra_per_100']:.1f} per 100 reps")


if __name__ == "__main__":
    main()
