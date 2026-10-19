"""Frame rows: one JSON object per frame, gzip JSONL per set (구현 설계 tab, 데이터 형식).

{"t": 12.533, "seq": 376, "plate": {"box": [x0, y0, x1, y1], "score": 0.91, "used": true,
 "center": [cx, cy], "size_px": [w, h], "src": "det" | "klt"}, "cam": [dx, dy]}
Person, keypoints and class probabilities are added by later steps.
"""
import gzip
import json


def write_rows(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
