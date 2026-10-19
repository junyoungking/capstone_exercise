"""Small COCO helpers shared by the plate-detector scripts. One category: 1 "plate".

Image entries carry extra keys used for splitting and reporting:
  "session" : recording day/place id (train/val/test are split by session, never by frame)
  "exercise": squat | bench | deadlift | unknown
  "source"  : own | public:<name>
"""
import json
import os

CATS = [{"id": 1, "name": "plate"}]


def empty():
    return {"images": [], "annotations": [], "categories": CATS}


def load(path):
    d = json.load(open(path))
    d.setdefault("categories", CATS)
    return d


def save(d, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)


def boxes_by_image(d):
    out = {im["id"]: [] for im in d["images"]}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        out.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
    return out


def add_image(d, file_name, w, h, **extra):
    iid = max((im["id"] for im in d["images"]), default=0) + 1
    d["images"].append(dict(id=iid, file_name=file_name, width=int(w), height=int(h), **extra))
    return iid


def add_box(d, image_id, box, score=None):
    x0, y0, x1, y1 = (float(v) for v in box)
    aid = max((a["id"] for a in d["annotations"]), default=0) + 1
    a = dict(id=aid, image_id=image_id, category_id=1, bbox=[x0, y0, x1 - x0, y1 - y0],
             area=(x1 - x0) * (y1 - y0), iscrowd=0)
    if score is not None:
        a["score"] = float(score)
    d["annotations"].append(a)
    return aid


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(truth, pred, thr=0.5):
    """truth: boxes (x0, y0, x1, y1); pred: (box, score) pairs. Greedy one-to-one matching, highest score
    first, IoU >= thr. Returns (true positives, missed truths, extra predictions, IoUs of the matches)."""
    used, ious = set(), []
    for pb, _ in sorted(pred, key=lambda p: -p[1]):
        best, bi = thr, None
        for i, tb in enumerate(truth):
            if i not in used:
                v = iou(pb, tb)
                if v >= best:
                    best, bi = v, i
        if bi is not None:
            used.add(bi)
            ious.append(best)
    tp = len(used)
    return tp, len(truth) - tp, len(pred) - tp, ious
