"""Build the YOLOX training folders from your reviewed labels and the public datasets.

python -m train.plate.make_dataset --own data/own/labels.json --own-images data/own/images \
    --public data/public/plate-detection --public data/public/barbell-detection \
    [--public-classes plate,plates] [--test-sessions 1103_gymB] [--val-sessions 1027_gymA] \
    [--blur 0.3] --out datasets/plate

Split (구현 설계 tab: 사람·장소 단위로 나눈다): your frames go to train / val / test by session, never by frame.
Without --test-sessions / --val-sessions, whole sessions are taken (smallest session names first after a
fixed shuffle) until test has >= 300 frames and val >= 15% of the rest. Public images only go to train.
--blur: this share of training images gets one extra copy with motion blur and a brightness/contrast change
(the 흐림·밝기 증강; size changes come from YOLOX's mosaic/scale augmentation).
Public datasets: Roboflow "COCO" exports (folders with train/valid/test and _annotations.coco.json). Only
categories named in --public-classes become "plate"; look at a few with review.py first - the box must go
around the plate rim.
Writes datasets/plate/{train2017,val2017,test2017}/ and annotations/instances_{train,val,test}2017.json.
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np

from train.plate import coco


def split_sessions(counts, test_min=300, val_frac=0.15, seed=7):
    """counts: {session: frames}. Returns (test sessions, val sessions)."""
    names = sorted(counts)
    random.Random(seed).shuffle(names)
    test, val, n_test = [], [], 0
    for s in names:
        if n_test < test_min:
            test.append(s)
            n_test += counts[s]
    rest = [s for s in names if s not in test]
    n_rest = sum(counts[s] for s in rest)
    n_val = 0
    for s in rest:
        if n_val < val_frac * n_rest and len(rest) - len(val) > 1:
            val.append(s)
            n_val += counts[s]
    return test, val


def read_public(root, classes):
    """Yields (image path, [boxes], name) from a Roboflow COCO export."""
    name = os.path.basename(os.path.normpath(root))
    for js in sorted(glob.glob(os.path.join(root, "**", "_annotations.coco.json"), recursive=True)):
        d = coco.load(js)
        cats = {c["id"]: c["name"] for c in d["categories"]}
        keep = {i for i, n in cats.items() if n.lower() in classes}
        if not keep:
            print(f"  {js}: no category in {sorted(classes)}; categories are {sorted(set(cats.values()))}")
            continue
        by = {}
        for a in d["annotations"]:
            if a["category_id"] in keep:
                x, y, w, h = a["bbox"]
                by.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
        for im in d["images"]:
            yield os.path.join(os.path.dirname(js), im["file_name"]), by.get(im["id"], []), name


def blur_copy(img, rng):
    k = int(rng.integers(7, 22)) | 1
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    ker = cv2.warpAffine(ker, rot, (k, k))
    ker /= max(ker.sum(), 1e-6)
    out = cv2.filter2D(img, -1, ker)
    a, b = rng.uniform(0.6, 1.4), rng.uniform(-40, 40)
    return np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", required=True)
    ap.add_argument("--own-images", required=True)
    ap.add_argument("--public", action="append", default=[])
    ap.add_argument("--public-classes", default="plate,plates,weight plate,weight-plate")
    ap.add_argument("--test-sessions", default=None)
    ap.add_argument("--val-sessions", default=None)
    ap.add_argument("--blur", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    own = coco.load(a.own)
    imgs = [im for im in own["images"] if im.get("reviewed") and not im.get("unusable")]
    print(f"own: {len(imgs)} reviewed frames of {len(own['images'])}")
    boxes = coco.boxes_by_image(own)
    counts = {}
    for im in imgs:
        counts[im["session"]] = counts.get(im["session"], 0) + 1
    if a.test_sessions or a.val_sessions:
        test = a.test_sessions.split(",") if a.test_sessions else []
        val = a.val_sessions.split(",") if a.val_sessions else []
    else:
        test, val = split_sessions(counts)
    print(f"test sessions {test}, val sessions {val}")
    split = {s: ("test" if s in test else "val" if s in val else "train") for s in counts}
    out = {k: coco.empty() for k in ("train", "val", "test")}
    for k in out:
        os.makedirs(os.path.join(a.out, f"{k}2017"), exist_ok=True)

    def put(k, src, bxs, name, **extra):
        img = cv2.imread(src)
        if img is None:
            print(f"  cannot read {src}")
            return
        cv2.imwrite(os.path.join(a.out, f"{k}2017", name), img)
        iid = coco.add_image(out[k], name, img.shape[1], img.shape[0], **extra)
        for b in bxs:
            coco.add_box(out[k], iid, b)
        if k == "train" and rng.random() < a.blur:
            bname = "blur_" + name
            cv2.imwrite(os.path.join(a.out, "train2017", bname), blur_copy(img, rng))
            iid = coco.add_image(out[k], bname, img.shape[1], img.shape[0], **extra, aug="blur")
            for b in bxs:
                coco.add_box(out[k], iid, b)

    for im in imgs:
        put(split[im["session"]], os.path.join(a.own_images, im["file_name"]), boxes.get(im["id"], []),
            im["file_name"], session=im["session"], exercise=im.get("exercise", "unknown"), source="own")
    classes = {c.strip().lower() for c in a.public_classes.split(",")}
    for root in a.public:
        n = 0
        for path, bxs, name in read_public(root, classes):
            put("train", path, bxs, f"{name}_{n:05d}{os.path.splitext(path)[1]}", session=f"public:{name}",
                exercise="unknown", source=f"public:{name}")
            n += 1
        print(f"public {root}: {n} images")
    os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
    for k, d in out.items():
        coco.save(d, os.path.join(a.out, "annotations", f"instances_{k}2017.json"))
        print(f"{k}: {len(d['images'])} images, {len(d['annotations'])} boxes")
    if len(out["test"]["images"]) < 300:
        print("warning: fewer than 300 test frames (구현 설계 tab: 300 held-out frames)")


if __name__ == "__main__":
    main()
