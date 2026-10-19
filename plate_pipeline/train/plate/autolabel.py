"""First boxes with Grounding DINO (zero-shot), to be corrected by hand with review.py.

python -m train.plate.autolabel --frames data/own/frames.json --images data/own/images --out data/own/auto.json
       [--model IDEA-Research/grounding-dino-base] [--prompt "weight plate."] [--box-thr 0.30]

Needs: pip install torch transformers (GPU recommended). The model is downloaded from Hugging Face once.
Boxes that are much wider than tall or tiny are dropped; the rest are kept with their score so review.py can
show low-score ones first. Label rule (README): every plate that is at least half visible, the whole rim
inside the box (not the bar end or the collar).
"""
import argparse
import os

import numpy as np
from PIL import Image

from train.plate import coco


def load_model(name, device):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(device).eval()
    return proc, model, torch


def detect(proc, model, torch, image, prompt, box_thr, text_thr, device):
    inputs = proc(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    size = [image.size[::-1]]
    try:      # transformers >= 4.51 renamed box_threshold -> threshold
        r = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=box_thr,
                                                        text_threshold=text_thr, target_sizes=size)[0]
    except TypeError:
        r = proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=box_thr,
                                                        text_threshold=text_thr, target_sizes=size)[0]
    return [(tuple(b.tolist()), float(s)) for b, s in zip(r["boxes"], r["scores"])]


def plausible(box, w, h, min_frac=0.04):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return False
    ar = bw / bh
    return 0.3 <= ar <= 1.6 and max(bw, bh) >= min_frac * min(w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--prompt", default="weight plate.")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    proc, model, torch = load_model(a.model, a.device)
    d = coco.load(a.frames)
    d["annotations"] = []
    for k, im in enumerate(d["images"]):
        img = Image.open(os.path.join(a.images, im["file_name"])).convert("RGB")
        for box, s in detect(proc, model, torch, img, a.prompt, a.box_thr, a.text_thr, a.device):
            if plausible(box, im["width"], im["height"]):
                coco.add_box(d, im["id"], box, s)
        if (k + 1) % 20 == 0:
            print(f"{k + 1}/{len(d['images'])}")
    coco.save(d, a.out)
    n = np.bincount([a_["image_id"] for a_ in d["annotations"]], minlength=len(d["images"]) + 1)[1:]
    print(f"boxes {len(d['annotations'])}, images without a box {int((n == 0).sum())} -> {a.out}")


if __name__ == "__main__":
    main()
