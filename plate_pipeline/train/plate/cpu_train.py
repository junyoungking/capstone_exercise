"""YOLOX fine-tuning loop that also runs without CUDA (YOLOX's own trainer is CUDA-only).

Called by train/plate/train.py when there is no GPU (or --device cpu). Same model, data loader, augmentation,
optimizer, learning-rate schedule and EMA as YOLOX's trainer; differences: one process, fixed input size (no
multi-scale), and the check after each epoch is our own recall / precision at IoU 0.5 on the val split
(core.recog.plate.decode_yolox, the same decoding the ONNX model gets), instead of COCO mAP. The best epoch
by recall (then precision) is saved as best_ckpt.pth in YOLOX's checkpoint format, so YOLOX's export works.
"""
import os
import time

import cv2
import numpy as np
import torch

from core.recog.plate import decode_yolox, letterbox
from train.plate import coco


def val_recall(model, data_dir, size, max_images=200, thr=0.5):
    ann = os.path.join(data_dir, "annotations", "instances_val2017.json")
    if not os.path.exists(ann):
        return None
    d = coco.load(ann)
    truth = coco.boxes_by_image(d)
    tp = fn = fp = 0
    model.eval()
    model.head.decode_in_inference = False
    with torch.no_grad():
        for im in d["images"][:max_images]:
            img = cv2.imread(os.path.join(data_dir, "val2017", im["file_name"]))
            x, r = letterbox(img, size)
            out = model(torch.from_numpy(x))[0].numpy()
            pred = [(p.box, p.score) for p in decode_yolox(out, size, r)]
            a, b, c, _ = coco.match(truth.get(im["id"], []), pred, thr)
            tp, fn, fp = tp + a, fn + b, fp + c
    model.train()
    return dict(recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), plates=tp + fn)


def train(exp_file, ckpt, out_dir, batch=8, threads=None, log_every=10):
    from yolox.exp import get_exp
    from yolox.utils import ModelEMA, load_ckpt

    if threads:
        torch.set_num_threads(threads)
    exp = get_exp(exp_file, None)
    size = exp.input_size[0]
    os.makedirs(out_dir, exist_ok=True)
    model = exp.get_model()
    if ckpt:
        model = load_ckpt(model, torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    model.train()
    loader = exp.get_data_loader(batch_size=batch, is_distributed=False, no_aug=False)
    iters = len(loader)
    optimizer = exp.get_optimizer(batch)
    sched = exp.get_lr_scheduler(exp.basic_lr_per_img * batch, iters)
    ema = ModelEMA(model, 0.9998)
    it = iter(loader)
    best = None
    t_start = time.time()
    print(f"CPU training: {len(loader.dataset)} images, input {size}, batch {batch}, {iters} iterations x "
          f"{exp.max_epoch} epochs, {torch.get_num_threads()} threads")
    for epoch in range(exp.max_epoch):
        if epoch == exp.max_epoch - exp.no_aug_epochs:
            print("last epochs without mosaic, with L1 loss")
            loader.close_mosaic()
            model.head.use_l1 = True
            it = iter(loader)
        t0 = time.time()
        losses = []
        for i in range(iters):
            imgs, targets, _, _ = next(it)
            imgs, targets = imgs.float(), targets.float()
            targets.requires_grad = False
            out = model(imgs, targets)
            loss = out["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            lr = sched.update_lr(epoch * iters + i + 1)
            for g in optimizer.param_groups:
                g["lr"] = lr
            losses.append(float(loss))
            if (i + 1) % log_every == 0:
                print(f"  epoch {epoch + 1} iter {i + 1}/{iters} loss {np.mean(losses[-log_every:]):.3f} lr {lr:.2e} "
                      f"({(time.time() - t0) / (i + 1):.2f} s/iter)")
        v = val_recall(ema.ema, exp.data_dir, size)
        state = {"model": ema.ema.state_dict(), "start_epoch": epoch + 1}
        torch.save(state, os.path.join(out_dir, "last_ckpt.pth"))
        msg = f"epoch {epoch + 1}/{exp.max_epoch} loss {np.mean(losses):.3f} ({time.time() - t0:.0f} s)"
        if v is not None:
            msg += f" val recall {v['recall']:.3f} precision {v['precision']:.3f} ({v['plates']} plates)"
            key = (v["recall"], v["precision"])
            if best is None or key > best:
                best = key
                torch.save(state, os.path.join(out_dir, "best_ckpt.pth"))
                msg += " *best"
        else:
            torch.save(state, os.path.join(out_dir, "best_ckpt.pth"))
        print(msg, flush=True)
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out_dir}")
    return os.path.join(out_dir, "best_ckpt.pth")
