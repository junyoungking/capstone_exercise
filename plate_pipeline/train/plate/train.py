"""Fine-tune YOLOX-Tiny on the plate data and export ONNX. GPU or CPU, Windows or Linux.

python -m train.plate.train --data datasets/plate                 # picks the GPU if there is one
python -m train.plate.train --data datasets/plate --device cpu    # CPU: input 320, batch 8, 40 epochs by default
python -m train.plate.train --export-only --ckpt <best_ckpt.pth>

Steps: clone YOLOX into --work (once) and use it from there (no build step), install its few Python
dependencies, download yolox_tiny.pth (COCO pretrained), train with exps/plate_tiny.py, export
models/plate_tiny.onnx without decoding in the graph (train/plate/export.py; the decoding is
core/recog/plate.py: decode_yolox) and check the ONNX output against PyTorch.
Training: train/plate/loop.py on the GPU (fp16) or the CPU - same model, data and schedule as YOLOX's trainer,
tried end to end. --trainer yolox uses YOLOX's own tools/train.py instead (GPU only, multi-scale, COCO mAP).
"""
import argparse
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WEIGHTS = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
DEPS = ["loguru", "tabulate", "pycocotools", "tqdm", "psutil", "thop", "tensorboard", "onnx", "onnxruntime",
        "opencv-python"]


def run(cmd, cwd=None, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def setup(work):
    os.makedirs(work, exist_ok=True)
    yolox = os.path.join(work, "YOLOX")
    if not os.path.isdir(os.path.join(yolox, "yolox")):
        run(["git", "clone", "--depth", "1", "https://github.com/Megvii-BaseDetection/YOLOX.git", yolox])
    missing = []
    for mod, pkg in (("loguru", "loguru"), ("tabulate", "tabulate"), ("pycocotools", "pycocotools"),
                     ("thop", "thop"), ("onnx", "onnx"), ("onnxruntime", "onnxruntime"), ("psutil", "psutil"),
                     ("tqdm", "tqdm")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        run([sys.executable, "-m", "pip", "install", *missing])
    w = os.path.join(yolox, "yolox_tiny.pth")
    if not os.path.exists(w):
        print("downloading", WEIGHTS, flush=True)
        urllib.request.urlretrieve(WEIGHTS, w)
    return yolox, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "plate"))
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=None, help="default 32 (GPU) / 8 (CPU)")
    ap.add_argument("--epochs", type=int, default=None, help="default 80 (GPU) / 40 (CPU)")
    ap.add_argument("--size", type=int, default=None, help="input size, multiple of 32; default 416 (GPU) / 320 (CPU)")
    ap.add_argument("--workers", type=int, default=None, help="data loader workers; default 4 (GPU) / 2 (CPU)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for training (default: torch's choice)")
    ap.add_argument("--devices", type=int, default=1, help="number of GPUs (--trainer yolox only)")
    ap.add_argument("--trainer", default="own", choices=["own", "yolox"])
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "plate_tiny.onnx"))
    a = ap.parse_args()
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    cpu = dev == "cpu"
    batch = a.batch or (8 if cpu else 32)
    epochs = a.epochs or (40 if cpu else 80)
    size = a.size or (320 if cpu else 416)
    workers = a.workers if a.workers is not None else (2 if cpu else 4)
    print(f"device {dev}, input {size}, batch {batch}, epochs {epochs}", flush=True)

    yolox, w = setup(a.work)
    exp = os.path.join(HERE, "exps", "plate_tiny.py")
    env = dict(os.environ, PLATE_DATA=os.path.abspath(a.data), PLATE_EPOCHS=str(epochs), PLATE_SIZE=str(size),
               PLATE_WORKERS=str(workers))
    env["PYTHONPATH"] = os.pathsep.join([yolox, ROOT, env.get("PYTHONPATH", "")])
    os.environ.update({k: env[k] for k in ("PLATE_DATA", "PLATE_EPOCHS", "PLATE_SIZE", "PLATE_WORKERS")})
    own = a.trainer == "own" or cpu
    out_dir = os.path.join(yolox, "YOLOX_outputs", "plate_tiny" + ("_own" if own else ""))
    if not a.export_only:
        if own:
            sys.path.insert(0, yolox)
            from train.plate.loop import train
            train(exp, w, out_dir, batch=batch, threads=a.threads, device="cpu" if cpu else "cuda")
        else:
            run([sys.executable, "tools/train.py", "-f", exp, "-d", str(a.devices), "-b", str(batch), "--fp16", "-o",
                 "-c", w], cwd=yolox, env=env)
    ckpt = a.ckpt or os.path.join(out_dir, "best_ckpt.pth")
    if not os.path.exists(ckpt):
        sys.exit(f"no checkpoint at {ckpt}")
    if yolox not in sys.path:
        sys.path.insert(0, yolox)
    from train.plate.export import export
    val_imgs = os.path.join(a.data, "val2017")
    first = sorted(os.listdir(val_imgs))[0] if os.path.isdir(val_imgs) and os.listdir(val_imgs) else None
    export(exp, ckpt, a.onnx, os.path.join(val_imgs, first) if first else None)


if __name__ == "__main__":
    main()
