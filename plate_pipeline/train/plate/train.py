"""Fine-tune YOLOX-Tiny on the plate data and export ONNX (works on Windows and Linux).

python -m train.plate.train --data datasets/plate [--batch 32] [--epochs 80] [--work train/plate/work]
python -m train.plate.train --export-only --ckpt train/plate/work/YOLOX/YOLOX_outputs/plate_tiny/best_ckpt.pth

Steps: clone YOLOX (once), install it without its pinned extras, download yolox_tiny.pth (COCO pretrained),
train with exps/plate_tiny.py, export models/plate_tiny.onnx without decoding in the graph (the decoding is in
core/recog/plate.py: decode_yolox). Needs a CUDA PyTorch already installed (see README).
"""
import argparse
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WEIGHTS = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
DEPS = ["loguru", "tabulate", "pycocotools", "tqdm", "psutil", "ninja", "thop", "tensorboard", "onnx",
        "onnxruntime", "opencv-python"]


def run(cmd, cwd=None, env=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def setup(work):
    os.makedirs(work, exist_ok=True)
    yolox = os.path.join(work, "YOLOX")
    if not os.path.isdir(yolox):
        run(["git", "clone", "--depth", "1", "https://github.com/Megvii-BaseDetection/YOLOX.git", yolox])
    try:
        import yolox as _  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", *DEPS])
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."], cwd=yolox)
    w = os.path.join(yolox, "yolox_tiny.pth")
    if not os.path.exists(w):
        print("downloading", WEIGHTS)
        urllib.request.urlretrieve(WEIGHTS, w)
    return yolox, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "plate"))
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "plate_tiny.onnx"))
    a = ap.parse_args()
    yolox, w = setup(a.work)
    exp = os.path.join(HERE, "exps", "plate_tiny.py")
    env = dict(os.environ, PLATE_DATA=os.path.abspath(a.data), PLATE_EPOCHS=str(a.epochs),
               PLATE_WORKERS=str(a.workers))
    if not a.export_only:
        run([sys.executable, "tools/train.py", "-f", exp, "-d", str(a.devices), "-b", str(a.batch), "--fp16", "-o",
             "-c", w], cwd=yolox, env=env)
    ckpt = a.ckpt or os.path.join(yolox, "YOLOX_outputs", "plate_tiny", "best_ckpt.pth")
    os.makedirs(os.path.dirname(os.path.abspath(a.onnx)), exist_ok=True)
    run([sys.executable, "tools/export_onnx.py", "--output-name", os.path.abspath(a.onnx), "-f", exp, "-c", ckpt,
         "--no-onnxsim"], cwd=yolox, env=env)
    print("ONNX ->", a.onnx)


if __name__ == "__main__":
    main()
