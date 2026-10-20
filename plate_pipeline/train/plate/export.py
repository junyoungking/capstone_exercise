"""Export a trained YOLOX checkpoint to ONNX and check it against PyTorch.

Used by train/plate/train.py (YOLOX's tools/export_onnx.py calls torch.onnx._export, which recent PyTorch
no longer has, and its logger swallows the error). The graph ends at the raw head output (no decoding);
core/recog/plate.py decode_yolox does the rest. After export the ONNX output is compared with the PyTorch
output on one image; a mismatch raises.
"""
import os

import numpy as np
import torch


def export(exp_file, ckpt, onnx_path, check_image=None, opset=17):
    from yolox.exp import get_exp
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    model.head.decode_in_inference = False
    h, w = exp.test_size
    dummy = torch.zeros(1, 3, h, w)
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    kw = dict(input_names=["images"], output_names=["output"], opset_version=opset)
    try:
        torch.onnx.export(model, dummy, onnx_path, dynamo=False, **kw)
    except TypeError:                      # older PyTorch without the dynamo switch
        torch.onnx.export(model, dummy, onnx_path, **kw)
    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX export wrote nothing: {onnx_path}")

    import onnxruntime as ort
    if check_image is not None:
        import cv2
        from core.recog.plate import letterbox
        x, _ = letterbox(cv2.imread(check_image), h)
    else:
        x = np.random.default_rng(0).uniform(0, 255, (1, 3, h, w)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x))[0].numpy()
    got = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"images": x})[0][0]
    err = float(np.abs(ref - got).max())
    if not np.allclose(ref, got, atol=1e-3, rtol=1e-3):
        raise RuntimeError(f"ONNX output differs from PyTorch (max abs diff {err:.4g})")
    print(f"ONNX -> {onnx_path} (input {h}, output {got.shape}, max diff vs PyTorch {err:.1e})")
    return onnx_path
