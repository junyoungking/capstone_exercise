import json
import sys

import cv2
import numpy as np
import pytest

from core.recog.plate import Detection, decode_yolox, letterbox, yolox_grids
from eval.counts import score
from eval.noise import noise_cm
from eval.plate_recall import evaluate
from train.plate import coco
from train.plate.make_dataset import split_sessions


def test_decode_yolox_matches_the_yolox_formula():
    size, ratio = 416, 0.5
    grid, stride = yolox_grids(size)
    out = np.zeros((len(grid), 6), np.float32)
    out[:, 2:4] = np.log(4.0)
    out[:, 4:] = 0.01                                       # nothing above the score limit ...
    n8 = (size // 8) ** 2
    k = n8 + 5 * (size // 16) + 7                           # ... except stride 16, grid x 7, y 5
    out[k] = [0.5, 0.25, np.log(5.0), np.log(3.0), 0.9, 0.8]
    dets = decode_yolox(out, size, ratio, score_min=0.3)
    assert len(dets) == 1
    cx, cy, w, h = (7 + 0.5) * 16, (5 + 0.25) * 16, 5 * 16, 3 * 16
    assert np.allclose(dets[0].box, np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) / ratio)
    assert dets[0].score == pytest.approx(0.72)


def test_letterbox():
    x, r = letterbox(np.zeros((1920, 1080, 3), np.uint8), 416)
    assert x.shape == (1, 3, 416, 416) and r == pytest.approx(416 / 1920)
    assert x[0, 0, 0, -1] == 114                              # padding on the right


def test_match():
    truth = [(0, 0, 10, 10), (20, 20, 30, 30)]
    pred = [((1, 1, 10, 10), 0.9), ((50, 50, 60, 60), 0.8)]
    tp, fn, fp, ious = coco.match(truth, pred)
    assert (tp, fn, fp) == (1, 1, 1) and ious[0] == pytest.approx(0.81)


def test_split_by_session():
    counts = {f"s{i}": 60 for i in range(12)}
    test, val = split_sessions(counts)
    assert sum(counts[s] for s in test) >= 300 and not set(test) & set(val)
    assert sum(counts[s] for s in val) >= 0.15 * sum(counts[s] for s in counts if s not in test)


def test_make_dataset(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    d = coco.empty()
    for s in ("a", "b", "c"):
        for k in range(2):
            name = f"{s}{k}.jpg"
            im = np.full((80, 120, 3), 200, np.uint8)
            cv2.circle(im, (60, 40), 20, (20, 20, 20), -1)
            cv2.imwrite(str(img_dir / name), im)
            iid = coco.add_image(d, name, 120, 80, session=s, exercise="squat", source="own", reviewed=True)
            coco.add_box(d, iid, (40, 20, 80, 60))
    coco.save(d, str(tmp_path / "labels.json"))
    from train.plate import make_dataset
    out = tmp_path / "ds"
    monkeypatch.setattr(sys, "argv", ["x", "--own", str(tmp_path / "labels.json"), "--own-images", str(img_dir),
                                      "--test-sessions", "c", "--val-sessions", "b", "--blur", "1.0", "--out", str(out)])
    make_dataset.main()
    tr = json.load(open(out / "annotations" / "instances_train2017.json"))
    te = json.load(open(out / "annotations" / "instances_test2017.json"))
    assert len(tr["images"]) == 4 and len(tr["annotations"]) == 4       # 2 frames + 2 blurred copies
    assert {im["session"] for im in te["images"]} == {"c"}
    assert (out / "train2017" / "blur_a0.jpg").exists()


def test_recall_with_a_stub_detector(tmp_path):
    d = coco.empty()
    cv2.imwrite(str(tmp_path / "f.jpg"), np.zeros((50, 50, 3), np.uint8))
    iid = coco.add_image(d, "f.jpg", 50, 50, exercise="bench")
    coco.add_box(d, iid, (10, 10, 30, 30))
    res = evaluate(lambda img: [Detection((11, 11, 30, 30), 0.9)], d, str(tmp_path))
    assert res["all"]["recall"] == 1.0 and res["bench"]["precision"] == 1.0


def test_counts_score():
    rows = [dict(reps=5, got=5, right=True), dict(reps=5, got=4, right=False), dict(reps=10, got=11, right=False)]
    s = score(rows)
    assert s["right"] == 1 and s["missed_per_100"] == pytest.approx(5.0) and s["extra_per_100"] == pytest.approx(5.0)


def test_noise_cm():
    rng = np.random.default_rng(1)
    rows = [{"t": k / 30, "plate": {"used": True, "center": [100 + rng.normal(0, 1), 300 + rng.normal(0, 1)],
                                    "size_px": [90, 90]}} for k in range(300)]
    r = noise_cm(rows, 450, skip_s=0)
    # 1 px sd in each frame -> frame-to-frame integration keeps it at about 0.5 cm (450 mm / 90 px)
    assert 0.35 < r["vertical"] < 0.65
