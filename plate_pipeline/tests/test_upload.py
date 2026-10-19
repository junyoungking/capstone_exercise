import os

import cv2
import numpy as np
import pytest

from core.io import video
from core.io.rows import read_rows, write_rows
from core.signal.height import height_signal
from server.jobs import process_video

R, PLATE_MM = 40, 450.0          # plate radius in px and its real diameter -> 1 px = 5.625 mm
ROM_PX = 100                     # -> 0.5625 m


def _plate_y(t, top=150):
    """3 squat-like reps: 2 s still, then per rep 1.2 s down, 1.0 s up, 1.5 s still."""
    t0 = 2.0
    for _ in range(3):
        if t < t0:
            return top
        u = t - t0
        if u < 1.2:
            return top + ROM_PX * (1 - np.cos(np.pi * u / 1.2)) / 2
        if u < 2.2:
            return top + ROM_PX * (1 + np.cos(np.pi * (u - 1.2) / 1.0)) / 2
        t0 += 3.7
    return top


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("v") / "synthetic.mp4")
    rng = np.random.default_rng(0)
    bg = (rng.integers(150, 230, (480, 320, 1)) * np.ones((1, 1, 3))).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 480))
    for k in range(int(14 * 30)):
        im = bg.copy()
        cy = _plate_y(k / 30)
        c = (160, int(round(cy)))
        cv2.circle(im, c, R, (25, 25, 25), -1)
        for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):           # holes and marks on the ring
            cv2.circle(im, (int(c[0] + 0.65 * R * np.cos(a)), int(c[1] + 0.65 * R * np.sin(a))), 4, (200, 200, 200), -1)
        cv2.circle(im, c, 8, (90, 90, 90), -1)                            # hub
        vw.write(im)
    vw.release()
    return path


def test_video_frames_have_times(clip):
    inf = video.info(clip)
    assert (inf.width, inf.height) == (320, 480)
    ts = [t for t, _ in video.frames(clip, gray=True)]
    assert len(ts) == 420 and np.allclose(np.diff(ts), 1 / 30, atol=1e-3)


def test_height_signal_scale_and_sign():
    rows = [{"t": k / 30, "plate": {"used": True, "center": [100, 300 - 2 * k], "size_px": [90, 90]}}
            for k in range(31)]
    t, y, x = height_signal(rows, plate_mm=450)
    assert y[-1] == pytest.approx(60 * 0.45 / 90)          # moved up 60 px, 90 px = 0.45 m
    assert np.allclose(x, 0)


def test_rows_roundtrip(tmp_path):
    rows = [{"t": 0.0, "seq": 0, "plate": {"box": [1, 2, 3, 4], "used": True}}]
    write_rows(tmp_path / "r.jsonl.gz", rows)
    assert read_rows(tmp_path / "r.jsonl.gz") == rows


def test_upload_end_to_end(clip, tmp_path):
    box = (160 - R, 150 - R, 160 + R, 150 + R)
    res, rows, _ = process_video(clip, "squat", PLATE_MM, init_box=box, out_dir=str(tmp_path))
    assert [s["reps"] for s in res["sets"]] == [3]
    roms = [r["rom_m"] for r in res["sets"][0]["rep_values"]]
    assert np.allclose(roms, ROM_PX * PLATE_MM / 1000 / (2 * R), rtol=0.06)
    downs = [r["down"] for r in res["sets"][0]["rep_values"]]
    assert np.allclose(downs, 1.2, atol=0.15)
    assert os.path.exists(tmp_path / "report.html") and os.path.exists(tmp_path / "rows.jsonl.gz")
