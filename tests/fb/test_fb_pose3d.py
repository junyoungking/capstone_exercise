import json
import re
from types import SimpleNamespace

import numpy as np

from track3d import run_pose3d


def test_select_largest_box_uses_area_and_skips_empty():
    boxes = np.array([[0, 0, 2, 2], [0, 0, 4, 3], [0, 0, 10, 1]], dtype=np.float32)

    assert run_pose3d.select_largest_box(np.empty((0, 4))) is None
    np.testing.assert_array_equal(run_pose3d.select_largest_box(boxes), boxes[1:2])


def test_short_gap_interpolates_and_edges_hold_nearest_frame():
    points = np.full((6, 17, 2), np.nan, dtype=np.float32)
    points[1] = 10
    points[4] = 40

    filled, interpolated, nulls = run_pose3d.interpolate_short_gaps(points, max_gap=10)

    assert interpolated.tolist() == [True, False, True, True, False, True]
    assert not nulls.any()
    np.testing.assert_allclose(filled[0], 10)
    np.testing.assert_allclose(filled[2], 20)
    np.testing.assert_allclose(filled[3], 30)
    np.testing.assert_allclose(filled[5], 40)


def test_gap_of_10_fills_but_gap_of_11_stays_null_and_splits_runs():
    points = np.full((26, 17, 2), np.nan, dtype=np.float32)
    points[0] = 0
    points[11] = 11
    points[23] = 23

    filled_10, interpolated_10, nulls_10 = run_pose3d.interpolate_short_gaps(points[:12], max_gap=10)
    assert interpolated_10[1:11].all()
    assert not nulls_10.any()
    np.testing.assert_allclose(filled_10[5], 5)

    filled_11, interpolated_11, nulls_11 = run_pose3d.interpolate_short_gaps(points, max_gap=10)
    assert not interpolated_11[12:23].any()
    assert nulls_11[12:23].all()
    assert [run.tolist() for run in run_pose3d.valid_runs_after_interpolation(filled_11)] == [
        list(range(12)),
        [23, 24, 25],
    ]


def test_write_pose3d_js_preserves_frame_count_fps_and_nulls(tmp_path):
    frames = np.zeros((3, 17, 3), dtype=np.float32)
    frames[1] = np.nan
    out = tmp_path / "pose3d_user.js"

    payload = run_pose3d.write_pose3d_js(out, "user", 29.97, frames)

    assert payload["joints"] == "h36m17"
    assert payload["fps"] == 29.97
    assert payload["total"] == 3
    assert payload["frames"][1] is None
    text = out.read_text(encoding="utf-8")
    match = re.search(r"window\.POSE3D\[(\"user\")\] = (.*);\n$", text)
    assert match
    parsed = json.loads(match.group(2))
    assert parsed == payload


def test_bone_length_cv_known_sequence():
    frames = np.zeros((4, 17, 3), dtype=np.float32)
    frames[:, 1, 0] = [1, 1, 2, 2]
    frames[:, 2, 0] = [3, 3, 4, 4]
    rows = run_pose3d.bone_length_cv(frames, parents=(-1, 0, 1) + (-1,) * 14)

    assert rows[0]["joint"] == 1
    assert rows[0]["samples"] == 4
    assert rows[0]["mean"] == 1.5
    assert rows[0]["cv"] == np.std([1, 1, 2, 2]) / 1.5
    assert rows[1]["mean"] == 2.0
    assert rows[1]["cv"] == 0.0


def test_run_streams_video_and_preserves_output_frame_count(monkeypatch, tmp_path):
    images = [np.full((4, 5, 3), i, dtype=np.uint8) for i in range(4)]

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                run_pose3d.cv2.CAP_PROP_FPS: 30.0,
                run_pose3d.cv2.CAP_PROP_FRAME_WIDTH: 5,
                run_pose3d.cv2.CAP_PROP_FRAME_HEIGHT: 4,
                run_pose3d.cv2.CAP_PROP_FRAME_COUNT: len(images),
            }
            return values.get(prop, 0)

        def read(self):
            if self.index >= len(images):
                return False, None
            image = images[self.index]
            self.index += 1
            return True, image

        def release(self):
            pass

    monkeypatch.setattr(run_pose3d.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(run_pose3d, "ensure_models", lambda *_args: {"det": tmp_path / "det.onnx", "pose": tmp_path / "pose.onnx"})
    monkeypatch.setattr(
        run_pose3d,
        "extract_2d_keypoints_from_video",
        lambda *_args: (np.ones((4, 17, 2), dtype=np.float32), np.array([True, True, False, True]), 4, 0.2),
    )
    monkeypatch.setattr(
        run_pose3d,
        "load_videopose_model",
        lambda _model_dir: (SimpleNamespace(receptive_field=lambda: 1), tmp_path / "checkpoint.bin"),
    )
    monkeypatch.setattr(
        run_pose3d,
        "infer_runs_3d",
        lambda _model, points, _width, _height: (np.arange(points.shape[0] * 17 * 3, dtype=np.float32).reshape(points.shape[0], 17, 3), 0.3),
    )

    out = tmp_path / "pose3d_ref.js"
    stats = run_pose3d.run(tmp_path / "video.mp4", "ref", out, "lightweight", tmp_path / "models", progress_every=0)

    assert stats.total_frames == 4
    assert stats.detected_frames == 3
    assert json.loads(out.with_suffix(".js.stats.json").read_text(encoding="utf-8"))["stats"]["total_frames"] == 4
    payload = re.search(r"window\.POSE3D\[(\"ref\")\] = (.*);\n$", out.read_text(encoding="utf-8")).group(2)
    assert len(json.loads(payload)["frames"]) == 4
