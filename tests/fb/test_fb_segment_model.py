import sys
import types

import numpy as np
import pytest
import torch

from feedback import config
from feedback import stgcn_infer
from feedback.segment import build_model_segments, hip_velocity, _select_endpoint
from unified_feedback_v4 import SIDE_IDS


class FakeF:
    @staticmethod
    def softmax(logits, dim):
        return torch.nn.functional.softmax(logits, dim=dim)


class FakeModel:
    def __call__(self, x):
        batch = x.shape[0]
        action = torch.zeros((batch, 3), dtype=torch.float32)
        phase = torch.zeros((batch, 3), dtype=torch.float32)
        action[:, 0] = 5.0
        for row in range(batch):
            if row < 10:
                phase[row, 0] = 5.0
            elif row < 30:
                phase[row, 1] = 5.0
            else:
                phase[row, 2] = 5.0
        return action, phase


class FakeInfer:
    def __init__(self, checkpoint_path, device, inference_interval, required):
        self.available = True
        self.model = FakeModel()
        self.clip_len = config.MODEL_CLIP_LENGTH
        self.cfg = {"derivative_mode": "pose"}
        self.device = "cpu"
        self.warning = None
        self.unavailable_reason = None


def fake_adapter(captured):
    module = types.SimpleNamespace()
    module.CLASS_LIST = ["squat", "benchpress", "deadlift"]
    module.PHASE_NAMES = ["ready", "down", "up"]
    module.REALTIME_DERIVATIVE_MODE = "pose"
    module.RealtimeSTGCNInfer = FakeInfer
    module.RealtimeModelUnavailable = RuntimeError
    module.F = FakeF
    module.torch = torch

    def normalize_kpts(window):
        captured.append(window.copy())
        return window

    def build_pose_input_features(window, derivative_mode):
        assert derivative_mode == "pose"
        return window

    module.normalize_kpts = normalize_kpts
    module.build_pose_input_features = build_pose_input_features
    return module


def synthetic_pose(total=64, invalid=()):
    hip = np.full(total, 0.2, dtype=np.float32)
    hip[10:31] = np.linspace(0.2, 0.8, 21)
    hip[31:35] = 0.8
    hip[35:56] = np.linspace(0.8, 0.2, 21)
    lms = np.zeros((total, 33, 4), dtype=np.float32)
    lms[:, :, 0] = 0.5
    lms[:, :, 1] = 0.3
    lms[:, :, 3] = 1.0
    lms[:, 11, :2] = (0.45, 0.1)
    lms[:, 12, :2] = (0.55, 0.1)
    lms[:, 23, :2] = np.column_stack([np.full(total, 0.45), hip])
    lms[:, 24, :2] = np.column_stack([np.full(total, 0.55), hip])
    lms[:, 25, :2] = (0.45, 0.9)
    lms[:, 26, :2] = (0.55, 0.9)
    lms[:, 27, :2] = (0.45, 1.1)
    lms[:, 28, :2] = (0.55, 1.1)
    for ids in SIDE_IDS.values():
        lms[:, ids["foot"], 0] = lms[:, ids["heel"], 0] - 0.1
    valid = np.ones(total, dtype=bool)
    for frame in invalid:
        valid[frame] = False
        lms[frame] = np.nan
    return {"lms": lms, "valid": valid, "fps": 30.0, "w": 640, "h": 480,
            "total": total, "cache_hit": False}


def pose_from_hip(hip):
    total = len(hip)
    hip = np.asarray(hip, dtype=np.float32)
    lms = np.zeros((total, 33, 4), dtype=np.float32)
    lms[:, :, 0] = 0.5
    lms[:, :, 1] = 0.3
    lms[:, :, 3] = 1.0
    lms[:, 11, :2] = (0.45, 0.1)
    lms[:, 12, :2] = (0.55, 0.1)
    lms[:, 23, :2] = np.column_stack([np.full(total, 0.45), hip])
    lms[:, 24, :2] = np.column_stack([np.full(total, 0.55), hip])
    lms[:, 25, :2] = (0.45, 0.9)
    lms[:, 26, :2] = (0.55, 0.9)
    lms[:, 27, :2] = (0.45, 1.1)
    lms[:, 28, :2] = (0.55, 1.1)
    for ids in SIDE_IDS.values():
        lms[:, ids["foot"], 0] = lms[:, ids["heel"], 0] - 0.1
    valid = np.isfinite(hip)
    lms[~valid] = np.nan
    return {"lms": lms, "valid": valid, "fps": 30.0, "w": 640, "h": 480,
            "total": total, "cache_hit": False}


def model_inference(total, down, up):
    phase_ids = [0] * total
    phase_ids[down[0]:down[1]] = [1] * (down[1] - down[0])
    phase_ids[up[0]:up[1]] = [2] * (up[1] - up[0])
    return {"exercise": "squat", "clean_phase_ids": phase_ids}


def base_velocity_hip(total=100):
    hip = np.full(total, 0.2, dtype=np.float32)
    hip[21:46] = np.linspace(0.23, 0.8, 25, dtype=np.float32)
    hip[46:51] = 0.8
    hip[51:81] = np.linspace(0.78, 0.2, 30, dtype=np.float32)
    hip[81:] = 0.2
    return hip


def first_rep(hip, down=(25, 51), up=(51, 85), overrides=None):
    result = build_model_segments(
        pose_from_hip(hip),
        video_settings(),
        model_inference(len(hip), down, up),
        overrides,
    )
    assert len(result["reps"]) == 1
    return result["reps"][0]


def video_settings():
    return {"video": "data/videos/synthetic.mp4", "view": "side"}


def test_infer_phases_reuses_realtime_adapter_and_anchors_windows_on_last_frame(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(stgcn_infer, "_adapter", lambda: fake_adapter(captured))
    pose = synthetic_pose(invalid={4})

    result = stgcn_infer.infer_phases(pose, tmp_path / "fake.pt")

    assert captured[0].shape == (config.MODEL_CLIP_LENGTH, 33, 3)
    np.testing.assert_equal(captured[0][:-1], captured[0][1:])
    np.testing.assert_allclose(captured[0][-1, :, :2], pose["lms"][0, :, :2])
    assert np.count_nonzero(captured[4][-1]) == 0
    assert result["exercise"] == "squat"
    assert result["exercise_ratio"] == 1.0
    assert result["votes"] == {"squat": 63, "benchpress": 0, "deadlift": 0}
    assert result["ratios"] == {"squat": 1.0, "benchpress": 0.0, "deadlift": 0.0}
    assert result["checkpoint"].endswith("fake.pt")
    assert result["phase_names"][9] == "ready"
    assert result["phase_names"][10] == "down"
    assert result["phase_names"][30] == "up"


def test_clean_phases_median_absorbs_short_runs_and_enforces_allowed_order(monkeypatch):
    monkeypatch.setattr(stgcn_infer, "_adapter", lambda: fake_adapter([]))
    raw = [0] * 7 + [1] * 7 + [0] * 3 + [2] * 7 + [0] * 6

    cleaned = stgcn_infer.clean_phases(raw)

    assert cleaned["phase_names"][:7] == ["ready"] * 7
    assert cleaned["phase_names"][7:17] == ["down"] * 10
    assert cleaned["phase_names"][17:24] == ["up"] * 7
    assert cleaned["phase_names"][24:] == ["ready"] * 6
    assert {note["type"] for note in cleaned["cleaning"]["notes"]} == {"short_run_absorbed"}


def test_phase_cleanup_run_length_boundary_and_allowed_transitions(monkeypatch):
    monkeypatch.setattr(stgcn_infer, "_adapter", lambda: fake_adapter([]))
    monkeypatch.setattr(config, "MODEL_PHASE_FILTER_WINDOW", 1)
    monkeypatch.setattr(config, "MODEL_MIN_PHASE_FRAMES", 5)

    retained = stgcn_infer.clean_phases([0] * 5 + [1] * 5 + [2] * 5 + [1] * 5)
    assert retained["phase_ids"] == [0] * 5 + [1] * 5 + [2] * 5 + [1] * 5
    assert retained["cleaning"]["notes"] == []

    absorbed = stgcn_infer.clean_phases([0] * 5 + [1] * 4 + [2] * 5)
    assert absorbed["phase_ids"] == [0] * 14
    assert absorbed["cleaning"]["notes"][0]["type"] == "short_run_absorbed"

    prohibited = stgcn_infer.clean_phases([0] * 5 + [2] * 5 + [1] * 5 + [0] * 5)
    assert prohibited["phase_ids"] == [0] * 10 + [1] * 10
    assert {note["type"] for note in prohibited["cleaning"]["notes"]} == {
        "invalid_transition_replaced",
    }


def test_build_model_segments_uses_fallback_hip_rules_and_override(monkeypatch):
    monkeypatch.setattr(config, "MODEL_STOP_SPEED_RATIOS", (0,))
    pose = synthetic_pose()
    inference = {
        "exercise": "squat",
        "clean_phase_ids": [0] * 10 + [1] * 20 + [2] * 26 + [0] * 8,
    }

    result = build_model_segments(pose, video_settings(), inference, {"1": {"BOTTOM": 32}})

    assert result["source"] == "model+override"
    assert result["exercise"] == "squat"
    assert result["facing"] == "left"
    rep = result["reps"][0]
    assert rep["rep"] == 1
    assert rep["positions"] == {
        "START": 10,
        "MID_DOWN": 21,
        "BOTTOM": 32,
        "MID_UP": 45,
        "LOCKOUT": 55,
    }
    assert rep["endpoint_methods"] == {
        "START": "fallback_5pct",
        "LOCKOUT": "fallback_5pct",
    }
    assert rep["phases"]["down"][0] == rep["positions"]["START"]
    assert rep["phases"]["up"][1] == rep["positions"]["LOCKOUT"] + 1
    assert all(a < b for a, b in zip(rep["positions"].values(), list(rep["positions"].values())[1:]))


def test_model_segments_selects_slow_start_from_stationary_top_run(monkeypatch):
    monkeypatch.setattr(config, "MODEL_STOP_SPEED_RATIOS", (0.01,))
    rep = first_rep(base_velocity_hip())

    assert rep["positions"]["START"] == 18
    assert rep["endpoint_methods"]["START"] == "velocity"


def test_model_segments_selects_slow_lockout_from_stationary_top_run(monkeypatch):
    monkeypatch.setattr(config, "MODEL_STOP_SPEED_RATIOS", (0.01,))
    rep = first_rep(base_velocity_hip())

    assert rep["positions"]["LOCKOUT"] == 81
    assert rep["endpoint_methods"]["LOCKOUT"] == "velocity"


def test_model_segments_ignores_mid_ascent_pause_outside_top_band(monkeypatch):
    monkeypatch.setattr(config, "MODEL_STOP_SPEED_RATIOS", (0.01,))
    hip = np.full(110, 0.2, dtype=np.float32)
    hip[21:46] = np.linspace(0.23, 0.8, 25, dtype=np.float32)
    hip[46:51] = 0.8
    hip[51:61] = np.linspace(0.76, 0.5, 10, dtype=np.float32)
    hip[61:69] = 0.5
    hip[69:88] = np.linspace(0.47, 0.2, 19, dtype=np.float32)
    hip[88:] = 0.2

    rep = first_rep(hip, up=(51, 93))

    assert rep["positions"]["LOCKOUT"] > 69
    assert rep["positions"]["LOCKOUT"] == 88
    assert rep["endpoint_methods"]["LOCKOUT"] == "velocity"


def test_model_segments_prefers_true_start_over_higher_presets_near_setup(monkeypatch):
    monkeypatch.setattr(config, "MODEL_STOP_SPEED_RATIOS", (0.01,))
    hip = base_velocity_hip()
    hip[5:12] = 0.24

    rep = first_rep(hip)

    assert rep["positions"]["START"] == 18
    assert rep["endpoint_methods"]["START"] == "velocity"


def test_model_segments_falls_back_when_only_two_stationary_frames_are_available(monkeypatch):
    monkeypatch.setattr(config, "MODEL_SEARCH_EXTRA_FRAMES", 0)
    monkeypatch.setattr(config, "MODEL_STOP_SPEED_RATIOS", (0.01,))

    rep = first_rep(base_velocity_hip(), down=(27, 51))

    assert rep["positions"]["START"] == 20
    assert rep["endpoint_methods"]["START"] == "fallback_5pct"


def test_model_segments_uses_strict_speed_boundary_for_stationary_frames(monkeypatch):
    monkeypatch.setattr(config, "MODEL_STOP_SPEED_RATIOS", (0.005,))

    rep = first_rep(base_velocity_hip())

    assert rep["positions"]["START"] == 17
    assert rep["endpoint_methods"]["START"] == "velocity"


def test_model_segments_does_not_use_nan_frames_for_velocity_endpoint():
    hip = base_velocity_hip()
    hip[:14] = 0.45
    hip[14:22] = np.nan

    rep = first_rep(hip)

    assert rep["positions"]["START"] == 24
    assert rep["endpoint_methods"]["START"] == "fallback_5pct"


def test_model_segments_does_not_search_more_than_configured_extra_frames():
    hip = np.full(140, 0.45, dtype=np.float32)
    hip[10:19] = 0.2
    hip[50:75] = np.linspace(0.23, 0.8, 25, dtype=np.float32)
    hip[75:80] = 0.8
    hip[80:110] = np.linspace(0.78, 0.2, 30, dtype=np.float32)
    hip[110:] = 0.2

    rep = first_rep(hip, down=(60, 80), up=(80, 114))

    assert rep["positions"]["START"] == 53
    assert rep["endpoint_methods"]["START"] == "fallback_5pct"


def test_model_segments_bounds_lockout_before_next_rep_start():
    hip = np.full(150, 0.2, dtype=np.float32)
    hip[21:46] = np.linspace(0.23, 0.8, 25, dtype=np.float32)
    hip[46:51] = 0.8
    hip[51:76] = np.linspace(0.78, 0.35, 25, dtype=np.float32)
    hip[76:84] = 0.35
    hip[85:110] = np.linspace(0.23, 0.8, 25, dtype=np.float32)
    hip[110:115] = 0.8
    hip[115:140] = np.linspace(0.78, 0.2, 25, dtype=np.float32)
    hip[140:] = 0.2
    inference = model_inference(len(hip), (25, 51), (51, 82))
    inference["clean_phase_ids"][85:115] = [1] * 30
    inference["clean_phase_ids"][115:143] = [2] * 28

    result = build_model_segments(pose_from_hip(hip), video_settings(), inference)

    assert len(result["reps"]) == 2
    assert result["reps"][0]["positions"]["LOCKOUT"] < result["reps"][1]["positions"]["START"]


def test_model_segments_marks_overridden_endpoint_method():
    rep = first_rep(base_velocity_hip(), overrides={"1": {"START": 19}})

    assert rep["positions"]["START"] == 19
    assert rep["endpoint_methods"]["START"] == "override"
    assert rep["endpoint_methods"]["LOCKOUT"] == "velocity"
    assert rep["endpoint_speed_ratios"]["START"] is None
    assert rep["endpoint_speed_ratios"]["LOCKOUT"] == 0.005


@pytest.mark.parametrize("step,expected_ratio,method,start,lockout", [
    (0.003, 0.005, "velocity", 4, 10),
    (0.007, 0.01, "velocity", 4, 10),
    (0.015, 0.02, "velocity", 4, 10),
    (0.03, None, "fallback_5pct", 1, 13),
])
def test_endpoint_cascade_uses_first_successful_speed_or_height_fallback(
        step, expected_ratio, method, start, lockout):
    hip = np.array([0, step, 2 * step, 3 * step, 4 * step, 5 * step,
                    0.5, 1, 0.5, 5 * step, 4 * step, 3 * step, 2 * step, step, 0])
    candidate = {"bottom": 7, "d1": 1, "d2": 1, "pre_min": 0, "post_min": 0,
                 "fallback_start": 1, "fallback_lockout": 13}
    for position, expected_frame in (("START", start), ("LOCKOUT", lockout)):
        assert _select_endpoint(hip, hip_velocity(hip), candidate, position, 0, len(hip)) == (
            expected_frame, method, expected_ratio)


def test_endpoint_cascade_prefers_stricter_run_over_nearer_relaxed_run():
    hip = np.array([0, 0, 0, 0, 0.007, 0.014, 0.021, 0.028, 0.035, 0.5, 1])
    candidate = {"bottom": 10, "d1": 1, "pre_min": 0, "fallback_start": 4}
    assert _select_endpoint(hip, hip_velocity(hip), candidate, "START", 0, len(hip)) == (
        3, "velocity", 0.005)


def test_model_segments_records_selected_speed_ratios_in_json(monkeypatch):
    import json

    rep = first_rep(base_velocity_hip())
    assert json.loads(json.dumps(rep))["endpoint_speed_ratios"] == {
        "START": 0.005, "LOCKOUT": 0.005}
    monkeypatch.setattr(config, "MODEL_STOP_SPEED_RATIOS", (0,))
    rep = first_rep(base_velocity_hip())
    assert rep["endpoint_methods"] == {"START": "fallback_5pct", "LOCKOUT": "fallback_5pct"}
    assert rep["endpoint_speed_ratios"] == {"START": None, "LOCKOUT": None}


def test_model_segments_reject_tiny_amplitude_and_unknown_overrides():
    pose = synthetic_pose()
    pose["lms"][:, 23:25, 1] = 0.4
    inference = {
        "exercise": "squat",
        "clean_phase_ids": [0] * 10 + [1] * 20 + [2] * 26 + [0] * 8,
    }
    assert build_model_segments(pose, video_settings(), inference)["reps"] == []
    with pytest.raises(ValueError):
        build_model_segments(synthetic_pose(), video_settings(), inference, {"2": {"BOTTOM": 32}})
    with pytest.raises(ValueError):
        build_model_segments(synthetic_pose(), video_settings(), inference, {"1": {"BOTTOM": 9}})


def test_model_segments_requires_full_length_clean_phases():
    with pytest.raises(ValueError):
        build_model_segments(synthetic_pose(), video_settings(), {"exercise": "squat", "clean_phase_ids": [0]})
