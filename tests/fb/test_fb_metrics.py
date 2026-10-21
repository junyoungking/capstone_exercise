"""Synthetic, video-free checks for stage 2 measurement."""

import json
from statistics import median

import numpy as np

from feedback import config
from feedback.metrics import build_metrics
import feedback.metrics as metrics_module
from unified_feedback_v4 import SIDE_IDS, compute_metrics as original_compute_metrics


def _landmarks(knee_offset: float = 0.0, visible_side: str = "left") -> np.ndarray:
    lms = np.zeros((33, 4), dtype=np.float32)
    lms[:, 3] = 0.2
    lms[0] = [0.5, 0.1, 0.0, 1.0]
    for side, x, sign in (("left", 0.4, 1.0), ("right", 0.6, -1.0)):
        ids = SIDE_IDS[side]
        visibility = 1.0 if side == visible_side else 0.65
        for key in ids.values():
            lms[key, 3] = visibility
        lms[ids["shoulder"], :2] = [x, 0.2]
        lms[ids["elbow"], :2] = [x + sign * 0.03, 0.35]
        lms[ids["wrist"], :2] = [x + sign * 0.03, 0.5]
        lms[ids["hip"], :2] = [x, 0.5]
        lms[ids["knee"], :2] = [x + sign * knee_offset, 0.7]
        lms[ids["ankle"], :2] = [x, 0.9]
        lms[ids["heel"], :2] = [x - sign * 0.05, 0.95]
        lms[ids["foot"], :2] = [x + sign * 0.05, 0.96]
        lms[ids["index"], :2] = [x + sign * 0.04, 0.55]
        lms[ids["thumb"], :2] = [x + sign * 0.02, 0.54]
    return lms


def _pose(total: int = 12, visible_side: str = "left") -> dict:
    lms = np.stack([_landmarks(frame * 0.01, visible_side) for frame in range(total)])
    return {"lms": lms, "valid": np.ones(total, dtype=bool), "fps": 10.0,
            "w": 640, "h": 480, "total": total}


def _segments(total: int = 12, side: str = "left") -> dict:
    return {
        "video": "data/videos/synthetic.mp4",
        "fps": 10.0,
        "size": [640, 480],
        "total": total,
        "exercise": "squat",
        "view": "side",
        "facing": "left",
        "source": "labels",
        "reps": [{
            "rep": 1,
            "phases": {"down": [0, 5], "bottom": [5, 7], "up": [7, 12]},
            "positions": {"START": 0, "MID_DOWN": 2, "BOTTOM": 6, "MID_UP": 8, "LOCKOUT": 11},
            "side": side,
        }],
    }


def test_frame_metrics_use_v4_and_position_values_are_clipped_window_medians(monkeypatch):
    calls = []

    def spy(lms, exercise, side):
        calls.append((exercise, side))
        return original_compute_metrics(lms, exercise, side)

    monkeypatch.setattr(metrics_module, "compute_metrics", spy)
    pose = _pose()
    result = build_metrics(pose, _segments())
    names = config.DISPLAY_METRICS["squat"]
    assert set(result["frame_metrics"]) == set(names)
    assert all(len(result["frame_metrics"][name]) == pose["total"] for name in names)
    assert len(calls) == pose["total"]
    assert all(call == ("squat", "left") for call in calls)

    bottom = result["reps"][0]["positions"]["BOTTOM"]["values"]
    expected = [
        original_compute_metrics(pose["lms"][frame], "squat", "left")["knee_angle"]
        for frame in range(4, 9)
    ]
    assert bottom["knee_angle"] == median(expected)

    start = result["reps"][0]["positions"]["START"]["values"]
    expected_start = [
        original_compute_metrics(pose["lms"][frame], "squat", "left")["knee_angle"]
        for frame in range(0, 3)
    ]
    assert start["knee_angle"] == median(expected_start)
    json.dumps(result, allow_nan=False)


def test_position_value_is_null_when_fewer_than_three_valid_values():
    pose = _pose()
    original_valid = pose["valid"].copy()
    pose["valid"][5:8] = False
    result = build_metrics(pose, _segments())
    bottom = result["reps"][0]["positions"]["BOTTOM"]["values"]
    assert all(bottom[name] is None for name in config.DISPLAY_METRICS["squat"])
    np.testing.assert_array_equal(original_valid, np.ones_like(original_valid))
    assert pose["valid"][5:8].tolist() == [False, False, False]


def test_tempo_and_detection_boundary_do_not_exclude_at_minimum():
    pose = _pose()
    pose["valid"][:5] = [True, True, True, False, False]
    result = build_metrics(pose, _segments())
    rep = result["reps"][0]
    assert rep["tempo"] == {"down_s": 0.5, "bottom_s": 0.2, "up_s": 0.5, "ratio": 1.0}
    assert rep["detect"] == {"down": 0.6, "bottom": 1.0, "up": 1.0}
    assert rep["excluded"] is False
    assert rep["exclude_reason"] is None


def test_detection_below_minimum_excludes_with_configured_reason():
    pose = _pose()
    pose["valid"][:5] = [True, True, False, False, False]
    result = build_metrics(pose, _segments())
    rep = result["reps"][0]
    expected_reason = config.DETECTION_REASON_TEMPLATE.format(
        phase=config.PHASE_LABELS["down"], ratio=0.4, minimum=config.MIN_VALID_RATIO
    )
    assert rep["detect"]["down"] == 0.4
    assert rep["excluded"] is True
    assert rep["exclude_reason"] == expected_reason


def test_rep_window_uses_rep_side_and_outside_uses_chronological_sidelock(monkeypatch):
    captured = []

    def fake_compute(_lms, _exercise, side):
        captured.append(side)
        return {"knee_angle": float(len(captured))}

    monkeypatch.setattr(metrics_module, "compute_metrics", fake_compute)
    pose = _pose(total=60, visible_side="left")
    segments = _segments(total=60, side="right")
    segments["reps"][0]["phases"] = {"down": [20, 21], "bottom": [21, 22], "up": [22, 23]}
    segments["reps"][0]["positions"] = {"START": 20, "MID_DOWN": 21, "BOTTOM": 22,
                                         "MID_UP": 23, "LOCKOUT": 24}
    result = build_metrics(pose, segments)
    assert captured[0] == "left"
    assert captured[20] == "right"
    assert captured[45] == "left"
    assert result["frame_metrics"]["knee_angle"][20] == 21.0
    assert result["frame_metrics"]["hip_angle"][20] is None
