import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

import unified_feedback_v4 as feedback


def test_choose_issue_accepts_model_phase_names_for_absolute_rules():
    squat_issue = feedback.choose_issue(
        "squat",
        {"knee_angle": 120.0, "trunk_lean": 0.0},
        {},
        {},
        "down",
    )
    assert squat_issue is not None
    assert squat_issue.key == "knee_depth"

    deadlift_issue = feedback.choose_issue(
        "deadlift",
        {"knee_angle": 120.0, "hip_angle": 180.0},
        {},
        {},
        "up",
    )
    assert deadlift_issue is not None
    assert deadlift_issue.key == "deadlift_knee_lockout"

    bench_issue = feedback.choose_issue(
        "benchpress",
        {"elbow_angle": 120.0, "wrist_elbow_x_diff": 0.0},
        {},
        {},
        "up",
    )
    assert bench_issue is not None
    assert bench_issue.key == "bench_lockout"


def test_raw_pose_detection_gate_rejects_missing_detection_after_smoothing():
    smoother = feedback.LandmarkSmoother()
    first = feedback.np.zeros((33, 4), dtype=feedback.np.float32)
    first[:, 3] = 1.0

    smoothed = smoother.update(first)
    assert feedback._has_raw_pose_detection(first) is True
    assert feedback.valid_lms(smoothed) is True

    stale = smoother.update(None)
    assert feedback._has_raw_pose_detection(None) is False
    assert feedback.valid_lms(stale) is True
