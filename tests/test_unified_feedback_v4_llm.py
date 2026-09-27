import copy

import pytest

pytest.importorskip("mediapipe")

import unified_feedback_v4_llm as llm_v4


def test_metric_payload_contains_metadata_and_delta():
    user_m = {
        "knee_angle": 110.0,
        "hip_angle": 92.0,
        "trunk_lean": 24.0,
    }
    expert_m = {
        "knee_angle": 84.0,
        "hip_angle": 88.0,
        "trunk_lean": 19.0,
    }
    deltas = llm_v4.base.compute_deltas(user_m, expert_m)

    payload = llm_v4.build_llm_feedback_payload("squat", "down", user_m, expert_m, deltas)
    records = {m["name"]: m for m in payload["metrics"]}

    assert payload["schema_version"] == "metric_feedback_v1"
    assert payload["context"]["exercise"] == "squat"
    assert records["knee_angle"]["description"]
    assert records["knee_angle"]["user_value"] == 110.0
    assert records["knee_angle"]["expert_value"] == 84.0
    assert records["knee_angle"]["delta_user_minus_expert"] == 26.0
    assert records["knee_angle"]["threshold"] == 18.0
    assert records["knee_angle"]["normalized_abs_delta"] > 1.0


def test_local_feedback_selects_largest_normalized_difference():
    user_m = {
        "knee_angle": 110.0,  # delta 26 / threshold 18 = 1.44
        "hip_angle": 96.0,    # delta 8 / threshold 18 = 0.44
        "trunk_lean": 25.0,   # delta 5 / threshold 10 = 0.50
    }
    expert_m = {
        "knee_angle": 84.0,
        "hip_angle": 88.0,
        "trunk_lean": 20.0,
    }
    payload = llm_v4.build_llm_feedback_payload(
        "squat",
        "down",
        user_m,
        expert_m,
        llm_v4.base.compute_deltas(user_m, expert_m),
    )

    feedback = llm_v4.local_metric_feedback(payload)

    assert feedback["provider"] == "local"
    assert feedback["priority_metric"] == "knee_angle"
    assert "knee_angle" in feedback["selected_metrics"]
    assert feedback["severity"] in {"medium", "high"}
    assert feedback["summary"]


def test_install_variant_is_idempotent_and_adds_llm_output_suffix():
    original_config = copy.deepcopy(llm_v4.base.VIDEO_CONFIG)
    original_realtime_output = llm_v4.base.REALTIME_OUTPUT
    original_choose_issue = llm_v4.base.choose_issue
    try:
        llm_v4.install_llm_variant()
        once = [cfg["output"] for cfg in llm_v4.base.VIDEO_CONFIG.values()]
        llm_v4.install_llm_variant()
        twice = [cfg["output"] for cfg in llm_v4.base.VIDEO_CONFIG.values()]

        assert once == twice
        assert all("_llm" in output for output in once)
        assert llm_v4.base.choose_issue == llm_v4.ENGINE.choose_issue
    finally:
        llm_v4.base.VIDEO_CONFIG.clear()
        llm_v4.base.VIDEO_CONFIG.update(original_config)
        llm_v4.base.REALTIME_OUTPUT = original_realtime_output
        llm_v4.base.choose_issue = original_choose_issue
