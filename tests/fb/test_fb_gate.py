"""Synthetic checks for stage 7 input gates and readable failures."""

from __future__ import annotations

import numpy as np
import pytest

from feedback import config
from feedback.gate import (
    GateFailure,
    check_comparable,
    check_detection,
    check_exercises,
    check_inputs,
    check_label_ranges,
    check_reference,
    classify_view,
    friendly_error,
    observe_angle,
)
import feedback.gate as gate_module


def _settings(**overrides):
    data = {
        "exercise": "squat",
        "user": {"video": "user.mp4", "view": "side"},
        "ref": {"video": "ref.mp4", "view": "side", "rep": 1},
    }
    for key, value in overrides.items():
        if key in ("user", "ref"):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return data


def _patch_video(monkeypatch, probe=None, cv=None):
    probe = {
        "codec": "h264",
        "profile": "High",
        "pix_fmt": "yuv420p",
        "w": 640,
        "h": 480,
        "r_fps": 30.0,
        "avg_fps": 30.0,
        "nb_frames": 10,
        "rotation": 0,
    } if probe is None else probe
    cv = {"fps": 30.0, "w": 640, "h": 480, "count": 10, "actual": 10} if cv is None else cv
    monkeypatch.setattr(gate_module.check_video, "ffprobe", lambda _path: dict(probe))
    monkeypatch.setattr(gate_module.check_video, "opencv_info", lambda _path: dict(cv))


@pytest.mark.parametrize(
    "probe, needle",
    [
        ({"codec": "hevc", "profile": "Main", "pix_fmt": "yuv420p", "w": 640, "h": 480,
          "r_fps": 30.0, "avg_fps": 30.0, "rotation": 0}, "지원되지 않습니다"),
        ({"codec": "h264", "profile": "High 10", "pix_fmt": "yuv420p10le", "w": 640, "h": 480,
          "r_fps": 30.0, "avg_fps": 30.0, "rotation": 0}, "10bit"),
        ({"codec": "h264", "profile": "High", "pix_fmt": "yuv420p10be", "w": 640, "h": 480,
          "r_fps": 30.0, "avg_fps": 30.0, "rotation": 0}, "10bit"),
        ({"codec": "h264", "profile": "High", "pix_fmt": "yuv420p", "w": 640, "h": 480,
          "r_fps": 30.0, "avg_fps": 24.0, "rotation": 0}, "fps가 일정하지 않습니다"),
        ({"codec": "h264", "profile": "High", "pix_fmt": "yuv420p", "w": 640, "h": 480,
          "r_fps": 30.0, "avg_fps": 30.0, "rotation": 90}, "회전 정보"),
        ({"codec": "h264", "profile": "High", "pix_fmt": "yuv420p", "w": 1280, "h": 720,
          "r_fps": 30.0, "avg_fps": 30.0, "rotation": 0}, "크기가 다릅니다"),
    ],
)
def test_video_gate_fails_first_check_for_browser_and_frame_problems(monkeypatch, probe, needle):
    _patch_video(monkeypatch, probe=probe)

    with pytest.raises(GateFailure) as error:
        check_inputs(_settings())

    assert error.value.check == 1
    assert needle in error.value.reason
    assert "{" not in error.value.reason


def test_video_gate_warns_when_ffprobe_is_missing_and_unknown_codec_warns(monkeypatch):
    cv = {"fps": 30.0, "w": 640, "h": 480, "count": 10, "actual": 10}
    calls = []

    def ffprobe(_path):
        calls.append(1)
        if len(calls) == 1:
            return {"_error": "ffprobe 가 없습니다. ffmpeg 를 설치하세요."}
        return {"codec": "mystery", "profile": "", "pix_fmt": "yuv420p", "w": 640, "h": 480,
                "r_fps": 30.0, "avg_fps": 30.0, "rotation": 0}

    monkeypatch.setattr(gate_module.check_video, "ffprobe", ffprobe)
    monkeypatch.setattr(gate_module.check_video, "opencv_info", lambda _path: dict(cv))

    result = check_inputs(_settings())

    assert any("ffprobe" in warning and "건너뛰었습니다" in warning for warning in result["warnings"])
    assert any("mystery" in warning and "확인하지 못했습니다" in warning for warning in result["warnings"])


def test_video_paths_are_resolved_like_cli_project_path(monkeypatch):
    seen = []
    _patch_video(monkeypatch)

    def opencv_info(path):
        seen.append(path)
        return {"fps": 30.0, "w": 640, "h": 480, "count": 10, "actual": 10}

    monkeypatch.setattr(gate_module.check_video, "opencv_info", opencv_info)

    result = check_inputs(_settings(user={"video": "data/videos/user.mp4"}, ref={"video": "data/videos/ref.mp4"}))

    assert seen[0] == config.PROJECT_ROOT / "data/videos/user.mp4"
    assert seen[1] == config.PROJECT_ROOT / "data/videos/ref.mp4"
    assert result["videos"]["user"]["video"] == (config.PROJECT_ROOT / "data/videos/user.mp4").as_posix()


def test_video_gate_fails_when_opencv_cannot_read_or_reads_zero(monkeypatch):
    _patch_video(monkeypatch, cv={"_error": "OpenCV 가 열지 못했습니다"})
    with pytest.raises(GateFailure) as error:
        check_inputs(_settings())
    assert error.value.check == 1
    assert "열 수 없습니다" in error.value.reason

    _patch_video(monkeypatch, cv={"fps": 30.0, "w": 640, "h": 480, "count": 0, "actual": 0})
    with pytest.raises(GateFailure) as empty:
        check_inputs(_settings())
    assert empty.value.check == 1
    assert "0개" in empty.value.reason


def test_exercise_and_view_mismatch_are_second_and_third_failures(monkeypatch):
    _patch_video(monkeypatch)
    with pytest.raises(GateFailure) as exercise:
        check_inputs(_settings(user={"exercise": "squat"}, ref={"exercise": "deadlift"}))
    assert exercise.value.check == 2
    assert "종목이 다릅니다" in exercise.value.reason

    with pytest.raises(GateFailure) as view:
        check_inputs(_settings(ref={"view": "front"}))
    assert view.value.check == 3
    assert "촬영 각도 설정이 다릅니다" in view.value.reason


def test_model_precheck_only_runs_video_gate_and_labels_keep_exercise_before_view(monkeypatch):
    _patch_video(monkeypatch)
    settings = _settings(user={"exercise": "squat"}, ref={"exercise": "deadlift", "view": "front"})

    model = check_inputs(settings, mode="model")
    assert model["warnings"] == []
    assert set(model["videos"]) == {"user", "ref"}

    with pytest.raises(GateFailure) as labels:
        check_inputs(settings, mode="labels")
    assert labels.value.check == 2
    assert "종목이 다릅니다" in labels.value.reason


def test_missing_model_exercise_defers_to_detection_gate():
    assert check_exercises(None, "squat") is None
    assert check_exercises("squat", None) is None


def test_detection_gate_uses_point_six_boundary():
    pose = {"valid": np.array([True, True, True, False, False]), "total": 5}
    assert check_detection("user", pose)["ratio"] == pytest.approx(0.6)

    pose["valid"][2] = False
    with pytest.raises(GateFailure) as error:
        check_detection("user", pose)
    assert error.value.check == 4
    assert "2/5프레임(40.0%)" in error.value.reason


def _measured(excluded=False, value=1.0):
    names = config.DISPLAY_METRICS["squat"]
    return {
        "exercise": "squat",
        "reps": [{
            "rep": 1,
            "excluded": excluded,
            "exclude_reason": "하강 구간 관절 검출 40% (기준 60%)" if excluded else None,
            "detect": {phase: 1.0 for phase in config.PHASES},
            "positions": {
                position: {"frame": index, "values": {metric: value for metric in names}}
                for index, position in enumerate(config.POSITIONS)
            },
        }],
    }


def test_reference_and_user_comparable_failures_are_fifth_check():
    with pytest.raises(GateFailure) as missing:
        check_reference({"exercise": "squat", "reps": []}, ref_rep=1)
    assert missing.value.check == 5
    assert "레퍼런스 rep 1" in missing.value.reason

    with pytest.raises(GateFailure) as ref_excluded:
        check_reference(_measured(excluded=True), ref_rep=1)
    assert ref_excluded.value.check == 5
    assert "비교에서 제외" in ref_excluded.value.reason

    with pytest.raises(GateFailure) as no_user:
        check_comparable({"reps": [{"rep": 1, "excluded": True}]}, _measured(), ref_rep=1)
    assert no_user.value.check == 5
    assert "비교 가능한 사용자 rep이 0/1개" in no_user.value.reason


def _pose_for_ratio(ratio: float, valid=True):
    lms = np.zeros((1, 33, 4), dtype=np.float32)
    lms[0, :, 3] = 1.0
    lms[0, 11, :2] = [0.5 - ratio / 2, 0.2]
    lms[0, 12, :2] = [0.5 + ratio / 2, 0.2]
    lms[0, 23, :2] = [0.5, 0.7]
    lms[0, 24, :2] = [0.5, 0.7]
    return {"lms": lms, "valid": np.array([valid]), "total": 1}


def test_angle_classification_boundaries_and_configured_view_warning():
    assert classify_view(0.349) == "side"
    assert classify_view(0.35) == "diagonal"
    assert classify_view(0.60) == "diagonal"
    assert classify_view(0.601) == "front"

    segments = {"view": "side", "reps": [{"rep": 1, "positions": {"START": 0}}]}
    observed = observe_angle("user", _pose_for_ratio(0.7), segments)
    assert observed["view"] == "front"
    assert observed["ratio"] == pytest.approx(0.7 / 0.5)
    assert "설정 측면" in observed["warning"]

    missing = observe_angle("user", _pose_for_ratio(0.2, valid=False), segments)
    assert missing["ratio"] is None
    assert "자동 확인하지 못했습니다" in missing["warning"]


def test_label_range_failure_is_readable_and_not_raw_dict():
    bad = {"reps": [{"index": 1, "phases": {"down": [-1, 3], "bottom": [3, 4], "up": [4, 8]}}]}

    with pytest.raises(GateFailure) as error:
        check_label_ranges("user", bad, total=8)

    assert error.value.check == 1
    assert error.value.reason == "사용자 rep 1의 하강 구간 시작(-1)이 영상 범위(0~7) 밖입니다."
    assert "{" not in error.value.reason and "span" not in error.value.reason


def test_friendly_error_keeps_phase_numbers_and_formats_positions_without_raw_dicts():
    phase = friendly_error(ValueError({"phase": "down", "span": [-1, 231], "previous_end": 0, "total": 910}))
    assert phase == "사용자 rep 1의 하강 구간 시작(-1)이 영상 범위(0~909) 밖입니다."
    assert "{" not in phase and "span" not in phase

    positions = {"START": 10, "MID_DOWN": 12, "BOTTOM": 11, "MID_UP": 20, "LOCKOUT": 30}
    order = friendly_error(ValueError({"positions": positions}), role="ref")
    assert "START 10, MID_DOWN 12, BOTTOM 11, MID_UP 20, LOCKOUT 30" in order
    assert "{" not in order and "}" not in order

    bounds = friendly_error(ValueError({"positions": positions, "bounds": [5, 28]}), role="user")
    assert "수정 범위(5~27)" in bounds
    assert "LOCKOUT 30" in bounds
    assert "{" not in bounds and "}" not in bounds
