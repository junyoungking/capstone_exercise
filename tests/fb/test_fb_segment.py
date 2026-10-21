"""합성 데이터만으로 1단계 포지션·방향·override·캐시를 검증한다."""

import json
import os

import numpy as np
import pytest

from feedback.pose import extract_pose
from feedback.segment import (
    apply_override,
    build_segments,
    hip_y_trajectory,
    positions_from_labels,
    smooth_hip_y,
)
import run_demo
from unified_feedback_v4 import SIDE_IDS


def cosine_hip():
    return 0.2 + 0.2 * (1 - np.cos(np.linspace(0, 2 * np.pi, 61)))


def phases(offset=0):
    return {"down": [offset, offset + 29], "bottom": [offset + 29, offset + 32],
            "up": [offset + 32, offset + 61]}


def synthetic_pose(hip, side="left", direction=-1):
    hip = np.asarray(hip)
    lms = np.zeros((len(hip), 33, 4), dtype=np.float32)
    lms[:, :, 0] = 0.5
    lms[:, :, 3] = 0.1
    lms[:, sorted(set(SIDE_IDS[side].values())), 3] = 1
    lms[:, 23:25, 1] = hip[:, None]
    for ids in SIDE_IDS.values():
        lms[:, ids["foot"], 0] += direction * 0.1
    return {"lms": lms, "valid": np.ones(len(hip), dtype=bool), "fps": 30.0,
            "w": 640, "h": 480, "total": len(hip), "cache_hit": False}


def settings(reps=None):
    return {"video": "data/videos/synthetic.mp4", "view": "side",
            "reps": [{"index": 1, "phases": phases()}] if reps is None else reps}


def test_cosine_start_lockout_bottom_and_first_mid_crossings():
    hip = hip_y_trajectory(synthetic_pose(cosine_hip()))
    positions = positions_from_labels(hip, phases())
    assert positions == {"START": 0, "MID_DOWN": 15, "BOTTOM": 30, "MID_UP": 46, "LOCKOUT": 60}
    assert positions["BOTTOM"] == 29 + np.nanargmax(hip[29:32])
    down_height = (hip[0] + hip[30]) / 2
    up_height = (hip[30] + hip[60]) / 2
    assert hip[15] >= down_height and np.all(hip[:15] < down_height)
    assert hip[46] <= up_height and np.all(hip[30:46] > up_height)
    assert all(a < b for a, b in zip(list(positions.values())[:-1], list(positions.values())[1:]))


def test_positions_are_frame_numbers_with_an_offset():
    hip = np.concatenate((np.full(10, 0.2), cosine_hip(), np.full(10, 0.2)))
    positions = positions_from_labels(hip_y_trajectory(synthetic_pose(hip)), phases(10))
    assert positions["START"] == 10
    assert positions["BOTTOM"] == 40
    assert positions["LOCKOUT"] == 70


def test_missing_bottom_and_mids_use_phase_midpoints():
    hip = np.full(61, np.nan)
    assert positions_from_labels(hip, phases()) == {
        "START": 0, "MID_DOWN": 14, "BOTTOM": 30, "MID_UP": 46, "LOCKOUT": 60,
    }


def test_missing_endpoint_uses_mid_fallback_even_with_a_detected_bottom():
    hip = hip_y_trajectory(synthetic_pose(cosine_hip()))
    hip[0] = np.nan
    hip[60] = np.nan
    positions = positions_from_labels(hip, phases())
    assert positions["MID_DOWN"] == 14
    assert positions["MID_UP"] == 46
    assert positions["BOTTOM"] == 30


def test_only_bounded_gaps_of_at_most_five_frames_are_interpolated():
    short_gap = np.arange(25, dtype=float)
    short_gap[7:12] = np.nan
    np.testing.assert_allclose(smooth_hip_y(short_gap)[7:12], np.arange(7, 12))
    long_gap = np.arange(25, dtype=float)
    long_gap[7:13] = np.nan
    assert np.isnan(smooth_hip_y(long_gap)[7:13]).all()
    edge_gaps = np.arange(25, dtype=float)
    edge_gaps[:3] = np.nan
    edge_gaps[-3:] = np.nan
    smoothed = smooth_hip_y(edge_gaps)
    assert np.isnan(smoothed[:3]).all() and np.isnan(smoothed[-3:]).all()


def test_moving_average_is_centered_and_preserves_the_short_input_length():
    impulse = np.zeros(11)
    impulse[5] = 1
    expected = np.zeros(11)
    expected[3:8] = 0.2
    np.testing.assert_allclose(smooth_hip_y(impulse), expected)
    np.testing.assert_allclose(smooth_hip_y(np.array([2.0, 4.0])), [3.0, 3.0])


def test_z_coordinates_do_not_affect_positions_side_or_facing():
    pose = synthetic_pose(cosine_hip())
    before = build_segments(pose, settings(), "squat")
    pose["lms"][:, :, 2] = np.arange(pose["total"] * 33).reshape(-1, 33) * 1000
    assert build_segments(pose, settings(), "squat") == before


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("direction, expected_facing", [(-1, "left"), (1, "right")])
def test_rep_side_and_facing_sign(side, direction, expected_facing):
    result = build_segments(synthetic_pose(cosine_hip(), side, direction), settings(), "squat")
    assert result["reps"][0]["side"] == side
    assert result["facing"] == expected_facing


def test_facing_uses_the_mean_of_five_start_frames_and_video_majority():
    pose = synthetic_pose(np.concatenate((cosine_hip(), cosine_hip(), cosine_hip())))
    left_ids = SIDE_IDS["left"]
    pose["lms"][0, left_ids["foot"], 0] = 0.8
    pose["lms"][1:5, left_ids["foot"], 0] = 0.4
    assert build_segments(pose, settings(), "squat")["facing"] == "left"
    pose["lms"][:5, left_ids["foot"], 0] = 0.6
    reps = [{"index": index + 1, "phases": phases(index * 61)} for index in range(3)]
    assert build_segments(pose, settings(reps), "squat")["facing"] == "left"


def test_missing_detections_remain_missing_after_hip_interpolation():
    pose = synthetic_pose(cosine_hip())
    pose["valid"][10:13] = False
    pose["lms"][10:13] = np.nan
    assert np.isfinite(hip_y_trajectory(pose)[10:13]).all()
    assert int(pose["valid"].sum()) == 58


def test_override_is_applied_without_changing_phases_and_marks_source():
    result = build_segments(synthetic_pose(cosine_hip()), settings(), "squat", {"1": {"BOTTOM": 31}})
    assert result["reps"][0]["positions"]["BOTTOM"] == 31
    assert result["reps"][0]["phases"] == phases()
    assert result["source"] == "labels+override"


def test_override_can_repair_base_positions_before_the_order_check():
    pose = synthetic_pose(np.full(61, 0.4))
    result = build_segments(pose, settings(), "squat", {"1": {"MID_DOWN": 15, "MID_UP": 45}})
    assert result["reps"][0]["positions"] == {
        "START": 0, "MID_DOWN": 15, "BOTTOM": 29, "MID_UP": 45, "LOCKOUT": 60,
    }


@pytest.mark.parametrize("changes", [
    {"MID_DOWN": 30}, {"MID_UP": 29}, {"BOTTOM": 15}, {"START": -1},
    {"LOCKOUT": 61}, {"BOTTOM": 30.5}, {"BOTTOM": True}, {"unknown": 30},
])
def test_override_rejects_broken_order_outside_frames_and_unknown_positions(changes):
    positions = positions_from_labels(hip_y_trajectory(synthetic_pose(cosine_hip())), phases())
    with pytest.raises(ValueError):
        apply_override(positions, changes, phases(), 61)


def test_override_rejects_unknown_rep_numbers():
    with pytest.raises(ValueError):
        build_segments(synthetic_pose(cosine_hip()), settings(), "squat", {"2": {"BOTTOM": 31}})


@pytest.mark.parametrize("bad_phases", [
    {"down": [-1, 29], "bottom": [29, 32], "up": [32, 61]},
    {"down": [0, 30], "bottom": [29, 32], "up": [32, 61]},
    {"down": [0, 29], "bottom": [29, 29], "up": [29, 61]},
    {"down": [0, 29], "bottom": [29, 32], "up": [32, 62]},
])
def test_phase_ranges_are_half_open_non_overlapping_and_within_the_video(bad_phases):
    with pytest.raises(ValueError):
        positions_from_labels(np.arange(61, dtype=float), bad_phases)


@pytest.fixture
def synthetic_extractor(tmp_path, monkeypatch):
    import export_overlay
    import offline_compare

    pose = synthetic_pose(cosine_hip())
    sequence = [frame.copy() for frame in pose["lms"]]
    sequence[7] = None
    calls = []

    def extract(path, exercise):
        calls.append((path, exercise))
        return sequence

    monkeypatch.setattr(offline_compare, "extract_sequence", extract)
    monkeypatch.setattr(export_overlay, "video_info",
                        lambda path: {"fps": 30.0, "w": 640, "h": 480, "n": 61})
    source = tmp_path / "사용자.bin"
    source.write_bytes(b"synthetic input identity")
    return source, tmp_path / "pose.npz", calls


def test_pose_cache_schema_and_reuse(synthetic_extractor):
    source, cache, calls = synthetic_extractor
    first = extract_pose(source, cache, "squat")
    saved_mtime = cache.stat().st_mtime_ns
    second = extract_pose(source, cache, "squat")
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert len(calls) == 1 and cache.stat().st_mtime_ns == saved_mtime
    with np.load(cache, allow_pickle=False) as saved:
        assert saved["lms"].shape == (61, 33, 4)
        assert saved["lms"].dtype == np.float32
        assert saved["valid"].shape == (61,) and saved["valid"].dtype == np.bool_
        assert not saved["valid"][7] and np.isnan(saved["lms"][7]).all()
        assert saved["video"].item() == source.resolve().as_posix()
        assert saved["mtime_ns"].item() == source.stat().st_mtime_ns
    np.testing.assert_equal(first["lms"], second["lms"])


def test_no_cache_and_video_mtime_changes_force_extraction(synthetic_extractor):
    source, cache, calls = synthetic_extractor
    extract_pose(source, cache, "squat")
    assert extract_pose(source, cache, "squat", no_cache=True)["cache_hit"] is False
    current = source.stat()
    os.utime(source, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000))
    assert extract_pose(source, cache, "squat")["cache_hit"] is False
    assert len(calls) == 3


def test_video_path_changes_and_corrupt_cache_force_extraction(synthetic_extractor, tmp_path):
    source, cache, calls = synthetic_extractor
    extract_pose(source, cache, "squat")
    other = tmp_path / "other.bin"
    other.write_bytes(source.read_bytes())
    os.utime(other, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns))
    assert extract_pose(other, cache, "squat")["cache_hit"] is False
    cache.write_bytes(b"not an npz file")
    assert extract_pose(other, cache, "squat")["cache_hit"] is False
    assert len(calls) == 3


def test_cli_override_applies_to_user_and_legacy_expert_without_video(tmp_path, monkeypatch):
    from feedback import gate, pose

    monkeypatch.setattr(gate, "check_inputs", lambda *args, **kwargs: {"warnings": []})

    monkeypatch.setattr(pose, "extract_pose", lambda *args, **kwargs: synthetic_pose(cosine_hip()))
    configuration = tmp_path / "demo.json"
    configuration.write_text(json.dumps({"exercise": "squat", "user": settings(), "expert": settings()}),
                             encoding="utf-8")
    overrides = tmp_path / "positions_override.json"
    overrides.write_text(json.dumps({"user": {"1": {"BOTTOM": 31}},
                                     "ref": {"1": {"LOCKOUT": 59}}}), encoding="utf-8")
    out = tmp_path / "out"
    assert run_demo.main(["--config", str(configuration), "--out", str(out), "--until", "segments",
                          "--override", str(overrides)]) == 0
    user = json.loads((out / "segments_user.json").read_text(encoding="utf-8"))
    ref = json.loads((out / "segments_ref.json").read_text(encoding="utf-8"))
    assert user["reps"][0]["positions"]["BOTTOM"] == 31
    assert ref["reps"][0]["positions"]["LOCKOUT"] == 59
    assert user["source"] == ref["source"] == "labels+override"
