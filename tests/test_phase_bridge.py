import numpy as np
import pytest

from phase_bridge import (
    PHASE_DOWN,
    PHASE_READY,
    PHASE_UP,
    PhaseSeg,
    Rep,
    align_realtime_phase_to_expert,
    build_phase_alignment_map,
    build_user_frame_meta,
    bridge_ready_gaps,
    count_phases_like_training,
    debounce_phase,
    num_reps_completed_until,
    pick_reference_rep,
    rep_from_dict,
    rep_to_dict,
    segment_reps,
    segment_up_reps,
    smooth_phase_like_training,
    windows_to_frame_labels,
)


def test_windows_to_frame_labels_last_and_center_nearest_fill():
    labels = windows_to_frame_labels([PHASE_DOWN, PHASE_UP], T=8, clip_len=4, stride=2, anchor="last")
    assert labels.tolist() == [PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_UP, PHASE_UP, PHASE_UP]

    centered = windows_to_frame_labels([PHASE_DOWN, PHASE_UP], T=7, clip_len=4, stride=2, anchor="center")
    assert centered.tolist() == [PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_UP, PHASE_UP, PHASE_UP]


def test_debounce_phase_absorbs_short_flicker():
    raw = np.array([PHASE_READY, PHASE_DOWN, PHASE_DOWN, PHASE_UP, PHASE_DOWN, PHASE_DOWN, PHASE_UP, PHASE_UP])
    smoothed = debounce_phase(raw, min_len=2)
    assert smoothed.tolist() == [PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_UP, PHASE_UP]


def test_bridge_ready_gaps_counts_down_ready_up_without_metric_fallback():
    raw = [
        PHASE_READY,
        PHASE_DOWN,
        PHASE_DOWN,
        PHASE_READY,
        PHASE_READY,
        PHASE_UP,
        PHASE_UP,
        PHASE_READY,
    ]
    bridged = bridge_ready_gaps(raw, max_gap=2)
    assert bridged.tolist() == [
        PHASE_READY,
        PHASE_DOWN,
        PHASE_DOWN,
        PHASE_DOWN,
        PHASE_DOWN,
        PHASE_UP,
        PHASE_UP,
        PHASE_READY,
    ]
    reps = segment_reps(bridged)
    assert len(reps) == 1
    assert reps[0].down == PhaseSeg(1, 5)
    assert reps[0].up == PhaseSeg(5, 7)


def test_bridge_ready_gaps_keeps_top_rest_between_reps():
    raw = [PHASE_UP, PHASE_UP, PHASE_READY, PHASE_READY, PHASE_DOWN, PHASE_DOWN]
    bridged = bridge_ready_gaps(raw, max_gap=2)
    assert bridged.tolist() == raw


def test_training_count_logic_counts_up_segments_not_down_up_pairs():
    # This mirrors model/train_ablation.py: smooth_phase(...), then
    # count_phases(...). It counts a long UP run even when READY separates
    # DOWN from UP, unlike alignment segmentation.
    raw = [
        PHASE_READY,
        PHASE_DOWN,
        PHASE_DOWN,
        PHASE_READY,
        PHASE_UP,
        PHASE_UP,
        PHASE_UP,
        PHASE_READY,
    ]
    phase = smooth_phase_like_training(raw, window=1)
    count, transitions = count_phases_like_training(phase, min_up_len=3)
    assert count == 1
    assert transitions == [7]
    assert segment_reps(raw) == []


def test_segment_reps_complete_down_up_only():
    phase = [PHASE_READY, PHASE_DOWN, PHASE_DOWN, PHASE_UP, PHASE_UP, PHASE_READY, PHASE_DOWN]
    reps = segment_reps(phase)
    assert len(reps) == 1
    rep = reps[0]
    assert rep.down == PhaseSeg(1, 3)
    assert rep.up == PhaseSeg(3, 5)
    assert rep.bottom_frame == 3
    assert rep.top_frame == 4


def test_segment_up_reps_supports_deadlift_concentric_only_reference():
    phase = [PHASE_UP, PHASE_UP, PHASE_UP, PHASE_READY, PHASE_UP, PHASE_UP]
    reps = segment_up_reps(phase, min_up_len=3)
    assert len(reps) == 1
    assert reps[0].down is None
    assert reps[0].up == PhaseSeg(0, 3)
    assert reps[0].bottom_frame == 0
    assert reps[0].top_frame == 2
    assert pick_reference_rep(reps) is None
    assert pick_reference_rep(reps, allow_single_phase=True) == reps[0]

    noisy = [
        Rep(0, None, PhaseSeg(0, 47), 0, 46),
        Rep(1, None, PhaseSeg(109, 119), 109, 118),
    ]
    assert pick_reference_rep(noisy, frames=[None] * 119, allow_single_phase=True).index == 0


def test_pick_reference_rep_explicit_and_median():
    reps = [
        Rep(0, PhaseSeg(0, 2), PhaseSeg(2, 4), 2, 3),
        Rep(1, PhaseSeg(5, 9), PhaseSeg(9, 13), 9, 12),
        Rep(2, PhaseSeg(14, 17), PhaseSeg(17, 20), 17, 19),
    ]
    assert pick_reference_rep(reps, preferred_index=1).index == 1
    # Durations are 4, 8, 6; median duration is 6, so rep 2 wins.
    assert pick_reference_rep(reps).index == 2


def test_rep_serialization_roundtrip():
    rep = Rep(3, PhaseSeg(10, 15), PhaseSeg(15, 21), 15, 20)
    assert rep_from_dict(rep_to_dict(rep)) == rep


def test_build_user_frame_meta_and_count():
    reps = [Rep(0, PhaseSeg(2, 5), PhaseSeg(5, 8), 5, 7)]
    meta = build_user_frame_meta([PHASE_READY] * 10, reps)
    assert meta[1] is None
    assert meta[2].phase == "down"
    assert meta[2].progress == 0.0
    assert meta[4].progress == 1.0
    assert meta[5].phase == "up"
    assert meta[7].progress == 1.0
    assert num_reps_completed_until(reps, 6) == 0
    assert num_reps_completed_until(reps, 7) == 1


def test_build_user_frame_meta_uses_unpaired_model_phase_runs_for_alignment():
    reps = [Rep(0, PhaseSeg(6, 8), PhaseSeg(8, 10), 8, 9)]
    phase = [
        PHASE_READY,
        PHASE_DOWN,
        PHASE_DOWN,
        PHASE_READY,
        PHASE_UP,
        PHASE_UP,
        PHASE_DOWN,
        PHASE_DOWN,
        PHASE_UP,
        PHASE_UP,
    ]
    meta = build_user_frame_meta(phase, reps)
    assert meta[0] is None
    assert meta[1].phase == "down"
    assert meta[1].rep_index == -1
    assert meta[4].phase == "up"
    assert meta[4].rep_index == -1
    assert meta[6].rep_index == 0


def test_build_phase_alignment_map_ratio_boundary():
    user_reps = [Rep(0, PhaseSeg(0, 3), PhaseSeg(3, 6), 3, 5)]
    user_meta = build_user_frame_meta([PHASE_DOWN] * 3 + [PHASE_UP] * 3, user_reps)
    expert = Rep(0, PhaseSeg(10, 14), PhaseSeg(14, 18), 14, 17)
    mapping = build_phase_alignment_map(user_meta, user_reps, expert, align="ratio")
    assert mapping[:3] == [10, 12, 13]
    assert mapping[3:] == [14, 16, 17]
    # User bottom frame maps to expert down->up boundary exactly at the first UP frame.
    assert mapping[3] == expert.up.start


def test_build_phase_alignment_map_maps_ready_to_ref_anchors():
    user_reps = [Rep(0, PhaseSeg(1, 3), PhaseSeg(3, 5), 3, 4)]
    user_meta = build_user_frame_meta(
        [PHASE_READY, PHASE_DOWN, PHASE_DOWN, PHASE_UP, PHASE_UP, PHASE_READY],
        user_reps,
    )
    expert = Rep(0, PhaseSeg(10, 12), PhaseSeg(12, 14), 12, 13)
    mapping = build_phase_alignment_map(user_meta, user_reps, expert, align="ratio")
    assert mapping == [10, 10, 11, 12, 13, 13]


def test_build_phase_alignment_map_maps_down_reverse_when_expert_is_up_only():
    user_reps = [Rep(0, PhaseSeg(1, 4), PhaseSeg(4, 7), 4, 6)]
    user_meta = build_user_frame_meta(
        [PHASE_READY, PHASE_DOWN, PHASE_DOWN, PHASE_DOWN, PHASE_UP, PHASE_UP, PHASE_UP, PHASE_READY],
        user_reps,
    )
    expert = Rep(0, None, PhaseSeg(10, 13), 10, 12)
    mapping = build_phase_alignment_map(user_meta, user_reps, expert, align="ratio")
    # Ready frames map to lockout/top. User down maps over the same expert UP
    # reference in reverse; user up maps forward.
    assert mapping == [12, 12, 11, 10, 10, 11, 12, 12]


def test_align_realtime_phase_to_expert_complete_and_up_only():
    complete = Rep(0, PhaseSeg(10, 13), PhaseSeg(13, 16), 13, 15)
    assert align_realtime_phase_to_expert(PHASE_READY, 1, complete) == 10
    assert align_realtime_phase_to_expert(PHASE_DOWN, 1, complete) == 10
    assert align_realtime_phase_to_expert(PHASE_DOWN, 3, complete) == 12
    assert align_realtime_phase_to_expert(PHASE_UP, 2, complete) == 14

    up_only = Rep(0, None, PhaseSeg(20, 24), 20, 23)
    assert align_realtime_phase_to_expert(PHASE_READY, 1, up_only) == 23
    assert align_realtime_phase_to_expert(PHASE_DOWN, 1, up_only) == 23
    assert align_realtime_phase_to_expert(PHASE_DOWN, 4, up_only) == 20
    assert align_realtime_phase_to_expert(PHASE_UP, 4, up_only) == 23


def test_build_phase_alignment_map_rejects_dtw_for_pr3():
    rep = Rep(0, PhaseSeg(0, 2), PhaseSeg(2, 4), 2, 3)
    meta = build_user_frame_meta([PHASE_DOWN, PHASE_DOWN, PHASE_UP, PHASE_UP], [rep])
    with pytest.raises(ValueError, match="DTW"):
        build_phase_alignment_map(meta, [rep], rep, align="dtw")



def test_unified_missing_expert_fail_fast_source_contract():
    """MediaPipe is absent in CI shell, so assert the no-fallback contract statically."""
    source = __import__("pathlib").Path("unified_feedback_v4.py").read_text(encoding="utf-8")
    start = source.index("def build_expert_profile")
    end = source.index("# ============================================================\n# 8.", start)
    body = source[start:end]
    assert "raise FileNotFoundError" in body
    assert "return [], 30.0" not in body
    assert "_empty_expert_phase_meta" not in body
    assert "_require_expert_phase_meta" in body


def test_checkpoint_loader_safe_boundary_source_contract():
    source = __import__("pathlib").Path("model/realtime_stgcn_infer.py").read_text(encoding="utf-8")
    start = source.index("def _torch_load")
    end = source.index("def _state_dict_from_checkpoint", start)
    body = source[start:end]
    assert "weights_only=True" in body
    assert "TRUST_TORCH_CHECKPOINT" in body
    assert "weights_only=False" in body
