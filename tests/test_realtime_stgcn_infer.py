from model.realtime_stgcn_infer import OnlinePhaseCounter, PHASE_DOWN, PHASE_READY, PHASE_UP


def test_online_phase_counter_hold_frame_extends_current_segment():
    counter = OnlinePhaseCounter(smooth_window=1, min_up_len=3)

    held_before_prediction = counter.hold_frame()
    assert held_before_prediction.phase_id == PHASE_READY
    assert held_before_prediction.segment_len == 0

    down = counter.update(PHASE_DOWN)
    assert down.segment_len == 1

    held = counter.hold_frame()
    assert held.phase_id == PHASE_DOWN
    assert held.segment_len == 2

    held = counter.hold_frame()
    assert held.phase_id == PHASE_DOWN
    assert held.segment_len == 3

    up = counter.update(PHASE_UP)
    assert up.phase_id == PHASE_UP
    assert up.segment_len == 1
    assert up.count == 0


def test_online_phase_counter_hold_frame_counts_with_extended_up_segment():
    counter = OnlinePhaseCounter(smooth_window=1, min_up_len=3)

    counter.update(PHASE_DOWN)
    up = counter.update(PHASE_UP)
    assert up.segment_len == 1

    counter.hold_frame()
    held = counter.hold_frame()
    assert held.phase_id == PHASE_UP
    assert held.segment_len == 3

    ready = counter.update(PHASE_READY)
    assert ready.last_increment is True
    assert ready.count == 1
