from dataclasses import replace
import json
import numpy as np
from core.config import TrackCfg
from core.recog.plate import Detection
from core.recog.plate_pick import choose_plate
from server.jobs import recognize


def det(x, y, size=100, score=.8):
    return Detection((x, y, x + size, y + size), score)


def stream(*rows):
    return [(i, i * .1, list(ds)) for i, ds in enumerate(rows)]


def test_weak_prefix_requires_later_confirmation():
    chosen, legacy, _ = choose_plate(stream([det(0, 0, score=.1)], [det(1, 1, score=.2)], [det(2, 2, score=.8)]))
    assert chosen is legacy
    assert chosen.first[0] == 0 and chosen.first_high(TrackCfg())[0] == 2


def test_weak_stationary_objects_do_not_start():
    chosen, legacy, _ = choose_plate(stream(*[[det(0, 0, score=.1)] for _ in range(20)]))
    assert chosen is legacy is None


def test_moving_confirmed_plate_beats_stationary_plate():
    frames = stream(*[[det(0, 0, size=120), det(300, y, score=.7)] for y in (0, 10, 20, 35, 50)])
    chosen, legacy, _ = choose_plate(frames)
    assert legacy.first[2].box[0] == 0
    assert chosen.first[2].box[0] == 300


def test_same_bar_moving_plates_keep_larger_initial_choice():
    frames = stream(*[[det(0, y, size=140), det(300, y, size=100)] for y in (0, 10, 20, 35, 50)])
    chosen, legacy, _ = choose_plate(frames)
    assert chosen is legacy and chosen.first[2].box[0] == 0


def test_no_moving_plate_keeps_legacy():
    frames = stream(*[[det(0, 0, size=120), det(300, 0)] for _ in range(6)])
    chosen, legacy, _ = choose_plate(frames)
    assert chosen is legacy and chosen.first[2].box[0] == 0


def test_disabled_and_unchanged_picks_preserve_row_bytes(monkeypatch):
    images = [np.random.default_rng(i).integers(0, 255, (160, 240, 3), dtype=np.uint8) for i in range(8)]
    monkeypatch.setattr('server.jobs.video.frames', lambda path: iter((i*.1, im.copy()) for i, im in enumerate(images)))
    frames = stream(*[[det(20, 20, size=70)] for _ in images])
    calls = []
    def detector(im, seq):
        calls.append(seq)
        return frames[seq][2]
    off = recognize('ignored', detector, rim=False, track_cfg=replace(TrackCfg(), plate_pick=False))
    assert calls == list(range(len(images)))
    calls.clear()
    on = recognize('ignored', detector, rim=False, track_cfg=replace(TrackCfg(), plate_pick=True))
    assert calls == list(range(len(images)))  # only the first pass runs inference
    assert json.dumps(off, separators=(',', ':')).encode() == json.dumps(on, separators=(',', ':')).encode()


def test_recognize_starts_at_confirmed_weak_prefix(monkeypatch):
    images = [np.random.default_rng(i).integers(0, 255, (160, 240, 3), dtype=np.uint8) for i in range(4)]
    monkeypatch.setattr('server.jobs.video.frames', lambda path: iter((i*.1, im.copy()) for i, im in enumerate(images)))
    frames = stream([det(20, 20, size=70, score=.1)], [det(21, 20, size=70, score=.15)],
                    [det(22, 20, size=70, score=.8)], [det(23, 20, size=70, score=.8)])
    calls=[]
    def detector(im, seq):
        calls.append(seq)
        return frames[seq][2]
    rows = recognize('ignored', detector, rim=False, track_cfg=replace(TrackCfg(), plate_pick=True))
    assert calls == list(range(4))
    assert rows[0]['plate']['src'] == 'det_low' and rows[0]['plate']['used']
    assert rows[0]['pick'] == 0


def test_rejects_spatial_or_size_discontinuity():
    frames = stream([det(0, 0, score=.1)], [det(200, 0, score=.8)])
    chosen, _, _ = choose_plate(frames)
    assert chosen.first[0] == 1


def test_weak_only_candidate_is_experimental():
    frames = stream(*[[det(0, i * 5, score=.15)] for i in range(20)])
    assert choose_plate(frames)[0] is None
    chosen, legacy, _ = choose_plate(frames, replace(TrackCfg(), plate_pick_weak=True))
    assert chosen is not None and legacy is None and chosen.first_high(TrackCfg()) is None
