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


def _tracked(y, src='det', used=True):
    return {'plate': {'src': src, 'used': used, 'center': [100., float(y)], 'size_px': [100., 100.]}}


def test_legacy_track_motion_overrides_fragmented_detection_track():
    frames = stream([det(20, 20, size=120), det(300, 20, score=.7)],
                    [det(20, 20, size=120), det(300, 30, score=.7)],
                    [det(300, 40, score=.7)], [det(300, 60, score=.7)],
                    [det(300, 80, score=.7)])
    rows = [_tracked(y, src='det_low' if i % 2 else 'det') for i,y in enumerate((0, 20, 45, 65))]
    chosen, legacy, _ = choose_plate(frames, legacy_rows=rows)
    assert len(legacy.items) == 2
    assert chosen is legacy


def test_tracked_stationary_legacy_yields_to_confirmed_mover():
    frames = stream(*[[det(20, 20, size=120), det(300, y, score=.7)] for y in (20, 30, 40, 55, 70)])
    chosen, legacy, _ = choose_plate(frames, legacy_rows=[_tracked(y) for y in (2, 3, 3, 4)])
    assert legacy.first[2].box[0] == 20 and chosen.first[2].box[0] == 300


def test_absent_detected_evidence_is_not_stationarity_or_klt_motion():
    from core.recog.plate_pick import tracked_motion
    frames = stream(*[[det(20, 20, size=120), det(300, y, score=.7)] for y in (20, 30, 40, 55, 70)])
    rows = [_tracked(0), _tracked(5)] + [_tracked(y, 'klt') for y in (20, 40, 60, 80)]
    assert tracked_motion(rows) is None
    chosen, legacy, _ = choose_plate(frames, legacy_rows=rows)
    assert chosen is legacy


def test_weak_backtrack_stops_at_one_pixel_frame_edge():
    from core.recog.plate_pick import safe_start
    frames = stream([det(1, 20, 100, .1)], [det(2, 20, 100, .1)],
                    [det(3, 20, 100, .2)], [det(4, 20, 100, .8)])
    chosen, _, _ = choose_plate(frames)
    assert chosen.first[0] == 0
    assert safe_start(chosen, (300, 200))[0] == 1
    assert safe_start(chosen, (104, 200))[0] == 3


def test_detection_runs_once_and_progress_distinguishes_tracking_passes(monkeypatch):
    images=[np.random.default_rng(i).integers(0,255,(180,240,3),dtype=np.uint8) for i in range(5)]
    monkeypatch.setattr('server.jobs.video.frames',lambda path:iter((i*.1,img.copy()) for i,img in enumerate(images)))
    calls=[];detected=[];legacy=[];final=[]
    def detector(img,seq):
        calls.append(seq)
        return [det(20,20,70,score=.8)]
    rows=recognize('ignored',detector,rim=False,track_cfg=replace(TrackCfg(),plate_pick=True),
                   on_detect=lambda seq,t,img:detected.append(seq),
                   on_legacy=lambda seq,t,img,st:legacy.append(seq),
                   on_frame=lambda seq,t,img,st:final.append(seq))
    assert len(rows)==5 and calls==detected==legacy==list(range(5))
    assert final==[]  # the unchanged original track is reused


def test_recognize_retracks_a_confirmed_moving_alternative(monkeypatch):
    images=[np.random.default_rng(i).integers(0,255,(250,500,3),dtype=np.uint8) for i in range(6)]
    monkeypatch.setattr('server.jobs.video.frames',lambda path:iter((i*.1,img.copy()) for i,img in enumerate(images)))
    calls=[];legacy=[];final=[]
    def detector(img,seq):
        calls.append(seq)
        return [det(20,20,120),det(300,20+seq*10,100,.7)]
    rows=recognize('ignored',detector,rim=False,track_cfg=replace(TrackCfg(),plate_pick=True,pick_retry_zero=False),
                   on_legacy=lambda seq,t,img,st:legacy.append(seq),
                   on_attempt=lambda i,n,seq,t,img,st:final.append(seq))
    assert calls==legacy==final==list(range(6))
    assert rows[0]['plate']['box'][0]==300


def test_zero_rep_retries_confirmed_moving_candidate(monkeypatch):
    from server import jobs
    frames=[np.random.default_rng(i).integers(0,255,(260,500,3),dtype=np.uint8) for i in range(8)]
    monkeypatch.setattr(jobs.video,'frames',lambda path:iter((i*.1,img.copy()) for i,img in enumerate(frames)))
    calls=[]
    def detector(img,seq):
        calls.append(seq);return [det(20,20,120),det(300,20+seq*8,100,.7)]
    def completed(rows,cfg):
        first=next((r['plate']['box'][0] for r in rows if r.get('plate',{}).get('used')),0)
        return int(first>200)
    monkeypatch.setattr(jobs,'_completed_both_sides',completed)
    rows=jobs.recognize('ignored',detector,rim=False,track_cfg=replace(TrackCfg(),plate_pick=True,pick_retry_zero=True))
    assert calls==list(range(8)) and rows[0]['plate']['box'][0]==300


def test_early_safe_initialization_uses_two_video_passes(monkeypatch):
    from server import jobs
    images=[np.random.default_rng(i).integers(0,255,(160,240,3),dtype=np.uint8) for i in range(6)]
    passes=[]
    def frames(path):
        passes.append(1);return iter((i*.1,img.copy()) for i,img in enumerate(images))
    monkeypatch.setattr(jobs.video,'frames',frames)
    rows=jobs.recognize('ignored',lambda img,i:[det(20,20,70,.1 if i<2 else .8)],rim=False,
                        track_cfg=replace(TrackCfg(),plate_pick=True))
    assert rows[0]['plate']['src']=='det_low' and len(passes)==2


def test_retry_count_uses_both_rest_sides_and_fixed_diameter(monkeypatch):
    from server import jobs
    calls=[]
    def fake(rows,exercise,fps,plate_mm,cfg):
        calls.append((exercise,plate_mm));return ([{'rep_values':[{'back':1.0}]}] if exercise=='deadlift' else []),{}
    monkeypatch.setattr(jobs,'analyse',fake)
    rows=[{'t':0.,'plate':{'used':True}},{'t':1.,'plate':{'used':True}}]
    assert jobs._completed_both_sides(rows)==1
    assert calls==[('squat',450.),('deadlift',450.)]


def test_zero_rep_without_successful_alternative_keeps_original(monkeypatch):
    from server import jobs
    images=[np.random.default_rng(i).integers(0,255,(260,500,3),dtype=np.uint8) for i in range(8)]
    monkeypatch.setattr(jobs.video,'frames',lambda path:iter((i*.1,img.copy()) for i,img in enumerate(images)))
    monkeypatch.setattr(jobs,'_completed_both_sides',lambda rows,cfg:0)
    detector=lambda img,seq:[det(20,20,120),det(300,20+seq*8,100,.7)]
    rows=jobs.recognize('ignored',detector,rim=False,track_cfg=replace(TrackCfg(),plate_pick=True,pick_retry_zero=True))
    assert rows[0]['plate']['box'][0]==20


def test_final_frame_callbacks_do_not_repeat_klt(monkeypatch):
    from server import jobs
    images=[np.random.default_rng(i).integers(0,255,(160,240,3),dtype=np.uint8) for i in range(6)]
    monkeypatch.setattr(jobs.video,'frames',lambda path:iter((i*.1,img.copy()) for i,img in enumerate(images)))
    original=jobs._recognize_pass;passes=[];seen=[]
    def tracked(*args,**kw):
        passes.append(1);return original(*args,**kw)
    monkeypatch.setattr(jobs,'_recognize_pass',tracked)
    rows=jobs.recognize('ignored',lambda img,i:[det(20,20,70,.1 if i<2 else .8)],rim=False,
                        track_cfg=replace(TrackCfg(),plate_pick=True),on_frame=lambda i,t,img,st:seen.append(i))
    assert len(rows)==len(seen)==6 and len(passes)==1
