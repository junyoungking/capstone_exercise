"""Initialization must honor the caller's circle-fallback setting."""
import numpy as np

from core.recog.plate import Detection
from server import jobs


def frames(monkeypatch, n=2):
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    monkeypatch.setattr(jobs.video, 'frames', lambda path: iter((i / 30, img.copy()) for i in range(n)))


def test_rim_disabled_waits_for_model_detection(monkeypatch):
    frames(monkeypatch)
    def forbidden(*args):
        raise AssertionError('circle fallback must be disabled')
    monkeypatch.setattr(jobs, 'find_plate_circle', forbidden)
    box = (20, 20, 60, 60)
    rows = jobs.recognize('unused', detector=lambda img, seq: [] if seq == 0 else [Detection(box, .8)], rim=False)
    assert 'plate' not in rows[0]
    assert rows[1]['plate']['box'] == list(box)
    assert rows[1]['plate']['used']


def test_rim_enabled_preserves_circle_initialization(monkeypatch):
    frames(monkeypatch, 1)
    box = (20, 20, 60, 60)
    monkeypatch.setattr(jobs, 'find_plate_circle', lambda img: Detection(box, .5))
    rows = jobs.recognize('unused', rim=True)
    assert rows[0]['plate']['box'] == list(box)


def test_explicit_initial_box_works_without_rim(monkeypatch):
    frames(monkeypatch, 1)
    box = (20, 20, 60, 60)
    rows = jobs.recognize('unused', init_box=box, rim=False)
    assert rows[0]['plate']['box'] == list(box)
