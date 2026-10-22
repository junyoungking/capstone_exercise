"""Offline bridge to the existing realtime ST-GCN adapter."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feedback import config


def _adapter():
    from model import realtime_stgcn_infer as rt

    return rt


def _pose_kpts(pose: dict) -> np.ndarray:
    lms = np.asarray(pose["lms"], dtype=np.float32)
    valid = np.asarray(pose["valid"], dtype=bool)
    if lms.ndim != 3 or lms.shape[1] != 33 or lms.shape[2] < 4:
        raise ValueError({"pose_lms_shape": list(lms.shape)})
    kpts = np.zeros((len(lms), 33, 3), dtype=np.float32)
    kpts[valid, :, 0] = lms[valid, :, 0]
    kpts[valid, :, 1] = lms[valid, :, 1]
    kpts[valid, :, 2] = lms[valid, :, 3]
    return kpts


def _last_frame_windows(kpts: np.ndarray, clip_len: int) -> np.ndarray:
    if clip_len < 1:
        raise ValueError({"clip_len": clip_len})
    if len(kpts) == 0:
        return np.zeros((0, clip_len, 33, 3), dtype=np.float32)
    padded = np.concatenate(
        [np.repeat(kpts[:1], clip_len - 1, axis=0), kpts],
        axis=0,
    )
    return np.stack([padded[i:i + clip_len] for i in range(len(kpts))], axis=0)


def _softmax(logits: Any) -> np.ndarray:
    rt = _adapter()
    return rt.F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)


def infer_phases(pose: dict, checkpoint: str | Path) -> dict:
    """Run the repo's ST-GCN checkpoint and attach each window to its last frame."""

    rt = _adapter()
    try:
        infer = rt.RealtimeSTGCNInfer(
            checkpoint_path=checkpoint,
            device="cpu",
            inference_interval=1,
            required=True,
        )
    except rt.RealtimeModelUnavailable as exc:
        raise ValueError(config.MODEL_UNAVAILABLE_TEMPLATE.format(reason=exc)) from exc
    except RuntimeError as exc:
        raise ValueError(config.MODEL_UNAVAILABLE_TEMPLATE.format(reason=exc)) from exc
    if not infer.available or infer.model is None or infer.clip_len is None:
        raise ValueError(config.MODEL_UNAVAILABLE_TEMPLATE.format(
            reason=infer.unavailable_reason or "ST-GCN unavailable"
        ))
    if int(infer.clip_len) != config.MODEL_CLIP_LENGTH:
        raise ValueError(config.MODEL_UNAVAILABLE_TEMPLATE.format(
            reason=f"checkpoint clip_len={infer.clip_len}, expected {config.MODEL_CLIP_LENGTH}"
        ))

    kpts = _pose_kpts(pose)
    windows = _last_frame_windows(kpts, int(infer.clip_len))
    if len(windows):
        normalized = np.stack([rt.normalize_kpts(window) for window in windows], axis=0)
        features = np.stack([
            rt.build_pose_input_features(window, infer.cfg.get("derivative_mode", rt.REALTIME_DERIVATIVE_MODE))
            for window in normalized
        ], axis=0)
        action_batches = []
        phase_batches = []
        previous_threads = rt.torch.get_num_threads()
        rt.torch.set_num_threads(config.MODEL_INFERENCE_THREADS)
        try:
            with rt.torch.no_grad():
                for start in range(0, len(features), config.MODEL_INFERENCE_BATCH_SIZE):
                    batch = features[start:start + config.MODEL_INFERENCE_BATCH_SIZE]
                    x = rt.torch.from_numpy(batch).permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
                    x = x.to(infer.device)
                    action_logit, phase_logit = infer.model(x)
                    action_batches.append(_softmax(action_logit))
                    phase_batches.append(_softmax(phase_logit))
        finally:
            rt.torch.set_num_threads(previous_threads)
        action_probs = np.concatenate(action_batches, axis=0)
        phase_probs = np.concatenate(phase_batches, axis=0)
    else:
        action_probs = np.zeros((0, len(rt.CLASS_LIST)), dtype=np.float32)
        phase_probs = np.zeros((0, len(rt.PHASE_NAMES)), dtype=np.float32)

    action_ids = np.argmax(action_probs, axis=1).astype(int) if len(action_probs) else np.zeros(0, dtype=int)
    phase_ids = np.argmax(phase_probs, axis=1).astype(int) if len(phase_probs) else np.zeros(0, dtype=int)
    valid = np.asarray(pose["valid"], dtype=bool)
    valid_action_ids = action_ids[valid[:len(action_ids)]]
    action_counts = Counter(int(value) for value in valid_action_ids)
    total_votes = sum(action_counts.values())
    if action_counts:
        exercise_id, exercise_count = action_counts.most_common(1)[0]
        exercise = rt.CLASS_LIST[exercise_id]
        exercise_confidence = exercise_count / total_votes
    else:
        exercise = None
        exercise_confidence = 0.0

    clean = clean_phases(phase_ids.tolist())
    return {
        "source": "model",
        "checkpoint": str(Path(checkpoint)),
        "model": {
            "class_names": list(rt.CLASS_LIST),
            "phase_names": list(rt.PHASE_NAMES),
            "clip_len": int(infer.clip_len),
            "cfg": dict(infer.cfg),
            "device": str(infer.device),
            "warning": infer.warning,
        },
        "exercise": exercise,
        "exercise_ratio": float(exercise_confidence),
        "votes": {name: int(action_counts.get(index, 0)) for index, name in enumerate(rt.CLASS_LIST)},
        "ratios": {
            name: float(action_counts.get(index, 0) / total_votes) if total_votes else 0.0
            for index, name in enumerate(rt.CLASS_LIST)
        },
        "action_ids": action_ids.tolist(),
        "action_probs": action_probs.tolist(),
        "phase_ids": phase_ids.tolist(),
        "phase_names": [rt.PHASE_NAMES[int(value)] for value in phase_ids],
        "phase_probs": phase_probs.tolist(),
        "clean_phase_ids": clean["phase_ids"],
        "clean_phase_names": clean["phase_names"],
        "cleaning": clean["cleaning"],
    }


def _median_filter(values: Sequence[int], width: int) -> list[int]:
    radius = width // 2
    output: list[int] = []
    for index in range(len(values)):
        window = values[max(0, index - radius):min(len(values), index + radius + 1)]
        output.append(int(np.median(np.asarray(window, dtype=np.int16))))
    return output


def _runs(values: Sequence[int]) -> list[dict[str, int]]:
    if not values:
        return []
    runs: list[dict[str, int]] = []
    start = 0
    current = int(values[0])
    for index, value in enumerate(values[1:], start=1):
        value = int(value)
        if value != current:
            runs.append({"phase_id": current, "start": start, "end": index})
            start = index
            current = value
    runs.append({"phase_id": current, "start": start, "end": len(values)})
    return runs


def _expand_runs(runs: Sequence[dict[str, int]], total: int) -> list[int]:
    output = [0] * total
    for run in runs:
        for index in range(run["start"], run["end"]):
            output[index] = int(run["phase_id"])
    return output


def _absorb_short_runs(values: Sequence[int], min_frames: int) -> tuple[list[int], list[dict[str, Any]]]:
    runs = _runs(values)
    notes: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        next_runs: list[dict[str, int]] = []
        for index, run in enumerate(runs):
            length = run["end"] - run["start"]
            if length >= min_frames or len(runs) == 1:
                next_runs.append(dict(run))
                continue
            left = next_runs[-1] if next_runs else None
            right = runs[index + 1] if index + 1 < len(runs) else None
            replacement = None
            if left is not None and right is not None and left["phase_id"] == right["phase_id"]:
                replacement = left["phase_id"]
            elif left is not None:
                replacement = left["phase_id"]
            elif right is not None:
                replacement = right["phase_id"]
            if replacement is None:
                next_runs.append(dict(run))
                continue
            notes.append({
                "type": "short_run_absorbed",
                "span": [run["start"], run["end"]],
                "from": int(run["phase_id"]),
                "to": int(replacement),
            })
            run = {**run, "phase_id": int(replacement)}
            if next_runs and next_runs[-1]["phase_id"] == run["phase_id"]:
                next_runs[-1]["end"] = run["end"]
            else:
                next_runs.append(run)
            changed = True
        runs = []
        for run in next_runs:
            if runs and runs[-1]["phase_id"] == run["phase_id"]:
                runs[-1]["end"] = run["end"]
            else:
                runs.append(run)
    return _expand_runs(runs, len(values)), notes


def _enforce_allowed(values: Sequence[int]) -> tuple[list[int], list[dict[str, Any]]]:
    allowed = {0: {0, 1}, 1: {1, 2}, 2: {2, 0, 1}}
    runs = _runs(values)
    notes: list[dict[str, Any]] = []
    for index in range(1, len(runs)):
        previous = int(runs[index - 1]["phase_id"])
        current = int(runs[index]["phase_id"])
        if current not in allowed[previous]:
            runs[index]["phase_id"] = previous
            notes.append({
                "type": "invalid_transition_replaced",
                "span": [runs[index]["start"], runs[index]["end"]],
                "from": current,
                "to": previous,
            })
    merged: list[dict[str, int]] = []
    for run in runs:
        if merged and merged[-1]["phase_id"] == run["phase_id"]:
            merged[-1]["end"] = run["end"]
        else:
            merged.append(run)
    return _expand_runs(merged, len(values)), notes


def clean_phases(phases: Sequence[int | str]) -> dict:
    """Apply the 6-stage smoothing rules to raw ready/down/up predictions."""

    rt = _adapter()
    ids = []
    for phase in phases:
        if isinstance(phase, str):
            ids.append(rt.PHASE_NAMES.index(phase))
        else:
            ids.append(int(phase))
    for phase_id in ids:
        if phase_id < 0 or phase_id >= len(rt.PHASE_NAMES):
            raise ValueError({"phase_id": phase_id, "allowed": list(range(len(rt.PHASE_NAMES)))})
    width = config.MODEL_PHASE_FILTER_WINDOW
    min_frames = config.MODEL_MIN_PHASE_FRAMES
    filtered = _median_filter(ids, int(width))
    absorbed, absorb_notes = _absorb_short_runs(filtered, int(min_frames))
    allowed, transition_notes = _enforce_allowed(absorbed)
    return {
        "phase_ids": allowed,
        "phase_names": [rt.PHASE_NAMES[value] for value in allowed],
        "cleaning": {
            "median_window": int(width),
            "min_run_frames": int(min_frames),
            "notes": absorb_notes + transition_notes,
        },
    }
