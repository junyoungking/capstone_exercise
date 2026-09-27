# -*- coding: utf-8 -*-
"""ST-GCN phase/action bridge for offline feedback rendering.

이 모듈은 `unified_feedback_v4.py`가 모델 phase를 perception backbone으로
쓸 수 있도록, 모델 추론 계약과 순수 phase/rep/alignment 헬퍼를 분리한다.
MediaPipe는 import하지 않으므로 순수 단위 테스트와 checkpoint smoke가 가볍게
동작한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


PHASE_READY = 0
PHASE_DOWN = 1
PHASE_UP = 2
PHASE_NAMES = ("ready", "down", "up")


@dataclass
class PhaseResult:
    phase_per_frame: np.ndarray
    action: str
    phase_conf: Optional[np.ndarray] = None
    action_conf: float = 0.0


@dataclass(frozen=True)
class PhaseSeg:
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))


@dataclass(frozen=True)
class Rep:
    index: int
    down: Optional[PhaseSeg]
    up: Optional[PhaseSeg]
    bottom_frame: Optional[int]
    top_frame: Optional[int]

    @property
    def duration(self) -> int:
        total = 0
        if self.down is not None:
            total += self.down.length
        if self.up is not None:
            total += self.up.length
        return total


@dataclass(frozen=True)
class FramePhaseMeta:
    rep_index: int
    phase: str
    progress: float


def phase_name(phase_id: int | str | None) -> str:
    if isinstance(phase_id, str):
        return phase_id if phase_id in PHASE_NAMES else "ready"
    if phase_id is None:
        return "ready"
    idx = int(phase_id)
    if 0 <= idx < len(PHASE_NAMES):
        return PHASE_NAMES[idx]
    return "ready"


def _as_phase_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int16).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.int8)
    arr = np.where((0 <= arr) & (arr <= 2), arr, PHASE_READY)
    return arr.astype(np.int8)


def _fill_nearest(labels: np.ndarray, default: int = PHASE_READY) -> np.ndarray:
    """Fill -1 gaps with the nearest assigned label."""
    arr = np.asarray(labels).copy()
    if arr.size == 0:
        return arr.astype(np.int8)
    valid = np.flatnonzero(arr >= 0)
    if valid.size == 0:
        return np.full(arr.shape, int(default), dtype=np.int8)

    all_idx = np.arange(arr.size)
    right_pos = np.searchsorted(valid, all_idx, side="left")
    right = valid[np.clip(right_pos, 0, valid.size - 1)]
    left = valid[np.clip(right_pos - 1, 0, valid.size - 1)]
    choose_left = np.abs(all_idx - left) <= np.abs(right - all_idx)
    nearest = np.where(choose_left, left, right)
    return arr[nearest].astype(np.int8)


def _fill_nearest_float(values: np.ndarray, valid_mask: np.ndarray, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if arr.size == 0:
        return arr
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return np.full(arr.shape, float(default), dtype=np.float32)
    all_idx = np.arange(arr.size)
    right_pos = np.searchsorted(valid, all_idx, side="left")
    right = valid[np.clip(right_pos, 0, valid.size - 1)]
    left = valid[np.clip(right_pos - 1, 0, valid.size - 1)]
    choose_left = np.abs(all_idx - left) <= np.abs(right - all_idx)
    nearest = np.where(choose_left, left, right)
    return arr[nearest].astype(np.float32)


def windows_to_frame_labels(
    per_window_label: Sequence[int] | np.ndarray,
    T: int,
    clip_len: int,
    stride: int,
    anchor: str = "last",
) -> np.ndarray:
    """Map sparse causal/window labels to dense per-frame labels.

    `anchor="last"` matches the existing causal training dataset where each
    window target is the end frame. `center` is retained for experiments whose
    labels were centered.
    """
    total = max(0, int(T))
    labels = np.full(total, -1, dtype=np.int16)
    if total == 0:
        return labels.astype(np.int8)
    if int(clip_len) <= 0 or int(stride) <= 0:
        raise ValueError("clip_len and stride must be positive")
    if anchor not in {"center", "last", "start"}:
        raise ValueError(f"unsupported anchor={anchor!r}; expected center/last/start")

    offset = 0
    if anchor == "center":
        offset = int(clip_len) // 2
    elif anchor == "last":
        offset = int(clip_len) - 1

    for i, lab in enumerate(_as_phase_array(per_window_label)):
        frame_idx = i * int(stride) + offset
        if 0 <= frame_idx < total:
            labels[frame_idx] = int(lab)
    return _fill_nearest(labels)


def _rle(values: Sequence[int] | np.ndarray) -> list[tuple[int, int, int]]:
    arr = _as_phase_array(values)
    if arr.size == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(arr[0])
    for i in range(1, arr.size):
        value = int(arr[i])
        if value != current:
            runs.append((current, start, i))
            current = value
            start = i
    runs.append((current, start, arr.size))
    return runs


def _longer_neighbor_label(runs: list[tuple[int, int, int]], idx: int) -> int:
    left = runs[idx - 1] if idx > 0 else None
    right = runs[idx + 1] if idx < len(runs) - 1 else None
    if left is None and right is None:
        return runs[idx][0]
    if left is None:
        return right[0]  # type: ignore[index]
    if right is None:
        return left[0]
    if left[0] == right[0]:
        return left[0]
    left_len = left[2] - left[1]
    right_len = right[2] - right[1]
    return left[0] if left_len >= right_len else right[0]


def debounce_phase(phase: Sequence[int] | np.ndarray, min_len: int = 4) -> np.ndarray:
    """Absorb very short phase runs into the longer adjacent run."""
    p = _as_phase_array(phase)
    if p.size == 0 or int(min_len) <= 1:
        return p.copy()

    changed = True
    guard = 0
    while changed and guard <= p.size * 2:
        guard += 1
        changed = False
        runs = _rle(p)
        for idx, (_lab, start, end) in enumerate(runs):
            if end - start < int(min_len) and len(runs) > 1:
                p[start:end] = _longer_neighbor_label(runs, idx)
                changed = True
                break
    return p.astype(np.int8)


def smooth_phase_like_training(phase_seq: Iterable[int], window: int = 5) -> np.ndarray:
    """Match `model/train_ablation.py::smooth_phase`.

    The phase experiments count reps after a simple majority-window smoothing
    pass over model phase labels.  Keep this helper pure so the feedback script
    can use the same counting semantics without importing the training module.
    """
    phase_seq = _as_phase_array(np.asarray(list(phase_seq), dtype=np.int64))
    if int(window) <= 1 or len(phase_seq) == 0:
        return phase_seq.copy()
    h = int(window) // 2
    smoothed = np.zeros_like(phase_seq)
    for i in range(len(phase_seq)):
        w = phase_seq[max(0, i - h) : min(len(phase_seq), i + h + 1)]
        vals, counts = np.unique(w, return_counts=True)
        smoothed[i] = vals[counts.argmax()]
    return smoothed.astype(np.int8)


def count_phases_like_training(phase_seq: Iterable[int], min_up_len: int = 3) -> tuple[int, list[int]]:
    """Match `model/train_ablation.py::count_phases`.

    Important: the experiment counter counts each sufficiently long UP segment
    when that UP segment ends. It does *not* require an adjacent DOWN->UP pair.
    """
    phase_seq = _as_phase_array(np.asarray(list(phase_seq), dtype=np.int64))
    count, transitions = 0, []
    in_up, up_start = False, -1
    for t, p in enumerate(phase_seq):
        if int(p) == PHASE_UP:
            if not in_up:
                in_up, up_start = True, t
        else:
            if in_up:
                if t - up_start >= int(min_up_len):
                    count += 1
                    transitions.append(t)
                in_up = False
    return count, transitions


def bridge_ready_gaps(phase: Sequence[int] | np.ndarray, max_gap: int = 30) -> np.ndarray:
    """Bridge short READY gaps that split active model phases.

    This is model-output postprocessing, not metric fallback: it only rewrites
    READY runs when they are bracketed by model-predicted active phases.

    Rules:
    - DOWN READY UP with a short READY gap becomes DOWN UP by assigning the gap
      to DOWN, preserving the bottom transition at the first UP frame.
    - X READY X with a short READY gap becomes one continuous X run.
    - UP READY DOWN is left as READY because it can be a true top/rest boundary
      between reps.
    """
    p = _as_phase_array(phase).copy()
    gap = max(0, int(max_gap))
    if p.size == 0 or gap <= 0:
        return p

    runs = _rle(p)
    for idx in range(1, len(runs) - 1):
        lab, start, end = runs[idx]
        if lab != PHASE_READY or (end - start) > gap:
            continue
        prev_lab = runs[idx - 1][0]
        next_lab = runs[idx + 1][0]
        if prev_lab not in {PHASE_DOWN, PHASE_UP} or next_lab not in {PHASE_DOWN, PHASE_UP}:
            continue
        if prev_lab == next_lab:
            p[start:end] = prev_lab
        elif prev_lab == PHASE_DOWN and next_lab == PHASE_UP:
            p[start:end] = PHASE_DOWN
    return p.astype(np.int8)


def segment_reps(phase: Sequence[int] | np.ndarray) -> list[Rep]:
    """Return complete reps defined by adjacent `down -> up` runs."""
    runs = _rle(phase)
    reps: list[Rep] = []
    i = 0
    while i < len(runs) - 1:
        lab, start, end = runs[i]
        next_lab, next_start, next_end = runs[i + 1]
        if lab == PHASE_DOWN and next_lab == PHASE_UP and end == next_start:
            reps.append(
                Rep(
                    index=len(reps),
                    down=PhaseSeg(start, end),
                    up=PhaseSeg(next_start, next_end),
                    bottom_frame=end,
                    top_frame=max(next_start, next_end - 1),
                )
            )
            i += 2
        else:
            i += 1
    return reps


def segment_up_reps(phase: Sequence[int] | np.ndarray, min_up_len: int = 3) -> list[Rep]:
    """Return reps represented by UP-only runs.

    Deadlift reference clips often contain only the concentric pull
    (floor -> lockout).  That is still model phase evidence, but it cannot be
    represented as a complete DOWN->UP pair.  The returned Rep keeps `down=None`
    and uses the UP segment as the reference motion.
    """
    reps: list[Rep] = []
    for lab, start, end in _rle(phase):
        if lab != PHASE_UP or end - start < int(min_up_len):
            continue
        reps.append(
            Rep(
                index=len(reps),
                down=None,
                up=PhaseSeg(start, end),
                bottom_frame=start,
                top_frame=max(start, end - 1),
            )
        )
    return reps


def rep_to_dict(rep: Rep) -> dict[str, Any]:
    return {
        "index": int(rep.index),
        "down": None if rep.down is None else [int(rep.down.start), int(rep.down.end)],
        "up": None if rep.up is None else [int(rep.up.start), int(rep.up.end)],
        "bottom": None if rep.bottom_frame is None else int(rep.bottom_frame),
        "top": None if rep.top_frame is None else int(rep.top_frame),
    }


def rep_from_dict(data: Mapping[str, Any]) -> Rep:
    def seg(value: Any) -> Optional[PhaseSeg]:
        if not isinstance(value, Sequence) or len(value) != 2:
            return None
        return PhaseSeg(int(value[0]), int(value[1]))

    return Rep(
        index=int(data.get("index", 0)),
        down=seg(data.get("down")),
        up=seg(data.get("up")),
        bottom_frame=None if data.get("bottom") is None else int(data.get("bottom")),
        top_frame=None if data.get("top") is None else int(data.get("top")),
    )


def pick_reference_rep(
    reps: Sequence[Rep],
    frames: Optional[Sequence[Any]] = None,
    preferred_index: Optional[int] = None,
    allow_single_phase: bool = False,
) -> Optional[Rep]:
    """Pick explicit valid rep or the complete rep closest to median duration."""
    usable = [
        r
        for r in reps
        if r.duration > 0 and (r.down is not None and r.up is not None or allow_single_phase and r.up is not None)
    ]
    if not usable:
        return None

    if preferred_index is not None:
        preferred = int(preferred_index)
        for rep in usable:
            if rep.index == preferred:
                return rep
        if 0 <= preferred < len(usable):
            return usable[preferred]

    complete = [r for r in usable if r.down is not None and r.up is not None]
    if allow_single_phase and usable and not complete:
        if frames:
            center = (len(frames) - 1) / 2.0
        else:
            center = float(np.mean([r.bottom_frame or 0 for r in usable]))
        return max(
            usable,
            key=lambda r: (
                r.duration,
                -abs(float(r.bottom_frame or 0) - center),
                -r.index,
            ),
        )

    durations = np.asarray([r.duration for r in usable], dtype=np.float32)
    median_duration = float(np.median(durations))
    if frames:
        center = (len(frames) - 1) / 2.0
    else:
        center = float(np.mean([r.bottom_frame or 0 for r in usable]))

    return min(
        usable,
        key=lambda r: (
            abs(float(r.duration) - median_duration),
            abs(float(r.bottom_frame or 0) - center),
            r.index,
        ),
    )


def build_user_frame_meta(
    phase: Sequence[int] | np.ndarray,
    reps: Sequence[Rep],
) -> list[Optional[FramePhaseMeta]]:
    """Mark model down/up frames with per-phase progress in [0, 1].

    Complete reps are filled first so rep-indexed counting stays stable. Extra
    model down/up runs outside a complete rep are still valid phase evidence for
    visual alignment, so they are filled with rep_index=-1 instead of being
    treated as ready/hold frames.
    """
    phase_arr = _as_phase_array(phase)
    total = len(phase_arr)
    meta: list[Optional[FramePhaseMeta]] = [None] * total

    def fill(rep_index: int, seg: Optional[PhaseSeg], name: str) -> None:
        if seg is None or seg.length <= 0:
            return
        denom = max(seg.length - 1, 1)
        for frame_idx in range(max(0, seg.start), min(total, seg.end)):
            progress = (frame_idx - seg.start) / float(denom)
            meta[frame_idx] = FramePhaseMeta(rep_index, name, float(np.clip(progress, 0.0, 1.0)))

    for rep in reps:
        fill(rep.index, rep.down, "down")
        fill(rep.index, rep.up, "up")

    for label, start, end in _rle(phase_arr):
        if label not in {PHASE_DOWN, PHASE_UP}:
            continue
        seg = PhaseSeg(start, end)
        name = "down" if label == PHASE_DOWN else "up"
        denom = max(seg.length - 1, 1)
        for frame_idx in range(max(0, seg.start), min(total, seg.end)):
            if meta[frame_idx] is not None:
                continue
            progress = (frame_idx - seg.start) / float(denom)
            meta[frame_idx] = FramePhaseMeta(-1, name, float(np.clip(progress, 0.0, 1.0)))
    return meta


def _frame_from_progress(seg: PhaseSeg, progress: float) -> int:
    if seg.length <= 0:
        return int(seg.start)
    idx = int(round(seg.start + float(np.clip(progress, 0.0, 1.0)) * (seg.length - 1)))
    return max(seg.start, min(idx, seg.end - 1))


def _progress_from_observed_len(observed_len: int, reference_len: int) -> float:
    ref = max(int(reference_len) - 1, 1)
    obs = max(int(observed_len) - 1, 0)
    return float(np.clip(obs / float(ref), 0.0, 1.0))


def align_realtime_phase_to_expert(
    phase_id: int | str | None,
    segment_len: int,
    expert_ref_rep: Rep,
) -> int:
    """Map the current online model phase to one expert reference frame.

    Offline alignment knows the complete user phase duration. Realtime only
    knows how long the current predicted phase has been active, so it estimates
    progress against the expert reference segment length and saturates at the
    segment end. For deadlift up-only expert refs, DOWN is mapped over the same
    UP segment in reverse.
    """
    phase = phase_name(phase_id)
    top = int(
        expert_ref_rep.top_frame
        if expert_ref_rep.top_frame is not None
        else expert_ref_rep.up.end - 1
        if expert_ref_rep.up is not None
        else 0
    )

    if phase == "down":
        if expert_ref_rep.down is not None:
            progress = _progress_from_observed_len(segment_len, expert_ref_rep.down.length)
            return _frame_from_progress(expert_ref_rep.down, progress)
        if expert_ref_rep.up is not None:
            progress = _progress_from_observed_len(segment_len, expert_ref_rep.up.length)
            return _frame_from_progress(expert_ref_rep.up, 1.0 - progress)
    if phase == "up" and expert_ref_rep.up is not None:
        progress = _progress_from_observed_len(segment_len, expert_ref_rep.up.length)
        return _frame_from_progress(expert_ref_rep.up, progress)
    if expert_ref_rep.down is not None:
        return int(expert_ref_rep.down.start)
    return top


def build_phase_alignment_map(
    user_frame_meta: Sequence[Optional[FramePhaseMeta]],
    user_reps: Sequence[Rep],
    expert_ref_rep: Rep,
    align: str = "ratio",
    user_feats: Optional[np.ndarray] = None,
    expert_feats: Optional[np.ndarray] = None,
) -> list[Optional[int]]:
    """Map each user frame to an expert frame from the same ref-rep phase.

    Down/up frames are phase-progress aligned. Ready frames outside complete
    reps are still model-phase frames, so map them to deterministic top/ready
    anchors instead of falling back to time-based expert matching.
    """
    _ = (user_feats, expert_feats)  # reserved for PR4 DTW.
    if align != "ratio":
        raise ValueError("Only phase-segmented ratio alignment is supported in PR1-PR3; DTW is out of scope.")

    usable_user = [r for r in user_reps if r.duration > 0 and (r.down is not None or r.up is not None)]
    if not usable_user:
        raise ValueError("user_reps must contain at least one usable model-phase rep")
    if expert_ref_rep.up is None:
        raise ValueError("expert_ref_rep must contain at least an UP segment")

    active = [i for i, meta in enumerate(user_frame_meta) if meta is not None]
    first_user_active = active[0] if active else min(
        int(seg.start)
        for r in usable_user
        for seg in (r.down, r.up)
        if seg is not None
    )
    expert_ready_start = int(expert_ref_rep.down.start if expert_ref_rep.down is not None else expert_ref_rep.up.end - 1)
    expert_ready_top = int(
        expert_ref_rep.top_frame
        if expert_ref_rep.top_frame is not None
        else max(expert_ref_rep.up.start, expert_ref_rep.up.end - 1)
    )
    expert_has_down = expert_ref_rep.down is not None

    out: list[Optional[int]] = []
    for frame_idx, meta in enumerate(user_frame_meta):
        if meta is None:
            out.append(expert_ready_start if frame_idx < first_user_active and expert_has_down else expert_ready_top)
            continue
        if meta.phase == "down" and expert_has_down:
            out.append(_frame_from_progress(expert_ref_rep.down, meta.progress))
        elif meta.phase == "down" and expert_ref_rep.up is not None:
            # Deadlift reference clips may contain only the pull (UP) phase.
            # Reuse that model-predicted UP segment in reverse so the user's
            # lowering/down frames still align to matching bottom/top poses.
            out.append(_frame_from_progress(expert_ref_rep.up, 1.0 - meta.progress))
        elif meta.phase == "up" and expert_ref_rep.up is not None:
            out.append(_frame_from_progress(expert_ref_rep.up, meta.progress))
        else:
            out.append(None)
    return out


def num_reps_completed_until(reps: Sequence[Rep], frame_idx: int) -> int:
    """Count reps whose up segment has ended at or before the given frame."""
    frame = int(frame_idx)
    count = 0
    for rep in reps:
        if rep.up is not None and rep.up.end - 1 <= frame:
            count += 1
    return count


def pose_seq_from_landmarks(lms_seq: Iterable[Any], num_kpt: int = 33) -> np.ndarray:
    """Convert `unified_feedback_v4` landmark arrays to `[T,33,4]` float32."""
    frames: list[np.ndarray] = []
    for lms in lms_seq:
        arr = np.asarray(lms, dtype=np.float32) if lms is not None else None
        if arr is None or arr.ndim != 2 or arr.shape[0] < num_kpt or arr.shape[1] < 4:
            frames.append(np.zeros((num_kpt, 4), dtype=np.float32))
        else:
            frames.append(arr[:num_kpt, :4].astype(np.float32, copy=True))
    if not frames:
        return np.zeros((0, num_kpt, 4), dtype=np.float32)
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


class PhaseModelAdapter:
    """Offline ST-GCN adapter that returns dense per-frame phase labels."""

    def __init__(
        self,
        ckpt_path: Optional[str | Path] = None,
        device: str = "auto",
        clip_len: Optional[int] = None,
        stride: int = 2,
        input_kind: Optional[str] = None,
        graph: str = "mediapipe33",
        anchor: str = "last",
        required: bool = False,
    ):
        self.ckpt_path = ckpt_path
        self.device_request = device
        self.clip_len_override = clip_len
        self.stride = max(1, int(stride))
        self.input_kind = input_kind
        self.graph = graph
        self.anchor = anchor
        self.required = bool(required)
        self.available = False
        self.unavailable_reason: Optional[str] = "not initialized"
        self.warning: Optional[str] = None
        self.model: Any = None
        self.cfg: dict[str, Any] = {}
        self.device: Any = "cpu"
        self.clip_len: Optional[int] = None
        self.resolved_checkpoint: Optional[Path] = None
        self._rt: Any = None

        try:
            self._initialize()
        except Exception as exc:
            self.available = False
            self.unavailable_reason = str(exc)
            if self.required:
                raise

    def _resolve_device(self, torch_module: Any) -> str:
        if self.device_request == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if self.device_request == "cuda" and not torch_module.cuda.is_available():
            return "cpu"
        return str(self.device_request)

    def _initialize(self) -> None:
        from model import realtime_stgcn_infer as rt

        rt._require_numpy()
        rt._require_torch()
        torch = rt.torch
        if torch is None:
            raise rt.RealtimeModelUnavailable("torch unavailable")

        selected, warning = rt.resolve_checkpoint_path(self.ckpt_path)
        if selected is None:
            raise rt.RealtimeModelUnavailable(warning or "checkpoint not found")
        if not selected.exists():
            raise rt.RealtimeModelUnavailable(f"checkpoint not found: {selected}")

        device = self._resolve_device(torch)
        ckpt = rt._torch_load(selected, map_location=device)
        cfg = rt.validate_checkpoint_metadata(ckpt, clip_len_override=self.clip_len_override)
        if self.input_kind:
            requested = rt.normalize_derivative_mode(self.input_kind)
            if requested != cfg.get("derivative_mode"):
                raise rt.RealtimeModelUnavailable(
                    f"checkpoint derivative_mode={cfg.get('derivative_mode')!r} "
                    f"does not match requested input_kind={requested!r}"
                )

        model = rt.build_model_from_cfg(cfg, device=device)
        state_dict = rt._strip_module_prefix(rt._state_dict_from_checkpoint(ckpt))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self._rt = rt
        self.model = model
        self.cfg = cfg
        self.device = device
        self.clip_len = int(cfg["clip_len"])
        self.warning = warning
        self.resolved_checkpoint = selected
        self.available = True
        self.unavailable_reason = None

    def _kpts_from_pose_seq(self, pose_seq: np.ndarray) -> np.ndarray:
        arr = np.asarray(pose_seq, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] < 33 or arr.shape[2] < 3:
            raise ValueError(f"expected pose_seq shape [T,33,4] or [T,33,>=3], got {arr.shape}")
        # unified_feedback stores [x, y, z, visibility]; model input expects [x, y, visibility].
        if arr.shape[2] >= 4:
            return arr[:, :33, [0, 1, 3]].astype(np.float32, copy=True)
        return arr[:, :33, :3].astype(np.float32, copy=True)

    def infer(self, pose_seq: np.ndarray) -> PhaseResult:
        if not self.available or self.model is None or self.clip_len is None or self._rt is None:
            raise RuntimeError(self.unavailable_reason or "phase model unavailable")

        rt = self._rt
        torch = rt.torch
        F = rt.F
        if torch is None or F is None:
            raise RuntimeError("torch unavailable")

        kpts = self._kpts_from_pose_seq(pose_seq)
        T = int(kpts.shape[0])
        if T == 0:
            return PhaseResult(np.zeros(0, dtype=np.int8), "squat", None, 0.0)

        norm = rt.normalize_kpts(kpts)
        clip_len = int(self.clip_len)
        stride = max(1, int(self.stride))
        ends = [T - 1] if T < clip_len else list(range(clip_len - 1, T, stride))
        if T >= clip_len and ends[-1] != T - 1:
            ends.append(T - 1)
        ends = sorted(set(int(e) for e in ends))

        window_labels: list[int] = []
        window_conf: list[float] = []
        action_probs: list[np.ndarray] = []

        for end in ends:
            start = end - clip_len + 1
            if start >= 0:
                clip = norm[start : end + 1]
            else:
                clip = np.concatenate([np.tile(norm[0:1], (-start, 1, 1)), norm[: end + 1]], axis=0)
            features = rt.build_pose_input_features(clip, self.cfg.get("derivative_mode", "pose"))
            x = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).contiguous()
            x = x.to(self.device)
            with torch.no_grad():
                action_logit, phase_logit = self.model(x)
                action_prob = F.softmax(action_logit, dim=1).detach().cpu().numpy()[0]
                phase_prob = F.softmax(phase_logit, dim=1).detach().cpu().numpy()[0]
            phase_id = int(phase_prob.argmax())
            window_labels.append(phase_id)
            window_conf.append(float(phase_prob[phase_id]))
            action_probs.append(action_prob.astype(np.float32))

        # `windows_to_frame_labels` assumes starts=i*stride. For end-anchored
        # causal windows with a forced final window, fill explicit anchor frames
        # to avoid drift when the final end is not on the stride grid.
        sparse = np.full(T, -1, dtype=np.int16)
        sparse_conf = np.zeros(T, dtype=np.float32)
        valid_conf = np.zeros(T, dtype=bool)
        for end, lab, conf in zip(ends, window_labels, window_conf):
            anchor_frame = end
            if self.anchor == "center":
                anchor_frame = max(0, end - clip_len // 2)
            elif self.anchor == "start":
                anchor_frame = max(0, end - clip_len + 1)
            sparse[anchor_frame] = int(lab)
            sparse_conf[anchor_frame] = float(conf)
            valid_conf[anchor_frame] = True

        phase_per_frame = _fill_nearest(sparse)
        phase_conf = _fill_nearest_float(sparse_conf, valid_conf)
        mean_action = np.mean(np.stack(action_probs), axis=0) if action_probs else np.eye(3, dtype=np.float32)[0]
        action_id = int(mean_action.argmax())
        action = rt.CLASS_LIST[action_id]
        return PhaseResult(
            phase_per_frame=phase_per_frame.astype(np.int8),
            action=str(action),
            phase_conf=phase_conf.astype(np.float32),
            action_conf=float(mean_action[action_id]),
        )


__all__ = [
    "PHASE_READY",
    "PHASE_DOWN",
    "PHASE_UP",
    "PHASE_NAMES",
    "PhaseResult",
    "PhaseSeg",
    "Rep",
    "FramePhaseMeta",
    "PhaseModelAdapter",
    "windows_to_frame_labels",
    "debounce_phase",
    "smooth_phase_like_training",
    "count_phases_like_training",
    "bridge_ready_gaps",
    "segment_reps",
    "segment_up_reps",
    "rep_to_dict",
    "rep_from_dict",
    "pick_reference_rep",
    "align_realtime_phase_to_expert",
    "build_user_frame_meta",
    "build_phase_alignment_map",
    "num_reps_completed_until",
    "pose_seq_from_landmarks",
    "phase_name",
]
