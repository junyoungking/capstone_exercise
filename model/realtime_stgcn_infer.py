"""Realtime ST-GCN inference adapter.

Side-effect-free runtime adapter for the multitask ST-GCN checkpoints trained by
the training script.  This module intentionally does *not* import that
training script because it performs label/cache work at import time.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:  # Keep basic import/count helpers usable in minimal shells.
    import numpy as np  # type: ignore
    _NUMPY_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - depends on local environment
    np = None  # type: ignore
    _NUMPY_IMPORT_ERROR = exc

try:  # Caught import: default-shell environments may not have torch installed.
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    _TORCH_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - depends on local environment
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc


CLASS_LIST = ["squat", "benchpress", "deadlift"]
PHASE_NAMES = ["ready", "down", "up"]
PHASE_READY = 0
PHASE_DOWN = 1
PHASE_UP = 2
REALTIME_DERIVATIVE_MODE = "pose"
REALTIME_INPUT_CHANNELS = 3
DERIVATIVE_INPUT_CHANNELS = {
    "pose": 3,
    "velocity": 5,
    "acceleration": 5,
    "velocity_acceleration": 7,
}
NUM_KPT = 33
NUM_CLASSES = len(CLASS_LIST)
NUM_PHASES = len(PHASE_NAMES)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEST_CHECKPOINT = (
    REPO_ROOT
    / "phase_experiments"
    / "temporal_head_full_ablation"
    / "mlp_h128_c16_pw_2.0_ts2_do03_aug1"
    / "best_ep013.pt"
)
DEFAULT_LATEST_CHECKPOINT = DEFAULT_BEST_CHECKPOINT.with_name("latest.pt")


# ══════════════════════════════════════════════════════════════
#  PyCharm Run 설정
# ══════════════════════════════════════════════════════════════
# 이 파일을 PyCharm에서 직접 실행할 때는 아래 값만 수정하면 됩니다.
# import 시에는 실행되지 않으므로 realtime_compare_side_infer.py 연동에는 영향이 없습니다.
PYCHARM_LAUNCH_REALTIME_APP = True  # True면 realtime_compare_side_infer.py GUI를 실행
# GUI 입력/체크포인트/웹캠 설정은 realtime_compare_side_infer.py 상단 설정을 사용합니다.
# 아래 PYCHARM_* 값들은 PYCHARM_LAUNCH_REALTIME_APP=False일 때 smoke test에만 사용됩니다.
PYCHARM_CHECKPOINT_PATH: Optional[str | os.PathLike[str]] = None
PYCHARM_DEVICE = "auto"  # "auto", "cpu", "cuda"
PYCHARM_CLIP_LEN_OVERRIDE: Optional[int] = None
PYCHARM_SMOOTH_WINDOW = 5
PYCHARM_MIN_UP_LEN = 3
PYCHARM_INFERENCE_INTERVAL = 1
PYCHARM_READY_BRIDGE_MAX_LEN = 30
PYCHARM_REQUIRED = False
PYCHARM_SMOKE_FRAMES: Optional[int] = None  # None이면 checkpoint clip_len만큼 테스트
PYCHARM_PRINT_EVERY_FRAME = False


class RealtimeModelUnavailable(RuntimeError):
    """Raised when the realtime model cannot be used safely."""


def _require_numpy() -> None:
    if np is None:
        raise RealtimeModelUnavailable(f"numpy unavailable: {_NUMPY_IMPORT_ERROR}")


@dataclass
class InferenceState:
    available: bool = False
    ready: bool = False
    pred_class: Optional[str] = None
    pred_class_id: Optional[int] = None
    action_confidence: float = 0.0
    action_probs: Optional[np.ndarray] = None
    phase: str = "ready"
    phase_id: int = PHASE_READY
    phase_confidence: float = 0.0
    phase_probs: Optional[np.ndarray] = None
    count: int = 0
    last_increment: bool = False
    segment_len: int = 0
    checkpoint_path: Optional[str] = None
    clip_len: Optional[int] = None
    warning: Optional[str] = None
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "ready": self.ready,
            "pred_class": self.pred_class,
            "pred_class_id": self.pred_class_id,
            "action_confidence": self.action_confidence,
            "action_probs": None if self.action_probs is None else self.action_probs.tolist(),
            "phase": self.phase,
            "phase_id": self.phase_id,
            "phase_confidence": self.phase_confidence,
            "phase_probs": None if self.phase_probs is None else self.phase_probs.tolist(),
            "count": self.count,
            "last_increment": self.last_increment,
            "segment_len": self.segment_len,
            "checkpoint_path": self.checkpoint_path,
            "clip_len": self.clip_len,
            "warning": self.warning,
            "unavailable_reason": self.unavailable_reason,
        }


def build_adjacency(num_node: int = 33, edges: Optional[Sequence[tuple[int, int]]] = None) -> np.ndarray:
    _require_numpy()
    edge_list = MP_EDGES if edges is None else edges
    adj = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edge_list:
        adj[i, j] = adj[j, i] = 1.0
    adj += np.eye(num_node, dtype=np.float32)
    degree = adj.sum(axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-6))
    return inv_sqrt @ adj @ inv_sqrt


# MediaPipe pose graph edges, matching the training-time graph.
MP_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

_A_NORM_NP = build_adjacency() if np is not None else None
A_NORM = torch.from_numpy(_A_NORM_NP).float() if torch is not None and _A_NORM_NP is not None else None


if nn is not None:

    class GraphConv(nn.Module):  # type: ignore[misc]
        def __init__(self, in_c: int, out_c: int, A: Any):
            super().__init__()
            self.register_buffer("A", A)
            self.conv = nn.Conv2d(in_c, out_c, 1)

        def forward(self, x: Any) -> Any:
            return torch.einsum("nctv,vw->nctw", self.conv(x), self.A)


    class STGCNBlock(nn.Module):  # type: ignore[misc]
        def __init__(self, in_c: int, out_c: int, A: Any, kernel_t: int = 9,
                     stride: int = 1, residual: bool = True):
            super().__init__()
            pad = (kernel_t - 1) // 2
            self.gcn = GraphConv(in_c, out_c, A)
            self.tcn = nn.Sequential(
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, (kernel_t, 1), stride=(stride, 1), padding=(pad, 0)),
                nn.BatchNorm2d(out_c),
            )
            if not residual:
                self.residual = lambda _: 0
            elif in_c == out_c and stride == 1:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_c),
                )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: Any) -> Any:
            res = self.residual(x)
            x = self.tcn(self.gcn(x))
            return self.relu(x + res)


    class STGCNBackbone(nn.Module):  # type: ignore[misc]
        def __init__(self, in_channels: int = 3, A: Any = None, base: int = 64):
            super().__init__()
            graph = A_NORM if A is None else A
            if graph is None:
                raise RealtimeModelUnavailable("torch unavailable: cannot create graph tensor")
            self.data_bn = nn.BatchNorm1d(in_channels * graph.size(0))
            self.layers = nn.ModuleList([
                STGCNBlock(in_channels, base, graph, residual=False),
                STGCNBlock(base, base, graph),
                STGCNBlock(base, base, graph),
                STGCNBlock(base, base * 2, graph, stride=2),
                STGCNBlock(base * 2, base * 2, graph),
                STGCNBlock(base * 2, base * 4, graph, stride=2),
                STGCNBlock(base * 4, base * 4, graph),
            ])
            self.out_channels = base * 4

        def forward(self, x: Any) -> Any:
            n, c, t, v, m = x.size()
            x = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
            x = self.data_bn(x)
            x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
            for block in self.layers:
                x = block(x)
            cout, tout, vout = x.size(1), x.size(2), x.size(3)
            return x.view(n, m, cout, tout, vout).mean(dim=1)


    class MultiTaskSTGCNLSTM(nn.Module):  # type: ignore[misc]
        def __init__(self, num_action: int = NUM_CLASSES, num_phase: int = NUM_PHASES,
                     in_c: int = 3, lstm_hidden: int = 128, lstm_layers: int = 1,
                     dropout: float = 0.3):
            super().__init__()
            self.backbone = STGCNBackbone(in_channels=in_c)
            channels = self.backbone.out_channels
            self.action_head = nn.Sequential(
                nn.Linear(channels, channels // 2), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(channels // 2, num_action),
            )
            self.lstm = nn.LSTM(
                input_size=channels,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
                bidirectional=False,
            )
            self.phase_head = nn.Sequential(
                nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(lstm_hidden, num_phase),
            )

        def forward(self, x: Any) -> Any:
            feat = self.backbone(x)
            action_logit = self.action_head(feat.mean(dim=(2, 3)))
            temporal = feat.mean(dim=-1).permute(0, 2, 1)
            lstm_out, _ = self.lstm(temporal)
            phase_logit = self.phase_head(lstm_out[:, -1, :])
            return action_logit, phase_logit


    class MultiTaskSTGCNMLP(nn.Module):  # type: ignore[misc]
        def __init__(self, num_action: int = NUM_CLASSES, num_phase: int = NUM_PHASES,
                     in_c: int = 3, mlp_hidden: int = 128, dropout: float = 0.3):
            super().__init__()
            self.backbone = STGCNBackbone(in_channels=in_c)
            channels = self.backbone.out_channels
            self.action_head = nn.Sequential(
                nn.Linear(channels, channels // 2), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(channels // 2, num_action),
            )
            self.phase_head = nn.Sequential(
                nn.Linear(channels, mlp_hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden // 2), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(mlp_hidden // 2, num_phase),
            )

        def forward(self, x: Any) -> Any:
            feat = self.backbone(x)
            pooled = feat.mean(dim=(2, 3))
            return self.action_head(pooled), self.phase_head(pooled)

else:

    class _TorchUnavailablePlaceholder:
        def __init__(self, *_: Any, **__: Any):
            raise RealtimeModelUnavailable(f"torch unavailable: {_TORCH_IMPORT_ERROR}")

    class GraphConv(_TorchUnavailablePlaceholder):
        pass

    class STGCNBlock(_TorchUnavailablePlaceholder):
        pass

    class STGCNBackbone(_TorchUnavailablePlaceholder):
        pass

    class MultiTaskSTGCNLSTM(_TorchUnavailablePlaceholder):
        pass

    class MultiTaskSTGCNMLP(_TorchUnavailablePlaceholder):
        pass


def normalize_kpts(k: np.ndarray) -> np.ndarray:
    """Normalize keypoints with the same scale/centering used during training."""
    _require_numpy()
    arr = np.asarray(k, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (NUM_KPT, 3):
        raise ValueError(f"expected keypoints shape [T, 33, 3], got {arr.shape}")
    xy = arr[..., :2].copy()
    conf = arr[..., 2:3].copy()
    hip = (xy[:, 23] + xy[:, 24]) / 2
    sho = (xy[:, 11] + xy[:, 12]) / 2
    scale = np.linalg.norm(sho - hip, axis=1, keepdims=True) + 1e-6
    xy = (xy - hip[:, None, :]) / scale[:, None, :]
    return np.concatenate([xy, conf], axis=-1).astype(np.float32)


def normalize_derivative_mode(value: Any) -> str:
    """Normalize the runtime input-feature mode used by training checkpoints."""
    mode = str(value or "pose").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "pose",
        "pose_only": "pose",
        "pose+velocity": "velocity",
        "pose_velocity": "velocity",
        "vel": "velocity",
        "v": "velocity",
        "pose+acceleration": "acceleration",
        "pose_acceleration": "acceleration",
        "acc": "acceleration",
        "a": "acceleration",
        "both": "velocity_acceleration",
        "pose+velocity+acceleration": "velocity_acceleration",
        "velocity+acceleration": "velocity_acceleration",
        "vel_acc": "velocity_acceleration",
        "v+a": "velocity_acceleration",
    }
    mode = aliases.get(mode, mode)
    if mode not in DERIVATIVE_INPUT_CHANNELS:
        raise RealtimeModelUnavailable(
            f"unsupported derivative_mode={value!r}; allowed={sorted(DERIVATIVE_INPUT_CHANNELS)}"
        )
    return mode


def temporal_xy_velocity(clip: np.ndarray) -> np.ndarray:
    """First finite difference over normalized x/y with zero first-frame padding."""
    xy = clip[..., :2].astype(np.float32, copy=False)
    velocity = np.zeros_like(xy, dtype=np.float32)
    if len(xy) > 1:
        velocity[1:] = xy[1:] - xy[:-1]
    return velocity


def temporal_xy_acceleration(clip: np.ndarray) -> np.ndarray:
    """Second finite difference over normalized x/y with zero first/two-frame padding."""
    xy = clip[..., :2].astype(np.float32, copy=False)
    acceleration = np.zeros_like(xy, dtype=np.float32)
    if len(xy) > 2:
        acceleration[2:] = xy[2:] - (2.0 * xy[1:-1]) + xy[:-2]
    return acceleration


def build_pose_input_features(clip: np.ndarray, derivative_mode: Any = "pose") -> np.ndarray:
    """Return training-compatible input channels for one normalized pose window."""
    mode = normalize_derivative_mode(derivative_mode)
    base = np.asarray(clip, dtype=np.float32)
    parts = [base]
    if mode in {"velocity", "velocity_acceleration"}:
        parts.append(temporal_xy_velocity(base))
    if mode in {"acceleration", "velocity_acceleration"}:
        parts.append(temporal_xy_acceleration(base))
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def landmarks_to_kpts(lms: Iterable[Any]) -> np.ndarray:
    """Convert MediaPipe landmark objects to [33, 3] = [x, y, visibility]."""
    _require_numpy()
    if lms is None:
        raise ValueError("landmarks are None")
    rows: list[list[float]] = []
    for lm in list(lms)[:NUM_KPT]:
        rows.append([
            float(getattr(lm, "x", 0.0)),
            float(getattr(lm, "y", 0.0)),
            float(getattr(lm, "visibility", 1.0)),
        ])
    while len(rows) < NUM_KPT:
        rows.append([0.0, 0.0, 0.0])
    return np.asarray(rows, dtype=np.float32)


def resolve_checkpoint_path(checkpoint_path: Optional[str | os.PathLike[str]] = None,
                            repo_root: Optional[str | os.PathLike[str]] = None) -> tuple[Optional[Path], Optional[str]]:
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    best = root / DEFAULT_BEST_CHECKPOINT.relative_to(REPO_ROOT)
    latest = root / DEFAULT_LATEST_CHECKPOINT.relative_to(REPO_ROOT)

    if checkpoint_path:
        return Path(checkpoint_path).expanduser(), None

    for env_name in ("TRAINED_MODEL_CHECKPOINT", "REALTIME_STGCN_CHECKPOINT"):
        env_path = os.environ.get(env_name)
        if env_path:
            return Path(env_path).expanduser(), None

    if best.exists():
        return best, None
    if latest.exists():
        return latest, "latest.pt selected because best checkpoint was not found; latest may not be deployable-best"
    return None, "no realtime ST-GCN checkpoint found"


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RealtimeModelUnavailable(f"torch unavailable: {_TORCH_IMPORT_ERROR}")


def _torch_load(path: Path, map_location: Any) -> Any:
    _require_torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[union-attr]
    except TypeError as exc:
        # Very old torch builds may not expose weights_only. Do not silently
        # fall back to pickle-capable loading for CLI-supplied checkpoints.
        if os.environ.get("TRUST_TORCH_CHECKPOINT") != "1":
            raise RealtimeModelUnavailable(
                "safe checkpoint loading is unavailable in this torch build. "
                "Set TRUST_TORCH_CHECKPOINT=1 only for trusted local .pt files."
            ) from exc
        print("[경고] TRUST_TORCH_CHECKPOINT=1: pickle-capable torch.load fallback is enabled for a trusted local checkpoint.")
        return torch.load(path, map_location=map_location)  # type: ignore[union-attr]
    except Exception as exc:
        if os.environ.get("TRUST_TORCH_CHECKPOINT") != "1":
            raise RealtimeModelUnavailable(
                "safe checkpoint load failed. Refusing pickle-capable fallback for an untrusted checkpoint; "
                "set TRUST_TORCH_CHECKPOINT=1 only for trusted local .pt files."
            ) from exc
        print("[경고] TRUST_TORCH_CHECKPOINT=1: retrying checkpoint load with pickle-capable torch.load fallback.")
        try:
            return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[union-attr]
        except TypeError:
            return torch.load(path, map_location=map_location)  # type: ignore[union-attr]


def _state_dict_from_checkpoint(ckpt: Any) -> Mapping[str, Any]:
    if isinstance(ckpt, Mapping):
        for key in ("model", "model_state", "state_dict"):
            val = ckpt.get(key)
            if isinstance(val, Mapping):
                return val
        # Raw state_dict fallback; metadata validation elsewhere decides if allowed.
        if ckpt and all(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    raise RealtimeModelUnavailable("checkpoint does not contain a recognized model state_dict")


def _strip_module_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    if not any(str(k).startswith("module.") for k in state.keys()):
        return dict(state)
    return {str(k).removeprefix("module."): v for k, v in state.items()}


def _metadata_from_checkpoint(ckpt: Any,
                              trusted_metadata: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    trusted = dict(trusted_metadata or {})
    data = dict(ckpt) if isinstance(ckpt, Mapping) else {}
    cfg = data.get("cfg") if isinstance(data.get("cfg"), Mapping) else {}
    metadata = {
        "classes": data.get("classes", trusted.get("classes")),
        "phase_names": data.get("phase_names", trusted.get("phase_names")),
        "cfg": {**dict(trusted.get("cfg", {})), **dict(cfg)},
        "clip_len": data.get("clip_len", trusted.get("clip_len")),
    }
    if metadata["clip_len"] is None:
        metadata["clip_len"] = metadata["cfg"].get("clip_len")
    return metadata


def validate_checkpoint_metadata(ckpt: Any,
                                 clip_len_override: Optional[int] = None,
                                 trusted_metadata: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    metadata = _metadata_from_checkpoint(ckpt, trusted_metadata=trusted_metadata)
    classes = metadata.get("classes")
    phase_names = metadata.get("phase_names")
    if list(classes or []) != CLASS_LIST:
        raise RealtimeModelUnavailable(
            f"checkpoint class order missing/mismatch: expected {CLASS_LIST}, got {classes!r}"
        )
    if list(phase_names or []) != PHASE_NAMES:
        raise RealtimeModelUnavailable(
            f"checkpoint phase order missing/mismatch: expected {PHASE_NAMES}, got {phase_names!r}"
        )

    cfg = dict(metadata.get("cfg") or {})
    model_type = str(cfg.get("model_type", "")).lower()
    if model_type not in {"mlp", "lstm"}:
        raise RealtimeModelUnavailable(f"unsupported/missing model_type in checkpoint cfg: {model_type!r}")
    if "hidden" not in cfg:
        raise RealtimeModelUnavailable("checkpoint cfg missing hidden")
    if "dropout" not in cfg:
        raise RealtimeModelUnavailable("checkpoint cfg missing dropout")
    if model_type == "lstm" and "lstm_layers" not in cfg:
        raise RealtimeModelUnavailable("checkpoint cfg missing lstm_layers")

    clip_len = clip_len_override if clip_len_override is not None else metadata.get("clip_len")
    if clip_len is None:
        raise RealtimeModelUnavailable("checkpoint missing clip_len and no override was provided")
    cfg["clip_len"] = int(clip_len)
    cfg["hidden"] = int(cfg["hidden"])
    cfg["dropout"] = float(cfg["dropout"])
    cfg["lstm_layers"] = int(cfg.get("lstm_layers", 1))
    cfg["model_type"] = model_type
    cfg["derivative_mode"] = normalize_derivative_mode(cfg.get("derivative_mode", REALTIME_DERIVATIVE_MODE))
    expected_channels = DERIVATIVE_INPUT_CHANNELS[cfg["derivative_mode"]]
    cfg["input_channels"] = int(cfg.get("input_channels") or expected_channels)
    if cfg["input_channels"] != expected_channels:
        raise RealtimeModelUnavailable(
            f"checkpoint input channel mismatch for derivative_mode={cfg['derivative_mode']!r}: "
            f"expected {expected_channels}, got {cfg['input_channels']}."
        )
    phase_head_type = str(cfg.get("phase_head_type") or "mlp").lower()
    if phase_head_type not in {"mlp", "baseline", "none"}:
        raise RealtimeModelUnavailable(
            "realtime ST-GCN adapter only supports plain MLP phase heads; "
            f"got phase_head_type={phase_head_type!r}."
        )
    return cfg


def build_model_from_cfg(cfg: Mapping[str, Any], device: Any = "cpu") -> Any:
    _require_torch()
    model_type = str(cfg["model_type"]).lower()
    input_channels = int(cfg.get("input_channels", REALTIME_INPUT_CHANNELS))
    if model_type == "lstm":
        model = MultiTaskSTGCNLSTM(
            num_action=NUM_CLASSES,
            num_phase=NUM_PHASES,
            in_c=input_channels,
            lstm_hidden=int(cfg["hidden"]),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
            dropout=float(cfg["dropout"]),
        )
    else:
        model = MultiTaskSTGCNMLP(
            num_action=NUM_CLASSES,
            num_phase=NUM_PHASES,
            in_c=input_channels,
            mlp_hidden=int(cfg["hidden"]),
            dropout=float(cfg["dropout"]),
        )
    return model.to(device)


@dataclass
class PhaseCounterState:
    phase: str
    phase_id: int
    count: int
    last_increment: bool
    segment_len: int


class OnlinePhaseCounter:
    """Streaming counterpart of count_phases().

    Counts when a valid UP segment ends.  In realtime, the phase model can insert a
    few READY frames inside one still-active rep; ready_bridge_max_len delays the
    UP finalization so UP-READY-UP does not become two counts.
    """

    def __init__(self, smooth_window: int = 1, min_up_len: int = 3,
                 require_prior_down: bool = False,
                 ready_bridge_max_len: int = 0):
        self.smooth_window = max(1, int(smooth_window))
        self.min_up_len = max(1, int(min_up_len))
        self.require_prior_down = bool(require_prior_down)
        self.ready_bridge_max_len = max(0, int(ready_bridge_max_len))
        self._window: deque[int] = deque(maxlen=self.smooth_window)
        self.reset()

    def reset(self) -> None:
        self._window.clear()
        self.count = 0
        self.last_increment = False
        self.current_phase_id: Optional[int] = None
        self.segment_len = 0
        self._seen_down_since_count = False
        self._up_started_after_down = False
        self._pending_ready_len = 0

    def clear_frame(self) -> PhaseCounterState:
        self.last_increment = False
        phase_id = self.current_phase_id if self.current_phase_id is not None else PHASE_READY
        return PhaseCounterState(
            phase=PHASE_NAMES[phase_id],
            phase_id=phase_id,
            count=self.count,
            last_increment=False,
            segment_len=self.segment_len,
        )

    def hold_frame(self) -> PhaseCounterState:
        """Extend the current phase over a source frame without new inference."""
        self.last_increment = False
        if self.current_phase_id is None:
            return self.clear_frame()

        if self._pending_ready_len > 0:
            self._pending_ready_len += 1
            self.segment_len += 1
            if self._pending_ready_len > self.ready_bridge_max_len:
                ready_len = self._pending_ready_len
                self._commit_current_up_if_valid()
                self.current_phase_id = PHASE_READY
                self.segment_len = ready_len
                self._pending_ready_len = 0
                self._up_started_after_down = False
                return self._state(PHASE_READY)
            return self._state(self.current_phase_id)

        self.segment_len += 1
        return self._state(self.current_phase_id)

    def _phase_to_id(self, phase: int | str) -> int:
        if isinstance(phase, str):
            if phase not in PHASE_NAMES:
                raise ValueError(f"unknown phase name: {phase!r}")
            return PHASE_NAMES.index(phase)
        idx = int(phase)
        if idx < 0 or idx >= NUM_PHASES:
            raise ValueError(f"unknown phase id: {idx}")
        return idx

    def _smooth(self, phase_id: int) -> int:
        self._window.append(phase_id)
        counts: dict[int, int] = {}
        for value in self._window:
            counts[value] = counts.get(value, 0) + 1
        latest_index = {value: idx for idx, value in enumerate(self._window)}
        return max(counts, key=lambda value: (counts[value], latest_index[value]))

    def _commit_current_up_if_valid(self) -> None:
        up_is_valid = self.current_phase_id == PHASE_UP and self.segment_len >= self.min_up_len
        prior_down_ok = (not self.require_prior_down) or self._up_started_after_down
        if up_is_valid and prior_down_ok:
            self.count += 1
            self.last_increment = True
            self._seen_down_since_count = False

    def _start_phase(self, phase_id: int) -> None:
        self.current_phase_id = phase_id
        self.segment_len = 1
        self._pending_ready_len = 0
        if phase_id == PHASE_DOWN:
            self._seen_down_since_count = True
        if phase_id == PHASE_UP:
            self._up_started_after_down = self._seen_down_since_count
        elif phase_id != PHASE_UP:
            self._up_started_after_down = False

    def update(self, phase: int | str) -> PhaseCounterState:
        self.last_increment = False
        phase_id = self._smooth(self._phase_to_id(phase))

        if self.current_phase_id is None:
            self._start_phase(phase_id)
            return self._state(phase_id)

        if self._pending_ready_len > 0:
            if phase_id == PHASE_READY:
                self._pending_ready_len += 1
                self.segment_len += 1
                if self._pending_ready_len > self.ready_bridge_max_len:
                    ready_len = self._pending_ready_len
                    self._commit_current_up_if_valid()
                    self.current_phase_id = PHASE_READY
                    self.segment_len = ready_len
                    self._pending_ready_len = 0
                    self._up_started_after_down = False
                    return self._state(PHASE_READY)
                return self._state(self.current_phase_id)

            if phase_id == PHASE_UP:
                # Short READY gap inside one UP segment: bridge it and keep counting as one rep.
                self._pending_ready_len = 0
                self.segment_len += 1
                return self._state(self.current_phase_id)

            # READY followed by DOWN means the UP rep really ended; count once, then enter DOWN.
            self._commit_current_up_if_valid()
            self._start_phase(phase_id)
            return self._state(phase_id)

        if phase_id == self.current_phase_id:
            self.segment_len += 1
            if phase_id == PHASE_DOWN:
                self._seen_down_since_count = True
            return self._state(phase_id)

        if (
            self.ready_bridge_max_len > 0
            and self.current_phase_id == PHASE_UP
            and phase_id == PHASE_READY
        ):
            self._pending_ready_len = 1
            self.segment_len += 1
            return self._state(self.current_phase_id)

        self._commit_current_up_if_valid()
        self._start_phase(phase_id)

        return self._state(phase_id)

    def _state(self, phase_id: int) -> PhaseCounterState:
        return PhaseCounterState(
            phase=PHASE_NAMES[phase_id],
            phase_id=phase_id,
            count=self.count,
            last_increment=self.last_increment,
            segment_len=self.segment_len,
        )


class RealtimeSTGCNInfer:
    """Online inference wrapper for multitask ST-GCN action + phase/count."""

    def __init__(self,
                 checkpoint_path: Optional[str | os.PathLike[str]] = None,
                 device: str = "auto",
                 clip_len_override: Optional[int] = None,
                 smooth_window: int = 5,
                 min_up_len: int = 3,
                 inference_interval: int = 1,
                 require_prior_down: bool = False,
                 ready_bridge_max_len: int = 0,
                 require_prior_down_actions: Optional[Sequence[str]] = None,
                 required: bool = False,
                 trusted_metadata: Optional[Mapping[str, Any]] = None,
                 repo_root: Optional[str | os.PathLike[str]] = None):
        self.required = bool(required)
        self.available = False
        self.unavailable_reason: Optional[str] = None
        self.warning: Optional[str] = None
        self.device = "cpu"
        self.model = None
        self.cfg: dict[str, Any] = {}
        self.clip_len: Optional[int] = None
        self.checkpoint_path: Optional[Path] = None
        self.inference_interval = max(1, int(inference_interval))
        self.require_prior_down_actions = (
            None if require_prior_down_actions is None
            else {str(name).lower().strip() for name in require_prior_down_actions}
        )
        self.count_exercise_override: Optional[str] = None
        self._frame_index = 0
        self._buffer: deque[np.ndarray] = deque()
        self.counter = OnlinePhaseCounter(
            smooth_window=smooth_window,
            min_up_len=min_up_len,
            require_prior_down=require_prior_down,
            ready_bridge_max_len=ready_bridge_max_len,
        )
        self.state = InferenceState(
            available=False,
            unavailable_reason="not initialized",
            warning=None,
        )
        try:
            self._initialize(
                checkpoint_path=checkpoint_path,
                device=device,
                clip_len_override=clip_len_override,
                trusted_metadata=trusted_metadata,
                repo_root=repo_root,
            )
        except RealtimeModelUnavailable as exc:
            self._mark_unavailable(str(exc))
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            self._mark_unavailable(f"unexpected realtime model init failure: {exc}")

        if self.required and not self.available:
            raise RealtimeModelUnavailable(self.unavailable_reason or "realtime model unavailable")

    def _initialize(self,
                    checkpoint_path: Optional[str | os.PathLike[str]],
                    device: str,
                    clip_len_override: Optional[int],
                    trusted_metadata: Optional[Mapping[str, Any]],
                    repo_root: Optional[str | os.PathLike[str]]) -> None:
        _require_numpy()
        _require_torch()
        selected, warning = resolve_checkpoint_path(checkpoint_path, repo_root=repo_root)
        if selected is None:
            raise RealtimeModelUnavailable(warning or "checkpoint not found")
        if not selected.exists():
            raise RealtimeModelUnavailable(f"checkpoint not found: {selected}")

        resolved_device = self._resolve_device(device)
        ckpt = _torch_load(selected, map_location=resolved_device)
        cfg = validate_checkpoint_metadata(
            ckpt,
            clip_len_override=clip_len_override,
            trusted_metadata=trusted_metadata,
        )
        model = build_model_from_cfg(cfg, device=resolved_device)
        state_dict = _strip_module_prefix(_state_dict_from_checkpoint(ckpt))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.available = True
        self.unavailable_reason = None
        self.warning = warning
        self.device = str(resolved_device)
        self.model = model
        self.cfg = cfg
        self.clip_len = int(cfg["clip_len"])
        self.checkpoint_path = selected
        self._buffer = deque(maxlen=self.clip_len)
        self.state = InferenceState(
            available=True,
            ready=False,
            checkpoint_path=str(selected),
            clip_len=self.clip_len,
            warning=warning,
        )

    def _resolve_device(self, device: str) -> Any:
        _require_torch()
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[union-attr]
        if device == "cuda" and not torch.cuda.is_available():  # type: ignore[union-attr]
            return "cpu"
        return device

    def _mark_unavailable(self, reason: str) -> None:
        self.available = False
        self.unavailable_reason = reason
        self.state = InferenceState(
            available=False,
            ready=False,
            count=self.counter.count,
            last_increment=False,
            unavailable_reason=reason,
            checkpoint_path=None if self.checkpoint_path is None else str(self.checkpoint_path),
            clip_len=self.clip_len,
            warning=self.warning,
        )

    def reset(self) -> None:
        self._frame_index = 0
        self._buffer.clear()
        self.counter.reset()
        self.state = InferenceState(
            available=self.available,
            ready=False,
            checkpoint_path=None if self.checkpoint_path is None else str(self.checkpoint_path),
            clip_len=self.clip_len,
            warning=self.warning,
            unavailable_reason=self.unavailable_reason,
        )

    def clear_frame(self) -> InferenceState:
        counter_state = self.counter.clear_frame()
        self.state.ready = False
        self.state.pred_class = None
        self.state.pred_class_id = None
        self.state.action_confidence = 0.0
        self.state.action_probs = None
        self.state.last_increment = False
        self.state.count = counter_state.count
        self.state.phase = counter_state.phase
        self.state.phase_id = counter_state.phase_id
        self.state.segment_len = counter_state.segment_len
        self.state.phase_confidence = 0.0
        self.state.phase_probs = None
        return self.state

    def update(self, lms: Iterable[Any]) -> InferenceState:
        return self.update_kpts(landmarks_to_kpts(lms))

    def set_count_exercise(self, exercise: Optional[str]) -> None:
        """Lock count gating to an externally selected exercise.

        Passing None returns to raw model action based gating.
        """
        if exercise is None:
            self.count_exercise_override = None
            return
        self.count_exercise_override = str(exercise).lower().strip()

    def update_kpts(self, kpts: np.ndarray) -> InferenceState:
        if not self.available or self.model is None or self.clip_len is None:
            self.counter.clear_frame()
            return self.state

        arr = np.asarray(kpts, dtype=np.float32)
        if arr.shape != (NUM_KPT, 3):
            raise ValueError(f"expected one frame [33, 3], got {arr.shape}")
        self._frame_index += 1
        self._buffer.append(arr)

        if len(self._buffer) < self.clip_len:
            self.state.ready = False
            self.state.last_increment = False
            self.state.count = self.counter.count
            self.state.segment_len = self.counter.segment_len
            return self.state

        if self._frame_index % self.inference_interval != 0 and self.state.ready:
            counter_state = self.counter.hold_frame()
            self.state.ready = True
            self.state.last_increment = False
            self.state.count = counter_state.count
            self.state.phase = counter_state.phase
            self.state.phase_id = counter_state.phase_id
            self.state.segment_len = counter_state.segment_len
            return self.state

        clip = np.stack(list(self._buffer), axis=0)
        norm = normalize_kpts(clip)
        features = build_pose_input_features(norm, self.cfg.get("derivative_mode", REALTIME_DERIVATIVE_MODE))
        x = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).contiguous()  # type: ignore[union-attr]
        x = x.to(self.device)
        with torch.no_grad():  # type: ignore[union-attr]
            action_logit, phase_logit = self.model(x)
            action_prob = F.softmax(action_logit, dim=1).detach().cpu().numpy()[0]
            phase_prob = F.softmax(phase_logit, dim=1).detach().cpu().numpy()[0]

        pred_cls_id = int(action_prob.argmax())
        pred_cls_name = CLASS_LIST[pred_cls_id]
        phase_id = int(phase_prob.argmax())
        if self.require_prior_down_actions is not None:
            # Squat/bench reps must be re-armed by a DOWN phase before another
            # UP segment can count.  Deadlift can remain UP-only when requested.
            count_action_name = self.count_exercise_override or pred_cls_name
            self.counter.require_prior_down = count_action_name in self.require_prior_down_actions
        counter_state = self.counter.update(phase_id)
        self.state = InferenceState(
            available=True,
            ready=True,
            pred_class=pred_cls_name,
            pred_class_id=pred_cls_id,
            action_confidence=float(action_prob[pred_cls_id]),
            action_probs=action_prob.astype(np.float32),
            phase=counter_state.phase,
            phase_id=counter_state.phase_id,
            phase_confidence=float(phase_prob[phase_id]),
            phase_probs=phase_prob.astype(np.float32),
            count=counter_state.count,
            last_increment=counter_state.last_increment,
            segment_len=counter_state.segment_len,
            checkpoint_path=None if self.checkpoint_path is None else str(self.checkpoint_path),
            clip_len=self.clip_len,
            warning=self.warning,
            unavailable_reason=None,
        )
        return self.state


def model_status_line(state: Optional[InferenceState]) -> str:
    if state is None:
        return "AI model: inactive"
    if not state.available:
        return f"AI model fallback: {state.unavailable_reason or 'unavailable'}"
    cls = state.pred_class or "warming"
    phase = state.phase or "ready"
    return (
        f"AI {cls} {state.action_confidence:.2f} | "
        f"phase {phase} {state.phase_confidence:.2f} | count {state.count}"
    )


def _make_pycharm_dummy_kpts(frame_index: int = 0) -> np.ndarray:
    """Create one synthetic MediaPipe-like frame for direct PyCharm smoke runs.

    This is not an accuracy test.  It only proves that checkpoint resolution,
    model loading, tensor shaping, and one online inference path work when the
    file is launched with PyCharm's Run button.
    """
    _require_numpy()
    t = float(frame_index)
    kpts = np.zeros((NUM_KPT, 3), dtype=np.float32)
    kpts[:, 0] = 0.5
    kpts[:, 1] = 0.5
    kpts[:, 2] = 0.2

    # Rough upright body pose in normalized image coordinates.
    coords = {
        0: (0.50, 0.12),
        11: (0.42, 0.28), 12: (0.58, 0.28),
        13: (0.38, 0.42), 14: (0.62, 0.42),
        15: (0.36, 0.56), 16: (0.64, 0.56),
        23: (0.44, 0.55), 24: (0.56, 0.55),
        25: (0.43, 0.74), 26: (0.57, 0.74),
        27: (0.42, 0.92), 28: (0.58, 0.92),
        29: (0.39, 0.95), 30: (0.61, 0.95),
        31: (0.46, 0.96), 32: (0.54, 0.96),
    }
    sway = 0.01 * np.sin(t / 4.0) if np is not None else 0.0
    for idx, (x, y) in coords.items():
        kpts[idx] = (x + sway, y, 1.0)
    return kpts


def _resolve_pycharm_checkpoint_override() -> Optional[Path]:
    """Resolve PyCharm-edited checkpoint paths from the repository root."""
    if PYCHARM_CHECKPOINT_PATH is None:
        return None
    path = Path(PYCHARM_CHECKPOINT_PATH).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_pycharm_smoke() -> int:
    """Run a no-CLI checkpoint smoke test for PyCharm.

    PyCharm 사용법:
    1. 이 파일(`model/realtime_stgcn_infer.py`)을 엽니다.
    2. 위의 `PyCharm Run 설정` 값이 필요하면 수정합니다.
    3. 우클릭 → Run `realtime_stgcn_infer`를 누릅니다.
    """
    print("=== Realtime ST-GCN PyCharm smoke ===")
    checkpoint_override = _resolve_pycharm_checkpoint_override()
    selected, warning = resolve_checkpoint_path(checkpoint_override)
    print(f"[설정] checkpoint = {selected or '자동 탐색 실패'}")
    if warning:
        print(f"[경고] {warning}")
    print(f"[설정] device = {PYCHARM_DEVICE}")

    try:
        infer = RealtimeSTGCNInfer(
            checkpoint_path=checkpoint_override,
            device=PYCHARM_DEVICE,
            clip_len_override=PYCHARM_CLIP_LEN_OVERRIDE,
            smooth_window=PYCHARM_SMOOTH_WINDOW,
            min_up_len=PYCHARM_MIN_UP_LEN,
            inference_interval=PYCHARM_INFERENCE_INTERVAL,
            ready_bridge_max_len=PYCHARM_READY_BRIDGE_MAX_LEN,
            required=PYCHARM_REQUIRED,
        )
    except Exception as exc:
        print(f"[실패] 모델 초기화 예외: {exc}")
        return 1

    if not infer.available:
        print(f"[실패] 모델 사용 불가: {infer.unavailable_reason}")
        print("[힌트] PyCharm 인터프리터에 torch/numpy가 설치되어 있고 checkpoint 경로가 맞는지 확인하세요.")
        return 1

    print(f"[성공] checkpoint 로드: {infer.checkpoint_path}")
    print(f"[성공] device={infer.device}, clip_len={infer.clip_len}, cfg={infer.cfg}")

    frames = PYCHARM_SMOKE_FRAMES or infer.clip_len or 1
    state = infer.state
    for frame_idx in range(int(frames)):
        state = infer.update_kpts(_make_pycharm_dummy_kpts(frame_idx))
        if PYCHARM_PRINT_EVERY_FRAME:
            print(f"[frame {frame_idx + 1:03d}] {model_status_line(state)}")

    print(f"[결과] {model_status_line(state)}")
    print(f"[결과 dict] {state.to_dict()}")
    return 0


def run_pycharm_gui_app() -> int:
    """Launch the realtime OpenCV app when this adapter file is run directly."""
    import runpy

    app_path = REPO_ROOT / "realtime_compare_side_infer.py"
    if not app_path.exists():
        print(f"[실패] realtime app 파일을 찾을 수 없습니다: {app_path}")
        return 1
    print(f"[실행] {app_path.name} GUI 앱을 시작합니다.")
    runpy.run_path(str(app_path), run_name="__main__")
    return 0


__all__ = [
    "A_NORM",
    "CLASS_LIST",
    "DEFAULT_BEST_CHECKPOINT",
    "DEFAULT_LATEST_CHECKPOINT",
    "GraphConv",
    "InferenceState",
    "MP_EDGES",
    "MultiTaskSTGCNLSTM",
    "MultiTaskSTGCNMLP",
    "OnlinePhaseCounter",
    "PHASE_NAMES",
    "PHASE_DOWN",
    "PHASE_READY",
    "PHASE_UP",
    "PhaseCounterState",
    "RealtimeModelUnavailable",
    "RealtimeSTGCNInfer",
    "STGCNBackbone",
    "STGCNBlock",
    "build_adjacency",
    "build_model_from_cfg",
    "build_pose_input_features",
    "landmarks_to_kpts",
    "model_status_line",
    "normalize_derivative_mode",
    "normalize_kpts",
    "resolve_checkpoint_path",
    "run_pycharm_gui_app",
    "run_pycharm_smoke",
    "validate_checkpoint_metadata",
]


if __name__ == "__main__":
    if PYCHARM_LAUNCH_REALTIME_APP:
        raise SystemExit(run_pycharm_gui_app())
    raise SystemExit(run_pycharm_smoke())
