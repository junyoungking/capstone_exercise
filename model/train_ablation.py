"""
Import-safe ST-GCN ablation/training utilities.

The original notebook-shaped script is kept usable through ``main()`` while all
heavy work (label loading, pose extraction, metadata construction, training) is
now behind explicit function calls.  This lets experiment runners import the
module safely from PyCharm.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "측면" / "data"
VIDEO_DIR = ROOT
LABEL_DIR = ROOT / "labels"

TEAM_WORK_DIR = ROOT / "workdir"
TEAM_POSE_DIR = TEAM_WORK_DIR / "poses"
TEAM_META_CSV = TEAM_WORK_DIR / "meta_v4.csv"

WORK_DIR = ROOT / "STGCN_LSTM"
POSE_DIR = WORK_DIR / "poses"
YOLO_POSE_DIR = WORK_DIR / "poses_yolo"
BARBELL_DIR = WORK_DIR / "barbell_yolo_world"
CKPT_DIR = WORK_DIR / "checkpoints"
PLOT_DIR = WORK_DIR / "plots"
CSV_DIR = WORK_DIR / "csv"
TMP_DIR = WORK_DIR / "tmp"
META_CSV = WORK_DIR / "meta_stgcn_lstm.csv"

CLASS_LIST = ["squat", "benchpress", "deadlift"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_LIST)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}
NUM_CLASSES = len(CLASS_LIST)

PHASE_READY = 0
PHASE_DOWN = 1
PHASE_UP = 2
NUM_PHASES = 3
PHASE_NAMES = ["ready", "down", "up"]
PHASE_LABEL_SCHEME_AS_LABELED = "as_labeled"
PHASE_LABEL_SCHEME_BAR_DIRECTION = "bar_direction"
ALLOWED_PHASE_LABEL_SCHEMES = (PHASE_LABEL_SCHEME_AS_LABELED, PHASE_LABEL_SCHEME_BAR_DIRECTION)
PHASE_LABEL_SCHEME_ALIASES = {
    "default": PHASE_LABEL_SCHEME_AS_LABELED,
    "current": PHASE_LABEL_SCHEME_AS_LABELED,
    "as-labelled": PHASE_LABEL_SCHEME_AS_LABELED,
    "as_labelled": PHASE_LABEL_SCHEME_AS_LABELED,
    "as-labeled": PHASE_LABEL_SCHEME_AS_LABELED,
    "as_labeled": PHASE_LABEL_SCHEME_AS_LABELED,
    "bar": PHASE_LABEL_SCHEME_BAR_DIRECTION,
    "bar_direction": PHASE_LABEL_SCHEME_BAR_DIRECTION,
    "barbell_direction": PHASE_LABEL_SCHEME_BAR_DIRECTION,
    "deadlift_bar_direction": PHASE_LABEL_SCHEME_BAR_DIRECTION,
    "physical_bar_direction": PHASE_LABEL_SCHEME_BAR_DIRECTION,
}

POSE_BACKEND_MEDIAPIPE = "mediapipe"
POSE_BACKEND_YOLO = "yolo"
POSE_BACKEND_BARBELL = "barbell"
POSE_BACKEND_MEDIAPIPE_BARBELL = "mediapipe_barbell"
ALLOWED_POSE_BACKENDS = (
    POSE_BACKEND_MEDIAPIPE,
    POSE_BACKEND_YOLO,
    POSE_BACKEND_BARBELL,
    POSE_BACKEND_MEDIAPIPE_BARBELL,
)

JOINT_SUBSET_ALL = "all"
JOINT_SUBSET_WRIST_ONLY = "wrist_only"
JOINT_SUBSET_POSE_ONLY = "pose_only"
ALLOWED_JOINT_SUBSETS = (JOINT_SUBSET_ALL, JOINT_SUBSET_WRIST_ONLY, JOINT_SUBSET_POSE_ONLY)
JOINT_SUBSET_ALIASES = {
    "full": JOINT_SUBSET_ALL,
    "full_pose": JOINT_SUBSET_ALL,
    "all_joints": JOINT_SUBSET_ALL,
    "pose": JOINT_SUBSET_ALL,
    "pose_only": JOINT_SUBSET_POSE_ONLY,
    "body_only": JOINT_SUBSET_POSE_ONLY,
    "mediapipe_only": JOINT_SUBSET_POSE_ONLY,
    "mp_only": JOINT_SUBSET_POSE_ONLY,
    "no_barbell": JOINT_SUBSET_POSE_ONLY,
    "wrist": JOINT_SUBSET_WRIST_ONLY,
    "wrists": JOINT_SUBSET_WRIST_ONLY,
    "wrist2": JOINT_SUBSET_WRIST_ONLY,
    "wrist_only": JOINT_SUBSET_WRIST_ONLY,
    "wrists_only": JOINT_SUBSET_WRIST_ONLY,
}

NUM_KPT = 33
YOLO_NUM_KPT = 17
BARBELL_NODE_INDEX = NUM_KPT
MEDIAPIPE_BARBELL_NUM_KPT = NUM_KPT + 1
BARBELL_EDGE_POLICY_NONE = "none"
BARBELL_EDGE_POLICY_WRISTS = "wrists"
BARBELL_EDGE_POLICY_NO_POSE_EDGES = "no_pose_edges"
BARBELL_EDGE_POLICY_HANDS = "hands"
ALLOWED_BARBELL_EDGE_POLICIES = (
    BARBELL_EDGE_POLICY_NONE,
    BARBELL_EDGE_POLICY_WRISTS,
    BARBELL_EDGE_POLICY_NO_POSE_EDGES,
    BARBELL_EDGE_POLICY_HANDS,
)
STGCN_TEMPORAL_STRIDES = (1, 1, 1, 2, 1, 2, 1)
ALLOWED_PHASE_POOLING = ("temporal_avg", "temporal_flatten", "last", "avg_last_concat")
POOLING_EXPERIMENT_VARIANTS = ["temporal_flatten", "last", "avg_last_concat"]
PHASE_HEAD_MLP = "mlp"
PHASE_HEAD_EXERCISE_ATTN = "exercise_attn"
PHASE_HEAD_PER_EXERCISE_MLP = "per_exercise_mlp"
PHASE_HEAD_BARBELL_BOX_MLP = "barbell_box_mlp"
BARBELL_BOX_INPUT_CHANNELS = 6  # normalized [x1, y1, x2, y2, confidence, detected]
PHASE_AUX_WRIST = "wrist"
PHASE_AUX_BARBELL = "barbell"
PHASE_AUX_ACCELERATION = "acceleration"
ALLOWED_PHASE_AUX_INPUTS = (PHASE_AUX_WRIST, PHASE_AUX_BARBELL, PHASE_AUX_ACCELERATION)
PHASE_AUX_INPUT_ALIASES = {
    "": "",
    "none": "",
    "off": "",
    "false": "",
    "no": "",
    "wrist": PHASE_AUX_WRIST,
    "wrists": PHASE_AUX_WRIST,
    "wrist_only": PHASE_AUX_WRIST,
    "wrist2": PHASE_AUX_WRIST,
    "bar": PHASE_AUX_BARBELL,
    "barbell": PHASE_AUX_BARBELL,
    "barbell_coord": PHASE_AUX_BARBELL,
    "barbell_coords": PHASE_AUX_BARBELL,
    "barbell_center": PHASE_AUX_BARBELL,
    "acc": PHASE_AUX_ACCELERATION,
    "accel": PHASE_AUX_ACCELERATION,
    "acceleration": PHASE_AUX_ACCELERATION,
}
EXERCISE_CONDITIONED_PHASE_HEADS = (PHASE_HEAD_EXERCISE_ATTN, PHASE_HEAD_PER_EXERCISE_MLP)
ALLOWED_PHASE_HEAD_TYPES = (
    PHASE_HEAD_MLP,
    PHASE_HEAD_EXERCISE_ATTN,
    PHASE_HEAD_PER_EXERCISE_MLP,
    PHASE_HEAD_BARBELL_BOX_MLP,
)
PHASE_HEAD_TYPE_ALIASES = {
    "baseline": PHASE_HEAD_MLP,
    "none": PHASE_HEAD_MLP,
    "temporal_mlp": PHASE_HEAD_MLP,
    "attn": PHASE_HEAD_EXERCISE_ATTN,
    "attention": PHASE_HEAD_EXERCISE_ATTN,
    "joint_attn": PHASE_HEAD_EXERCISE_ATTN,
    "joint_attention": PHASE_HEAD_EXERCISE_ATTN,
    "exercise_attention": PHASE_HEAD_EXERCISE_ATTN,
    "exercise_joint_attention": PHASE_HEAD_EXERCISE_ATTN,
    "per_exercise": PHASE_HEAD_PER_EXERCISE_MLP,
    "per_exercise_head": PHASE_HEAD_PER_EXERCISE_MLP,
    "per_exercise_heads": PHASE_HEAD_PER_EXERCISE_MLP,
    "per_exercise_mlp": PHASE_HEAD_PER_EXERCISE_MLP,
    "exercise_head": PHASE_HEAD_PER_EXERCISE_MLP,
    "exercise_heads": PHASE_HEAD_PER_EXERCISE_MLP,
    "exercise_split_head": PHASE_HEAD_PER_EXERCISE_MLP,
    "exercise_split_heads": PHASE_HEAD_PER_EXERCISE_MLP,
    "split_head": PHASE_HEAD_PER_EXERCISE_MLP,
    "split_heads": PHASE_HEAD_PER_EXERCISE_MLP,
    "box": PHASE_HEAD_BARBELL_BOX_MLP,
    "bbox": PHASE_HEAD_BARBELL_BOX_MLP,
    "box_mlp": PHASE_HEAD_BARBELL_BOX_MLP,
    "bbox_mlp": PHASE_HEAD_BARBELL_BOX_MLP,
    "direct_box": PHASE_HEAD_BARBELL_BOX_MLP,
    "direct_bbox": PHASE_HEAD_BARBELL_BOX_MLP,
    "barbell_box": PHASE_HEAD_BARBELL_BOX_MLP,
    "barbell_bbox": PHASE_HEAD_BARBELL_BOX_MLP,
    "barbell_box_mlp": PHASE_HEAD_BARBELL_BOX_MLP,
    "barbell_bbox_mlp": PHASE_HEAD_BARBELL_BOX_MLP,
}
PHASE_CONDITIONING_NONE = "none"
PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL = "ground_truth_exercise_label"
PHASE_CONDITIONING_PREDICTED_ACTION = "predicted_action"
ALLOWED_PHASE_CONDITIONING = (
    PHASE_CONDITIONING_NONE,
    PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
    PHASE_CONDITIONING_PREDICTED_ACTION,
)
PHASE_CONDITIONING_ALIASES = {
    "": PHASE_CONDITIONING_NONE,
    "none": PHASE_CONDITIONING_NONE,
    "off": PHASE_CONDITIONING_NONE,
    "false": PHASE_CONDITIONING_NONE,
    "gt": PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
    "ground_truth": PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
    "ground_truth_label": PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
    "ground_truth_exercise": PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
    "ground_truth_exercise_label": PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
    "oracle": PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
    "label": PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
    "exercise_label": PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
    "pred": PHASE_CONDITIONING_PREDICTED_ACTION,
    "predicted": PHASE_CONDITIONING_PREDICTED_ACTION,
    "predicted_action": PHASE_CONDITIONING_PREDICTED_ACTION,
    "action_prediction": PHASE_CONDITIONING_PREDICTED_ACTION,
    "action_softmax": PHASE_CONDITIONING_PREDICTED_ACTION,
}
EXERCISE_ID_SOURCE_NONE = "not_used"
EXERCISE_ID_SOURCE_GROUND_TRUTH_LABEL = "ground_truth_label"
EXERCISE_ID_SOURCE_PREDICTED_ACTION = "predicted_action_logits"
ALLOWED_DERIVATIVE_MODES = ("pose", "velocity", "acceleration", "velocity_acceleration")
DERIVATIVE_MODE_ALIASES = {
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

ABLATION_ROOT = WORK_DIR / "ablation"
ABLATION_LOG_CSV = ABLATION_ROOT / "ablation_results.csv"
ABLATION_LOG_JSON = ABLATION_ROOT / "ablation_results.json"

DEFAULT_EXPERIMENT_CONFIG: Dict[str, Any] = {
    "model_type": "mlp",
    "phase_head_type": "mlp",
    "phase_pooling": "temporal_avg",  # legacy MLP behavior; pooling runner overrides explicitly
    "derivative_mode": "pose",
    "phase_aux_inputs": [],
    "phase_aux_dim": 0,
    "phase_label_scheme": PHASE_LABEL_SCHEME_AS_LABELED,
    "hidden": 128,
    "lstm_layers": 1,
    "clip_len": 16,
    "train_stride": 2,
    "dropout": 0.3,
    "aug": True,
    "epochs": 60,
    "batch": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "phase_loss_alpha": 2.0,
    "num_workers": 0,
    "seed": SEED,
    "run_kind": "legacy",
    "output_root": str(ABLATION_ROOT),
    "root": str(ROOT),
    "video_dir": str(VIDEO_DIR),
    "label_dir": str(LABEL_DIR),
    "work_dir": str(WORK_DIR),
    "pose_dir": str(POSE_DIR),
    "yolo_pose_dir": str(YOLO_POSE_DIR),
    "barbell_dir": str(BARBELL_DIR),
    "team_pose_dir": str(TEAM_POSE_DIR),
    "team_meta_csv": str(TEAM_META_CSV),
    "meta_csv": str(META_CSV),
    "pose_backend": POSE_BACKEND_MEDIAPIPE,
    "joint_subset": JOINT_SUBSET_ALL,
    "barbell_edge_policy": BARBELL_EDGE_POLICY_NONE,
    "num_kpt": NUM_KPT,
    "source_num_kpt": NUM_KPT,
    "model_num_kpt": NUM_KPT,
    "pose_graph_id": "mediapipe33",
    "source_graph_id": "mediapipe33",
    "model_graph_id": "mediapipe33",
    "selected_joint_indices": [],
    "selected_joint_names": ["all"],
    "yolo_missing_policy": "error",
    "barbell_missing_policy": "error",
    "use_team_split": True,
    "write_meta_csv": True,
    "extract_missing_pose": False,
    "resume": True,
    "skip_completed": True,
    "force_retrain": False,
    "fresh_run_tag": None,
    "overwrite_existing": False,
    "max_videos_per_split_type": None,
    "pin_memory": True,
}

DEFAULT_POOLING_EXPERIMENT: Dict[str, Any] = {
    **DEFAULT_EXPERIMENT_CONFIG,
    "model_type": "mlp",
    "hidden": 128,
    "clip_len": 16,
    "train_stride": 2,
    "dropout": 0.3,
    "aug": True,
    "epochs": 60,
    "batch": 64,
    "phase_loss_alpha": 2.0,
    "run_kind": "full",
    "output_root": str(PROJECT_ROOT / "phase_experiments" / "pooling_ablation" / "full"),
    "write_meta_csv": False,
}

# PyCharm에서 model/train_ablation.py를 직접 실행할 때의 기본 동작.
# True면 같은 checkpoint가 있어도 eval-only로 끝내지 않고 새 run 디렉터리에 학습한다.
# 기존 best/latest checkpoint를 보존하려고 DIRECT_RUN_FRESH_RERUN도 기본 True로 둔다.
DIRECT_RUN_FORCE_TRAIN = False
DIRECT_RUN_FRESH_RERUN = False

# Import-safe placeholders.  prepare_context(update_globals=True) populates them
# for legacy helper calls that still expect module-level metadata.
LABELS: Dict[Tuple[str, str], Dict[str, Any]] = {}
meta_df = pd.DataFrame()
train_meta = pd.DataFrame()
val_meta = pd.DataFrame()
ACTIVE_CONTEXT: Optional["ExperimentContext"] = None


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


@dataclass
class PoseBackendSpec:
    name: str
    num_kpt: int
    graph_id: str
    edges: Tuple[Tuple[int, int], ...]
    flip_pairs: Tuple[Tuple[int, int], ...]
    l_hip: int
    r_hip: int
    l_sho: int
    r_sho: int


@dataclass(frozen=True)
class JointSubsetView:
    name: str
    indices: Optional[Tuple[int, ...]]
    names: Tuple[str, ...]
    num_kpt: int
    graph_id: str
    edges: Tuple[Tuple[int, int], ...]
    flip_pairs: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class ModelKeypointLoadResult:
    kpts: np.ndarray
    alignment: Dict[str, Any]


@dataclass
class ExperimentContext:
    cfg: Dict[str, Any]
    project_root: Path
    root: Path
    video_dir: Path
    label_dir: Path
    work_dir: Path
    pose_dir: Path
    yolo_pose_dir: Path
    barbell_dir: Path
    team_pose_dir: Path
    team_meta_csv: Path
    meta_csv: Path
    pose_backend: str
    num_kpt: int
    pose_graph_id: str
    labels: Dict[Tuple[str, str], Dict[str, Any]]
    meta_df: pd.DataFrame
    train_meta: pd.DataFrame
    val_meta: pd.DataFrame
    ablation_root: Path
    ablation_log_csv: Path
    ablation_log_json: Path
    device: str
    pose_todo: List[Tuple[Path, Path]]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _resolve_path(value: Any, base: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_token(value: float) -> str:
    return str(float(value))


def _dropout_token(value: float) -> str:
    return str(float(value)).replace(".", "")


def _safe_name_token(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")


def normalize_pose_backend(value: Any) -> str:
    backend = str(value or POSE_BACKEND_MEDIAPIPE).strip().lower().replace("+", "_").replace("-", "_")
    if backend in {"mp", "media_pipe"}:
        backend = POSE_BACKEND_MEDIAPIPE
    if backend in {"yolo_pose", "coco", "coco17"}:
        backend = POSE_BACKEND_YOLO
    if backend in {"yolo_world_barbell", "barbell_yolo_world", "barbell_yolo", "bar"}:
        backend = POSE_BACKEND_BARBELL
    if backend in {
        "mp_barbell",
        "mp_pose_barbell",
        "media_pipe_barbell",
        "mediapipe_bar",
        "mediapipe_barbell",
        "pose_barbell",
        "joint_fusion",
        "jointfusion",
    }:
        backend = POSE_BACKEND_MEDIAPIPE_BARBELL
    if backend not in ALLOWED_POSE_BACKENDS:
        raise ValueError(f"unsupported pose_backend: {value!r} / allowed={ALLOWED_POSE_BACKENDS}")
    return backend


def is_mediapipe_barbell_backend(value: Any) -> bool:
    return normalize_pose_backend(value) == POSE_BACKEND_MEDIAPIPE_BARBELL


def normalize_joint_subset(value: Any) -> str:
    subset = str(value or JOINT_SUBSET_ALL).strip().lower().replace("-", "_").replace(" ", "_")
    subset = JOINT_SUBSET_ALIASES.get(subset, subset)
    if subset not in ALLOWED_JOINT_SUBSETS:
        raise ValueError(f"unsupported joint_subset: {value!r} / allowed={ALLOWED_JOINT_SUBSETS}")
    return subset


def normalize_barbell_edge_policy(value: Any, pose_backend: Any = None) -> str:
    backend = normalize_pose_backend(pose_backend or POSE_BACKEND_MEDIAPIPE)
    if value is None or str(value).strip() in {"", "None"}:
        return BARBELL_EDGE_POLICY_WRISTS if backend == POSE_BACKEND_MEDIAPIPE_BARBELL else BARBELL_EDGE_POLICY_NONE
    policy = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "default": BARBELL_EDGE_POLICY_WRISTS,
        "wrist": BARBELL_EDGE_POLICY_WRISTS,
        "both_wrists": BARBELL_EDGE_POLICY_WRISTS,
        "hand": BARBELL_EDGE_POLICY_HANDS,
        "both_hands": BARBELL_EDGE_POLICY_HANDS,
        "no_edges": BARBELL_EDGE_POLICY_NO_POSE_EDGES,
        "no_edge": BARBELL_EDGE_POLICY_NO_POSE_EDGES,
        "no_pose_edge": BARBELL_EDGE_POLICY_NO_POSE_EDGES,
        "no_pose_edges": BARBELL_EDGE_POLICY_NO_POSE_EDGES,
        "pose_only": BARBELL_EDGE_POLICY_NO_POSE_EDGES,
        "off": BARBELL_EDGE_POLICY_NONE,
        "disabled": BARBELL_EDGE_POLICY_NONE,
    }
    policy = aliases.get(policy, policy)
    if policy not in ALLOWED_BARBELL_EDGE_POLICIES:
        raise ValueError(
            f"unsupported barbell_edge_policy: {value!r} / allowed={ALLOWED_BARBELL_EDGE_POLICIES}"
        )
    if backend != POSE_BACKEND_MEDIAPIPE_BARBELL and policy != BARBELL_EDGE_POLICY_NONE:
        raise ValueError("barbell_edge_policy is only valid for pose_backend='mediapipe_barbell'")
    return policy


def normalize_derivative_mode(value: Any) -> str:
    mode = str(value or "pose").strip().lower().replace("-", "_").replace(" ", "_")
    mode = DERIVATIVE_MODE_ALIASES.get(mode, mode)
    if mode not in ALLOWED_DERIVATIVE_MODES:
        raise ValueError(f"unsupported derivative_mode: {value!r} / allowed={ALLOWED_DERIVATIVE_MODES}")
    return mode


def normalize_phase_head_type(value: Any) -> str:
    head = str(value or "mlp").strip().lower().replace("-", "_").replace(" ", "_")
    head = PHASE_HEAD_TYPE_ALIASES.get(head, head)
    if head not in ALLOWED_PHASE_HEAD_TYPES:
        raise ValueError(f"unsupported phase_head_type: {value!r} / allowed={ALLOWED_PHASE_HEAD_TYPES}")
    return head


def normalize_phase_conditioning(value: Any, phase_head_type: Any = PHASE_HEAD_MLP) -> str:
    head_type = normalize_phase_head_type(phase_head_type)
    raw = value
    try:
        missing = bool(pd.isna(raw))
    except Exception:
        missing = False
    if raw is None or missing:
        raw = (
            PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL
            if head_type in EXERCISE_CONDITIONED_PHASE_HEADS
            else PHASE_CONDITIONING_NONE
        )
    conditioning = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    conditioning = PHASE_CONDITIONING_ALIASES.get(conditioning, conditioning)
    if conditioning not in ALLOWED_PHASE_CONDITIONING:
        raise ValueError(f"unsupported phase_conditioning: {value!r} / allowed={ALLOWED_PHASE_CONDITIONING}")
    if head_type not in EXERCISE_CONDITIONED_PHASE_HEADS and conditioning != PHASE_CONDITIONING_NONE:
        raise ValueError(
            "non-none phase_conditioning requires an exercise-conditioned phase head "
            f"{EXERCISE_CONDITIONED_PHASE_HEADS}"
        )
    if head_type in EXERCISE_CONDITIONED_PHASE_HEADS and conditioning == PHASE_CONDITIONING_NONE:
        raise ValueError(
            f"phase_head_type={head_type!r} requires explicit phase_conditioning="
            "'ground_truth_exercise_label' or 'predicted_action'"
        )
    return conditioning


def exercise_id_source_for_conditioning(conditioning: Any) -> str:
    conditioning = str(conditioning)
    if conditioning == PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL:
        return EXERCISE_ID_SOURCE_GROUND_TRUTH_LABEL
    if conditioning == PHASE_CONDITIONING_PREDICTED_ACTION:
        return EXERCISE_ID_SOURCE_PREDICTED_ACTION
    return EXERCISE_ID_SOURCE_NONE


def normalize_phase_label_scheme(value: Any) -> str:
    scheme = str(value or PHASE_LABEL_SCHEME_AS_LABELED).strip().lower().replace("-", "_").replace(" ", "_")
    scheme = PHASE_LABEL_SCHEME_ALIASES.get(scheme, scheme)
    if scheme not in ALLOWED_PHASE_LABEL_SCHEMES:
        raise ValueError(f"unsupported phase_label_scheme: {value!r} / allowed={ALLOWED_PHASE_LABEL_SCHEMES}")
    return scheme


def normalize_exercise_type(value: Any) -> str:
    exercise = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bench": "benchpress",
        "bench_press": "benchpress",
        "benchpress": "benchpress",
        "dead": "deadlift",
        "dead_lift": "deadlift",
        "deadlift": "deadlift",
        "squat": "squat",
        "squats": "squat",
    }
    exercise = aliases.get(exercise, exercise)
    if exercise not in CLASS_TO_ID:
        raise ValueError(f"unsupported exercise_type: {value!r} / allowed={tuple(CLASS_LIST)}")
    return exercise


def uses_direct_barbell_box_head(value: Any) -> bool:
    return normalize_phase_head_type(value) == PHASE_HEAD_BARBELL_BOX_MLP


def uses_exercise_conditioned_phase_head(value: Any) -> bool:
    return normalize_phase_head_type(value) in EXERCISE_CONDITIONED_PHASE_HEADS


def derivative_mode_input_channels(mode: Any) -> int:
    mode = normalize_derivative_mode(mode)
    channels = 3  # normalized x, y, visibility
    if mode in {"velocity", "velocity_acceleration"}:
        channels += 2  # dx, dy
    if mode in {"acceleration", "velocity_acceleration"}:
        channels += 2  # ddx, ddy
    return channels


def normalize_phase_aux_inputs(value: Any) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        raw_parts = value.replace("+", ",").replace(";", ",").split(",")
    elif isinstance(value, Iterable):
        raw_parts = list(value)
    else:
        raw_parts = [value]
    normalized: List[str] = []
    for raw in raw_parts:
        token = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        token = PHASE_AUX_INPUT_ALIASES.get(token, token)
        if not token:
            continue
        if token not in ALLOWED_PHASE_AUX_INPUTS:
            raise ValueError(f"unsupported phase_aux_inputs token: {raw!r} / allowed={ALLOWED_PHASE_AUX_INPUTS}")
        if token not in normalized:
            normalized.append(token)
    return tuple(normalized)


def phase_aux_feature_dim(clip_len: int, num_kpt: int, phase_aux_inputs: Sequence[str]) -> int:
    L = int(clip_len)
    V = int(num_kpt)
    dim = 0
    for token in normalize_phase_aux_inputs(phase_aux_inputs):
        if token == PHASE_AUX_WRIST:
            dim += L * 2 * 3
        elif token == PHASE_AUX_BARBELL:
            dim += L * 1 * 3
        elif token == PHASE_AUX_ACCELERATION:
            dim += L * V * 2
        else:  # defensive; normalizer should catch this
            raise ValueError(f"unsupported phase aux input: {token!r}")
    return int(dim)


def normalize_cfg(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    raw_cfg: Dict[str, Any] = dict(cfg) if cfg else {}
    out: Dict[str, Any] = copy.deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    if cfg:
        out.update(raw_cfg)

    out["model_type"] = str(out.get("model_type", "mlp")).lower()
    if out["model_type"] not in {"mlp", "lstm"}:
        raise ValueError(f"unsupported model_type: {out['model_type']}")
    out["phase_head_type"] = normalize_phase_head_type(out.get("phase_head_type", "mlp"))
    if out["model_type"] != "mlp" and out["phase_head_type"] not in {
        PHASE_HEAD_MLP,
        PHASE_HEAD_EXERCISE_ATTN,
        PHASE_HEAD_PER_EXERCISE_MLP,
    }:
        raise ValueError("model_type='lstm' supports only mlp/exercise_attn/per_exercise_mlp phase heads")
    out["phase_conditioning"] = normalize_phase_conditioning(
        raw_cfg.get("phase_conditioning"),
        out["phase_head_type"],
    )
    out["exercise_id_source"] = exercise_id_source_for_conditioning(out["phase_conditioning"])
    out["phase_label_scheme"] = normalize_phase_label_scheme(
        out.get("phase_label_scheme", PHASE_LABEL_SCHEME_AS_LABELED)
    )

    if "phase_pooling" not in out or out.get("phase_pooling") in {None, ""}:
        out["phase_pooling"] = "temporal_avg"
    out["phase_pooling"] = str(out["phase_pooling"])
    if out["model_type"] == "mlp" and out["phase_pooling"] not in ALLOWED_PHASE_POOLING:
        raise ValueError(f"unsupported phase_pooling: {out['phase_pooling']} / allowed={ALLOWED_PHASE_POOLING}")

    out["pose_backend"] = normalize_pose_backend(out.get("pose_backend", POSE_BACKEND_MEDIAPIPE))
    edge_policy_value = raw_cfg.get("barbell_edge_policy", None)
    if out["pose_backend"] == POSE_BACKEND_MEDIAPIPE_BARBELL and edge_policy_value == BARBELL_EDGE_POLICY_NONE:
        # Preserve the hybrid default requested by the experiment plan even when
        # callers start from DEFAULT_EXPERIMENT_CONFIG, whose non-hybrid default
        # is intentionally "none".  Use "no_pose_edges" to request a disconnected
        # barbell node explicitly.
        edge_policy_value = None
    out["barbell_edge_policy"] = normalize_barbell_edge_policy(edge_policy_value, out["pose_backend"])
    if uses_direct_barbell_box_head(out["phase_head_type"]) and out["pose_backend"] != POSE_BACKEND_BARBELL:
        raise ValueError("phase_head_type='barbell_box_mlp' requires pose_backend='barbell'")
    spec = pose_backend_spec(out["pose_backend"])
    out["joint_subset"] = normalize_joint_subset(out.get("joint_subset", JOINT_SUBSET_ALL))
    if out["pose_backend"] == POSE_BACKEND_MEDIAPIPE_BARBELL and out["joint_subset"] not in {
        JOINT_SUBSET_ALL,
        JOINT_SUBSET_POSE_ONLY,
    }:
        raise ValueError("pose_backend='mediapipe_barbell' supports joint_subset='all' or 'pose_only'")
    view = joint_subset_view(
        out["pose_backend"],
        out["joint_subset"],
        barbell_edge_policy=out["barbell_edge_policy"],
    )
    out["source_num_kpt"] = int(spec.num_kpt)
    out["source_graph_id"] = str(view.graph_id if out["pose_backend"] == POSE_BACKEND_MEDIAPIPE_BARBELL else spec.graph_id)
    out["model_num_kpt"] = int(view.num_kpt)
    out["num_kpt"] = int(view.num_kpt)
    out["pose_graph_id"] = str(view.graph_id)
    out["model_graph_id"] = str(view.graph_id)
    out["selected_joint_indices"] = list(view.indices) if view.indices is not None else []
    out["selected_joint_names"] = list(view.names)
    out["yolo_missing_policy"] = str(out.get("yolo_missing_policy", "error")).strip().lower()
    if out["yolo_missing_policy"] not in {"error", "skip"}:
        raise ValueError("yolo_missing_policy must be 'error' or 'skip'")
    out["barbell_missing_policy"] = str(out.get("barbell_missing_policy", "error")).strip().lower()
    if out["barbell_missing_policy"] not in {"error", "skip"}:
        raise ValueError("barbell_missing_policy must be 'error' or 'skip'")

    out["derivative_mode"] = normalize_derivative_mode(out.get("derivative_mode", "pose"))
    if uses_direct_barbell_box_head(out["phase_head_type"]):
        out["input_channels"] = BARBELL_BOX_INPUT_CHANNELS
    else:
        out["input_channels"] = derivative_mode_input_channels(out["derivative_mode"])
    out["phase_aux_inputs"] = list(normalize_phase_aux_inputs(out.get("phase_aux_inputs", [])))
    if out["phase_aux_inputs"] and uses_direct_barbell_box_head(out["phase_head_type"]):
        raise ValueError("phase_aux_inputs are not supported with phase_head_type='barbell_box_mlp'")
    out["phase_aux_dim"] = phase_aux_feature_dim(
        int(out["clip_len"]),
        int(out["num_kpt"]),
        out["phase_aux_inputs"],
    )

    for key in ["hidden", "lstm_layers", "clip_len", "train_stride", "epochs", "batch", "num_workers", "seed"]:
        out[key] = int(out[key])
    for key in ["dropout", "lr", "weight_decay", "phase_loss_alpha"]:
        out[key] = float(out[key])
    for key in [
        "aug",
        "use_team_split",
        "write_meta_csv",
        "extract_missing_pose",
        "resume",
        "skip_completed",
        "force_retrain",
        "overwrite_existing",
        "pin_memory",
    ]:
        out[key] = _coerce_bool(out[key])
    if out.get("fresh_run_tag") in {"", None, "None"}:
        out["fresh_run_tag"] = None
    else:
        out["fresh_run_tag"] = str(out["fresh_run_tag"])
    if out.get("max_videos_per_split_type") in {"", None, "None"}:
        out["max_videos_per_split_type"] = None
    else:
        out["max_videos_per_split_type"] = int(out["max_videos_per_split_type"])

    for key in [
        "root",
        "video_dir",
        "label_dir",
        "work_dir",
        "pose_dir",
        "yolo_pose_dir",
        "barbell_dir",
        "team_pose_dir",
        "team_meta_csv",
        "meta_csv",
        "output_root",
    ]:
        out[key] = str(_resolve_path(out[key]))
    out["run_kind"] = str(out.get("run_kind", "legacy"))
    return out


def is_incomplete(val: Any) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, str):
        s = val.strip()
        try:
            float(s)
            return False
        except ValueError:
            return True
    return False


def parse_label_row(row: pd.Series, max_reps: int = 80) -> Dict[str, Any]:
    cnt = int(row["count"]) if "count" in row.index and pd.notna(row.get("count")) else 0
    reps: List[Tuple[int, int, int]] = []
    incomplete = None
    for i in range(max_reps):
        s_col, b_col, f_col = f"L{3 * i + 1}", f"L{3 * i + 2}", f"L{3 * i + 3}"
        if s_col not in row.index:
            break
        s, b, f_ = row.get(s_col), row.get(b_col), row.get(f_col)
        if pd.isna(s) and pd.isna(b) and pd.isna(f_):
            break
        if is_incomplete(s) or is_incomplete(b) or is_incomplete(f_):
            incomplete = i
            break
        try:
            s_i, b_i, f_i = int(float(s)), int(float(b)), int(float(f_))
            if f_i > s_i:
                reps.append((s_i, b_i, f_i))
        except (ValueError, TypeError):
            break
    return {"count": cnt, "reps": reps, "incomplete": incomplete}


def load_labels(label_dir: Path | str = LABEL_DIR, class_to_id: Mapping[str, int] = CLASS_TO_ID) -> Dict[Tuple[str, str], Dict[str, Any]]:
    label_dir = Path(label_dir)
    csv_list = sorted(label_dir.glob("*.csv"))
    if not csv_list:
        raise FileNotFoundError(f"label CSV not found: {label_dir}")
    raw_df = pd.read_csv(csv_list[0])
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    raw_df = raw_df.loc[:, ~raw_df.columns.str.startswith("Unnamed")]

    labels: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in raw_df.iterrows():
        typ = str(row["type"]).strip()
        name = str(row["name"]).strip()
        if typ not in class_to_id:
            continue
        info = parse_label_row(row)
        info["cls"] = int(class_to_id[typ])
        labels[(typ, name)] = info
    return labels


def get_mp_pose():
    import mediapipe as mp

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
        raise RuntimeError(
            "mediapipe.solutions.pose is unavailable. Existing pose caches can be used, "
            "but new pose extraction requires a MediaPipe build with the solutions API."
        )
    return mp.solutions.pose


def cache_path_in(base_dir: Path | str, typ: str, name: str) -> Path:
    return Path(base_dir) / typ / f"{Path(name).stem}.npz"


def pose_path(
    typ: str,
    name: str,
    pose_dir: Path | str = POSE_DIR,
    team_pose_dir: Path | str = TEAM_POSE_DIR,
    *,
    pose_backend: str = POSE_BACKEND_MEDIAPIPE,
    yolo_pose_dir: Path | str = YOLO_POSE_DIR,
    barbell_dir: Path | str = BARBELL_DIR,
) -> Path:
    backend = normalize_pose_backend(pose_backend)
    if backend == POSE_BACKEND_YOLO:
        return cache_path_in(yolo_pose_dir, typ, name)
    if backend == POSE_BACKEND_BARBELL:
        return cache_path_in(barbell_dir, typ, name)
    team_p = cache_path_in(team_pose_dir, typ, name)
    return team_p if team_p.exists() else cache_path_in(pose_dir, typ, name)


def pose_out_path(
    typ: str,
    name: str,
    pose_dir: Path | str = POSE_DIR,
    *,
    pose_backend: str = POSE_BACKEND_MEDIAPIPE,
    yolo_pose_dir: Path | str = YOLO_POSE_DIR,
    barbell_dir: Path | str = BARBELL_DIR,
) -> Path:
    backend = normalize_pose_backend(pose_backend)
    if backend == POSE_BACKEND_YOLO:
        return cache_path_in(yolo_pose_dir, typ, name)
    if backend == POSE_BACKEND_BARBELL:
        return cache_path_in(barbell_dir, typ, name)
    return cache_path_in(pose_dir, typ, name)


def extract_pose(video_path: Path | str, out_path: Path | str, model_complexity: int = 1) -> Tuple[int, float]:
    import cv2

    video_path = Path(video_path)
    out_path = Path(out_path)
    mp_pose = get_mp_pose()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"open fail: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    kpts = []
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        arr = np.zeros((NUM_KPT, 3), dtype=np.float32)
        if res.pose_landmarks is not None:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                arr[i] = [lm.x, lm.y, lm.visibility]
        kpts.append(arr)
    pose.close()
    cap.release()
    kpts_arr = np.asarray(kpts, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, kpts=kpts_arr, fps=fps, num_frames=len(kpts_arr), source=str(video_path))
    return len(kpts_arr), fps


def ensure_pose_cache(
    labels: Mapping[Tuple[str, str], Dict[str, Any]],
    video_dir: Path | str = VIDEO_DIR,
    pose_dir: Path | str = POSE_DIR,
    team_pose_dir: Path | str = TEAM_POSE_DIR,
    yolo_pose_dir: Path | str = YOLO_POSE_DIR,
    barbell_dir: Path | str = BARBELL_DIR,
    pose_backend: str = POSE_BACKEND_MEDIAPIPE,
    extract_missing: bool = False,
    verbose: bool = True,
) -> List[Tuple[Path, Path]]:
    backend = normalize_pose_backend(pose_backend)
    video_dir, pose_dir, team_pose_dir, yolo_pose_dir, barbell_dir = (
        Path(video_dir),
        Path(pose_dir),
        Path(team_pose_dir),
        Path(yolo_pose_dir),
        Path(barbell_dir),
    )
    todo: List[Tuple[Path, Path]] = []
    for (typ, name), _ in labels.items():
        src = video_dir / typ / name
        if backend == POSE_BACKEND_MEDIAPIPE_BARBELL:
            required = (
                (
                    pose_path(
                        typ,
                        name,
                        pose_dir=pose_dir,
                        team_pose_dir=team_pose_dir,
                        pose_backend=POSE_BACKEND_MEDIAPIPE,
                        yolo_pose_dir=yolo_pose_dir,
                        barbell_dir=barbell_dir,
                    ),
                    pose_out_path(
                        typ,
                        name,
                        pose_dir=pose_dir,
                        pose_backend=POSE_BACKEND_MEDIAPIPE,
                        yolo_pose_dir=yolo_pose_dir,
                        barbell_dir=barbell_dir,
                    ),
                ),
                (
                    pose_path(
                        typ,
                        name,
                        pose_dir=pose_dir,
                        team_pose_dir=team_pose_dir,
                        pose_backend=POSE_BACKEND_BARBELL,
                        yolo_pose_dir=yolo_pose_dir,
                        barbell_dir=barbell_dir,
                    ),
                    pose_out_path(
                        typ,
                        name,
                        pose_dir=pose_dir,
                        pose_backend=POSE_BACKEND_BARBELL,
                        yolo_pose_dir=yolo_pose_dir,
                        barbell_dir=barbell_dir,
                    ),
                ),
            )
            for cached, out_path in required:
                if src.exists() and not cached.exists():
                    todo.append((src, out_path))
            continue
        cached = pose_path(
            typ,
            name,
            pose_dir=pose_dir,
            team_pose_dir=team_pose_dir,
            pose_backend=backend,
            yolo_pose_dir=yolo_pose_dir,
            barbell_dir=barbell_dir,
        )
        if src.exists() and not cached.exists():
            todo.append(
                (
                    src,
                    pose_out_path(
                        typ,
                        name,
                        pose_dir=pose_dir,
                        pose_backend=backend,
                        yolo_pose_dir=yolo_pose_dir,
                        barbell_dir=barbell_dir,
                    ),
                )
            )
    if todo and extract_missing:
        if backend == POSE_BACKEND_YOLO:
            raise RuntimeError(
                "extract_missing_pose=True with pose_backend='yolo' will not call the MediaPipe extractor. "
                "Run `python model/extract_pose_yolo.py` first, or set extract_missing_pose=False and provide YOLO npz files."
            )
        if backend == POSE_BACKEND_BARBELL:
            raise RuntimeError(
                "extract_missing_pose=True with pose_backend='barbell' will not call the YOLO-World extractor. "
                "Run `python model/extract_barbell_yolo_world.py` first, or set extract_missing_pose=False and provide barbell npz files."
            )
        if backend == POSE_BACKEND_MEDIAPIPE_BARBELL:
            raise RuntimeError(
                "extract_missing_pose=True with pose_backend='mediapipe_barbell' cannot build the hybrid cache automatically. "
                "Provide both MediaPipe pose npz files and barbell YOLO-World npz files first."
            )
        if verbose:
            print(f"pose extraction targets: {len(todo)}")
        for src, dst in tqdm(todo, desc="pose", leave=False):
            try:
                extract_pose(src, dst)
            except Exception as exc:  # pragma: no cover - extraction environment dependent
                print("FAIL:", src, exc)
    elif verbose:
        print("pose extraction skipped" if not todo else f"pose cache missing for {len(todo)} videos (extract_missing_pose=False)")
    return todo


def build_meta_df(
    labels: Mapping[Tuple[str, str], Dict[str, Any]],
    video_dir: Path | str = VIDEO_DIR,
    pose_dir: Path | str = POSE_DIR,
    team_pose_dir: Path | str = TEAM_POSE_DIR,
    yolo_pose_dir: Path | str = YOLO_POSE_DIR,
    barbell_dir: Path | str = BARBELL_DIR,
    pose_backend: str = POSE_BACKEND_MEDIAPIPE,
    yolo_missing_policy: str = "error",
    barbell_missing_policy: str = "error",
    barbell_edge_policy: Any = None,
    team_meta_csv: Path | str = TEAM_META_CSV,
    use_team_split: bool = True,
    seed: int = SEED,
) -> pd.DataFrame:
    backend = normalize_pose_backend(pose_backend)
    edge_policy = normalize_barbell_edge_policy(barbell_edge_policy, backend)
    spec = pose_backend_spec(backend)
    missing_policy_value = barbell_missing_policy if backend in {POSE_BACKEND_BARBELL, POSE_BACKEND_MEDIAPIPE_BARBELL} else yolo_missing_policy
    missing_policy = str(missing_policy_value or "error").strip().lower()
    if missing_policy not in {"error", "skip"}:
        raise ValueError("missing policy must be 'error' or 'skip'")
    video_dir, pose_dir, team_pose_dir, yolo_pose_dir, barbell_dir, team_meta_csv = (
        Path(video_dir),
        Path(pose_dir),
        Path(team_pose_dir),
        Path(yolo_pose_dir),
        Path(barbell_dir),
        Path(team_meta_csv),
    )
    records: List[Dict[str, Any]] = []
    for (typ, name), info in labels.items():
        src = video_dir / typ / name
        if not src.exists():
            continue

        if backend == POSE_BACKEND_MEDIAPIPE_BARBELL:
            pose_npz = pose_path(
                typ,
                name,
                pose_dir=pose_dir,
                team_pose_dir=team_pose_dir,
                pose_backend=POSE_BACKEND_MEDIAPIPE,
                yolo_pose_dir=yolo_pose_dir,
                barbell_dir=barbell_dir,
            )
            barbell_npz = pose_path(
                typ,
                name,
                pose_dir=pose_dir,
                team_pose_dir=team_pose_dir,
                pose_backend=POSE_BACKEND_BARBELL,
                yolo_pose_dir=yolo_pose_dir,
                barbell_dir=barbell_dir,
            )
            missing = [p for p in (pose_npz, barbell_npz) if not p.exists()]
            if missing:
                if missing_policy == "error":
                    missing_text = ", ".join(str(p) for p in missing)
                    raise FileNotFoundError(
                        f"hybrid cache missing for {typ}/{name}: {missing_text}. "
                        "Run MediaPipe and YOLO-World barbell extractors first, or set barbell_missing_policy='skip'."
                    )
                continue
            with np.load(pose_npz) as pose_data, np.load(barbell_npz) as barbell_data:
                pose_kpts = np.asarray(pose_data["kpts"], dtype=np.float32)
                if pose_kpts.ndim != 3 or pose_kpts.shape[1] != NUM_KPT or pose_kpts.shape[2] < 3:
                    raise ValueError(
                        f"mediapipe pose cache has invalid shape for {typ}/{name}: "
                        f"expected [T,{NUM_KPT},3], got {tuple(pose_kpts.shape)} at {pose_npz}"
                    )
                barbell_kpts = barbell_center_kpts_from_npz(barbell_data)
                alignment = temporal_alignment_summary(len(pose_kpts), len(barbell_kpts))
                if alignment["alignment_status"] == "failed":
                    raise ValueError(
                        f"hybrid temporal alignment failed for {typ}/{name}: "
                        f"pose_T={alignment['pose_T']} barbell_T={alignment['barbell_T']}"
                    )
                fps = float(pose_data["fps"]) if "fps" in pose_data.files else (float(barbell_data["fps"]) if "fps" in barbell_data.files else 30.0)
            view = joint_subset_view(backend, JOINT_SUBSET_ALL, barbell_edge_policy=edge_policy)
            T = int(alignment["T_used"])
            records.append(
                {
                    "type": typ,
                    "name": name,
                    "cls": int(info["cls"]),
                    "T": T,
                    "T_used": T,
                    "fps": fps,
                    "n_reps": len(info["reps"]),
                    "has_incomplete": info["incomplete"] is not None,
                    "npz": str(pose_npz),
                    "pose_npz": str(pose_npz),
                    "barbell_npz": str(barbell_npz),
                    "video_path": str(src),
                    "pose_backend": backend,
                    "barbell_edge_policy": edge_policy,
                    "num_kpt": int(view.num_kpt),
                    "source_num_kpt": int(view.num_kpt),
                    "model_num_kpt": int(view.num_kpt),
                    "pose_graph_id": view.graph_id,
                    "source_graph_id": view.graph_id,
                    "model_graph_id": view.graph_id,
                    **alignment,
                }
            )
            continue

        npz = pose_path(
            typ,
            name,
            pose_dir=pose_dir,
            team_pose_dir=team_pose_dir,
            pose_backend=backend,
            yolo_pose_dir=yolo_pose_dir,
            barbell_dir=barbell_dir,
        )
        if not npz.exists():
            if backend in {POSE_BACKEND_YOLO, POSE_BACKEND_BARBELL} and missing_policy == "error":
                raise FileNotFoundError(
                    f"{backend} cache missing for {typ}/{name}: {npz}. "
                    "Run the matching extractor first, or set the missing policy to 'skip'."
                )
            continue
        with np.load(npz) as d:
            kpts = d["kpts"]
            if kpts.ndim != 3 or kpts.shape[1] != spec.num_kpt or kpts.shape[2] < 3:
                raise ValueError(
                    f"{backend} pose cache has invalid shape for {typ}/{name}: "
                    f"expected [T,{spec.num_kpt},3], got {tuple(kpts.shape)} at {npz}"
                )
            T = int(d["num_frames"]) if "num_frames" in d else int(kpts.shape[0])
            fps = float(d["fps"]) if "fps" in d else 30.0
        records.append(
            {
                "type": typ,
                "name": name,
                "cls": int(info["cls"]),
                "T": T,
                "fps": fps,
                "n_reps": len(info["reps"]),
                "has_incomplete": info["incomplete"] is not None,
                "npz": str(npz),
                "video_path": str(src),
                "pose_backend": backend,
                "barbell_edge_policy": edge_policy,
                "num_kpt": int(spec.num_kpt),
                "source_num_kpt": int(spec.num_kpt),
                "model_num_kpt": int(spec.num_kpt),
                "pose_graph_id": spec.graph_id,
                "source_graph_id": spec.graph_id,
                "model_graph_id": spec.graph_id,
            }
        )
    meta = pd.DataFrame(records)
    if meta.empty:
        raise RuntimeError("no metadata rows were built; check labels, videos, and pose caches")

    if use_team_split and team_meta_csv.exists():
        team = pd.read_csv(team_meta_csv)
        if {"type", "name", "split"}.issubset(team.columns):
            meta = meta.merge(team[["type", "name", "split"]].drop_duplicates(), on=["type", "name"], how="left")
            idx_need = meta[meta["split"].isna()].index
            if len(idx_need) >= NUM_CLASSES:
                strat = meta.loc[idx_need, "cls"] if meta.loc[idx_need, "cls"].nunique() > 1 else None
                tr, va = train_test_split(idx_need, test_size=0.2, random_state=seed, stratify=strat)
                meta.loc[tr, "split"] = "train"
                meta.loc[va, "split"] = "val"
            elif len(idx_need):
                meta.loc[idx_need, "split"] = "train"
            return meta

    strat = meta["cls"] if meta["cls"].nunique() > 1 else None
    tr, va = train_test_split(np.arange(len(meta)), test_size=0.2, random_state=seed, stratify=strat)
    meta["split"] = "train"
    meta.loc[va, "split"] = "val"
    return meta

def limit_meta_per_split_type(meta: pd.DataFrame, max_videos_per_split_type: Optional[int]) -> pd.DataFrame:
    if not max_videos_per_split_type:
        return meta.reset_index(drop=True)
    limited = (
        meta.sort_values(["split", "type", "name"])
        .groupby(["split", "type"], group_keys=False)
        .head(int(max_videos_per_split_type))
        .reset_index(drop=True)
    )
    return limited


def data_split_fingerprint(meta: pd.DataFrame) -> str:
    cols = [
        c
        for c in [
            "split",
            "type",
            "name",
            "T",
            "T_used",
            "npz",
            "pose_npz",
            "barbell_npz",
            "barbell_edge_policy",
            "alignment_status",
        ]
        if c in meta.columns
    ]
    payload = meta[cols].sort_values(cols).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def prepare_context(
    cfg: Optional[Mapping[str, Any]] = None,
    *,
    verbose: bool = True,
    update_globals: bool = True,
) -> ExperimentContext:
    norm = normalize_cfg(cfg)
    seed_everything(int(norm["seed"]))

    root = Path(norm["root"])
    video_dir = Path(norm["video_dir"])
    label_dir = Path(norm["label_dir"])
    work_dir = Path(norm["work_dir"])
    pose_dir = Path(norm["pose_dir"])
    yolo_pose_dir = Path(norm["yolo_pose_dir"])
    barbell_dir = Path(norm["barbell_dir"])
    team_pose_dir = Path(norm["team_pose_dir"])
    team_meta_csv = Path(norm["team_meta_csv"])
    meta_csv = Path(norm["meta_csv"])
    ablation_root = Path(norm["output_root"])
    ablation_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    if norm["pose_backend"] == POSE_BACKEND_YOLO:
        yolo_pose_dir.mkdir(parents=True, exist_ok=True)
    elif norm["pose_backend"] == POSE_BACKEND_BARBELL:
        barbell_dir.mkdir(parents=True, exist_ok=True)
    elif norm["pose_backend"] == POSE_BACKEND_MEDIAPIPE_BARBELL:
        pose_dir.mkdir(parents=True, exist_ok=True)
        barbell_dir.mkdir(parents=True, exist_ok=True)
    else:
        pose_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(label_dir)
    pose_todo = ensure_pose_cache(
        labels,
        video_dir=video_dir,
        pose_dir=pose_dir,
        team_pose_dir=team_pose_dir,
        yolo_pose_dir=yolo_pose_dir,
        barbell_dir=barbell_dir,
        pose_backend=str(norm["pose_backend"]),
        extract_missing=bool(norm["extract_missing_pose"]),
        verbose=verbose,
    )
    meta = build_meta_df(
        labels,
        video_dir=video_dir,
        pose_dir=pose_dir,
        team_pose_dir=team_pose_dir,
        yolo_pose_dir=yolo_pose_dir,
        barbell_dir=barbell_dir,
        pose_backend=str(norm["pose_backend"]),
        yolo_missing_policy=str(norm["yolo_missing_policy"]),
        barbell_missing_policy=str(norm["barbell_missing_policy"]),
        barbell_edge_policy=str(norm["barbell_edge_policy"]),
        team_meta_csv=team_meta_csv,
        use_team_split=bool(norm["use_team_split"]),
        seed=int(norm["seed"]),
    )
    meta = limit_meta_per_split_type(meta, norm.get("max_videos_per_split_type"))
    if bool(norm["write_meta_csv"]):
        meta_csv.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_csv, index=False, encoding="utf-8-sig")
    train = meta[meta["split"] == "train"].reset_index(drop=True)
    val = meta[meta["split"] == "val"].reset_index(drop=True)
    if train.empty or val.empty:
        raise RuntimeError(f"train/val split empty: train={len(train)} val={len(val)}")
    if verbose:
        print("device:", DEVICE, "| torch:", torch.__version__, "| numpy:", np.__version__)
        print("ROOT     :", root)
        print("WORK_DIR :", work_dir)
        print(
            "POSE     :",
            norm["pose_backend"],
            "| source_V:",
            norm["source_num_kpt"],
            "| model_V:",
            norm["model_num_kpt"],
            "| subset:",
            norm["joint_subset"],
            "| graph:",
            norm["pose_graph_id"],
            "| barbell_edge_policy:",
            norm["barbell_edge_policy"],
        )
        print("labels   :", len(labels))
        print(meta.groupby(["split", "type"]).size().unstack(fill_value=0).reindex(["train", "val"]))

    context = ExperimentContext(
        cfg=norm,
        project_root=PROJECT_ROOT,
        root=root,
        video_dir=video_dir,
        label_dir=label_dir,
        work_dir=work_dir,
        pose_dir=pose_dir,
        yolo_pose_dir=yolo_pose_dir,
        barbell_dir=barbell_dir,
        team_pose_dir=team_pose_dir,
        team_meta_csv=team_meta_csv,
        meta_csv=meta_csv,
        pose_backend=str(norm["pose_backend"]),
        num_kpt=int(norm["source_num_kpt"]),
        pose_graph_id=str(norm["source_graph_id"]),
        labels=labels,
        meta_df=meta,
        train_meta=train,
        val_meta=val,
        ablation_root=ablation_root,
        ablation_log_csv=ablation_root / "ablation_results.csv",
        ablation_log_json=ablation_root / "ablation_results.json",
        device=DEVICE,
        pose_todo=pose_todo,
    )

    if update_globals:
        global LABELS, meta_df, train_meta, val_meta, ACTIVE_CONTEXT
        LABELS = labels
        meta_df = meta
        train_meta = train
        val_meta = val
        ACTIVE_CONTEXT = context
    return context


# Normalization + phase targets
L_HIP, R_HIP = 23, 24
L_SHO, R_SHO = 11, 12


def normalize_kpts(k: np.ndarray, pose_backend: str = POSE_BACKEND_MEDIAPIPE) -> np.ndarray:
    backend = normalize_pose_backend(pose_backend)
    spec = pose_backend_spec(backend)
    arr = np.asarray(k, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != spec.num_kpt or arr.shape[2] < 3:
        raise ValueError(
            f"{spec.name} keypoints must have shape [T,{spec.num_kpt},3], got {tuple(arr.shape)}"
        )
    if backend == POSE_BACKEND_BARBELL:
        xy = arr[..., :2].copy()
        conf = arr[..., 2:3].copy()
        # Barbell caches store frame-normalized center coordinates in [0, 1].
        # Center them around the image midpoint so the existing ST-GCN input
        # receives a zero-centered one-node trajectory without needing a body
        # scale/hip-shoulder reference.
        xy = (xy - 0.5) * 2.0
        return np.concatenate([xy, conf], axis=-1).astype(np.float32)
    xy = arr[..., :2].copy()
    conf = arr[..., 2:3].copy()
    hip = (xy[:, spec.l_hip] + xy[:, spec.r_hip]) / 2
    sho = (xy[:, spec.l_sho] + xy[:, spec.r_sho]) / 2
    scale = np.linalg.norm(sho - hip, axis=1, keepdims=True) + 1e-6
    xy = (xy - hip[:, None, :]) / scale[:, None, :]
    return np.concatenate([xy, conf], axis=-1).astype(np.float32)


def temporal_alignment_summary(pose_T: int, barbell_T: int, tolerance_ratio: float = 0.01) -> Dict[str, Any]:
    pose_T = int(pose_T)
    barbell_T = int(barbell_T)
    T_used = min(pose_T, barbell_T)
    diff = abs(pose_T - barbell_T)
    tolerance_frames = max(1, int(math.floor(min(pose_T, barbell_T) * float(tolerance_ratio))))
    if pose_T <= 0 or barbell_T <= 0:
        status = "failed"
        warning = "one or both streams are empty"
    elif diff == 0:
        status = "exact"
        warning = ""
    elif diff <= tolerance_frames:
        status = "min_aligned_within_tolerance"
        warning = f"trimmed to min stream length; frame_count_diff={diff}"
    else:
        status = "failed"
        warning = f"frame_count_diff={diff} exceeds tolerance_frames={tolerance_frames}"
    return {
        "alignment_status": status,
        "alignment_policy": "min_trim",
        "alignment_warning": warning,
        "pose_T": pose_T,
        "barbell_T": barbell_T,
        "T_used": int(T_used),
        "frame_count_diff": int(diff),
        "tolerance_frames": int(tolerance_frames),
    }


def barbell_center_kpts_from_npz(npz: Any) -> np.ndarray:
    files = set(getattr(npz, "files", []))
    if "kpts" not in files:
        raise KeyError("barbell npz must contain kpts shaped [T,1,3]")
    kpts = np.asarray(npz["kpts"], dtype=np.float32)
    if kpts.ndim != 3 or kpts.shape[1] != 1 or kpts.shape[2] < 2:
        raise ValueError(f"barbell keypoints must have shape [T,1,>=2], got {tuple(kpts.shape)}")
    out = np.zeros((kpts.shape[0], 1, 3), dtype=np.float32)
    out[..., :2] = kpts[..., :2]
    out[..., 2] = np.clip(kpts[..., 2] if kpts.shape[2] >= 3 else 1.0, 0.0, 1.0)
    return out


def normalize_mediapipe_barbell_kpts(pose_kpts: np.ndarray, barbell_kpts: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_kpts, dtype=np.float32)
    barbell = np.asarray(barbell_kpts, dtype=np.float32)
    if pose.ndim != 3 or pose.shape[1] != NUM_KPT or pose.shape[2] < 3:
        raise ValueError(f"MediaPipe pose must have shape [T,{NUM_KPT},3], got {tuple(pose.shape)}")
    if barbell.ndim != 3 or barbell.shape[1] != 1 or barbell.shape[2] < 3:
        raise ValueError(f"barbell keypoints must have shape [T,1,3], got {tuple(barbell.shape)}")
    if len(pose) != len(barbell):
        raise ValueError(f"pose/barbell length mismatch after alignment: {len(pose)} vs {len(barbell)}")
    fused = np.concatenate([pose[..., :3], barbell[..., :3]], axis=1).astype(np.float32, copy=False)
    return normalize_kpts(fused, pose_backend=POSE_BACKEND_MEDIAPIPE_BARBELL)


def _row_int(row: pd.Series, key: str, default: int) -> int:
    value = row.get(key, default)
    if pd.isna(value):
        return int(default)
    return int(value)


def load_model_keypoints_from_row(row: pd.Series, cfg: Mapping[str, Any]) -> ModelKeypointLoadResult:
    norm = normalize_cfg(cfg)
    backend = normalize_pose_backend(norm["pose_backend"])
    subset = normalize_joint_subset(norm["joint_subset"])
    head_type = normalize_phase_head_type(norm["phase_head_type"])
    if uses_direct_barbell_box_head(head_type):
        with np.load(row["npz"]) as npz:
            features = barbell_box_features_from_npz(npz)
        T = min(_row_int(row, "T_used", _row_int(row, "T", len(features))), len(features))
        alignment = {
            "alignment_status": "exact",
            "alignment_policy": "single_stream",
            "alignment_warning": "",
            "pose_T": int(T),
            "barbell_T": int(T),
            "T_used": int(T),
            "frame_count_diff": 0,
            "tolerance_frames": 0,
        }
        return ModelKeypointLoadResult(features[:T].astype(np.float32, copy=False), alignment)

    if backend == POSE_BACKEND_MEDIAPIPE_BARBELL:
        if "pose_npz" not in row.index or "barbell_npz" not in row.index:
            raise KeyError("hybrid rows must contain explicit pose_npz and barbell_npz columns")
        with np.load(row["pose_npz"]) as pose_data, np.load(row["barbell_npz"]) as barbell_data:
            pose_raw = np.asarray(pose_data["kpts"], dtype=np.float32)
            barbell_raw = barbell_center_kpts_from_npz(barbell_data)
        alignment = temporal_alignment_summary(len(pose_raw), len(barbell_raw))
        if alignment["alignment_status"] == "failed":
            raise ValueError(
                f"hybrid temporal alignment failed for {row.get('type')}/{row.get('name')}: "
                f"pose_T={alignment['pose_T']} barbell_T={alignment['barbell_T']}"
            )
        T = min(int(alignment["T_used"]), _row_int(row, "T_used", _row_int(row, "T", int(alignment["T_used"]))))
        fused = normalize_mediapipe_barbell_kpts(pose_raw[:T], barbell_raw[:T])
        return ModelKeypointLoadResult(select_joint_subset(fused, backend, subset), {**alignment, "T_used": int(T)})

    with np.load(row["npz"]) as npz:
        raw = np.asarray(npz["kpts"], dtype=np.float32)
    T = min(_row_int(row, "T_used", _row_int(row, "T", len(raw))), len(raw))
    normalized = normalize_kpts(raw[:T], pose_backend=backend)
    selected = select_joint_subset(normalized, pose_backend=backend, joint_subset=subset)
    alignment = {
        "alignment_status": "exact",
        "alignment_policy": "single_stream",
        "alignment_warning": "",
        "pose_T": int(T),
        "barbell_T": int(T if backend == POSE_BACKEND_BARBELL else 0),
        "T_used": int(T),
        "frame_count_diff": 0,
        "tolerance_frames": 0,
    }
    return ModelKeypointLoadResult(selected.astype(np.float32, copy=False), alignment)


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
    """Return model input features for one normalized causal pose window.

    Base channels are always normalized ``[x, y, visibility]``.  Derivative
    variants append finite-difference ``[dx, dy]`` and/or ``[ddx, ddy]`` after
    spatial augmentation, keeping visibility as a confidence channel rather
    than differentiating it as a motion signal.
    """
    mode = normalize_derivative_mode(derivative_mode)
    base = clip.astype(np.float32, copy=False)
    parts = [base]
    if mode in {"velocity", "velocity_acceleration"}:
        parts.append(temporal_xy_velocity(base))
    if mode in {"acceleration", "velocity_acceleration"}:
        parts.append(temporal_xy_acceleration(base))
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def _interpolate_temporal_features(raw: np.ndarray, valid: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    features = np.asarray(raw, dtype=np.float32).copy()
    if features.ndim != 2:
        raise ValueError(f"temporal features must be [T,C], got {tuple(features.shape)}")
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if len(features) != len(mask):
        raise ValueError(f"feature/mask length mismatch: {len(features)} vs {len(mask)}")
    if len(features) == 0:
        return features
    fallback = np.asarray(fallback, dtype=np.float32).reshape(-1)
    if not mask.any():
        features[:] = fallback[None, :]
        return features
    frame_idx = np.arange(len(features), dtype=np.float32)
    for dim in range(features.shape[1]):
        values = features[:, dim]
        dim_valid = mask & np.isfinite(values)
        if not dim_valid.any():
            features[:, dim] = fallback[min(dim, len(fallback) - 1)]
            continue
        features[:, dim] = np.interp(frame_idx, frame_idx[dim_valid], values[dim_valid]).astype(np.float32)
    return features


def barbell_box_features_from_arrays(
    *,
    boxes: Optional[np.ndarray] = None,
    kpts: Optional[np.ndarray] = None,
    detection_conf: Optional[np.ndarray] = None,
    detected: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build direct phase-head features from YOLO-World barbell boxes.

    Output shape is ``[T, 1, 6]`` with channels
    ``[x1, y1, x2, y2, confidence, detected]``.  Box coordinates are frame
    normalized by the extractor, interpolated across missing frames, clipped to
    ``[0, 1]``, then centered to ``[-1, 1]`` so the MLP sees a stable coordinate
    range.  Confidence and detected mask remain in ``[0, 1]``.
    """

    lengths = []
    if boxes is not None:
        lengths.append(len(np.asarray(boxes)))
    if kpts is not None:
        lengths.append(len(np.asarray(kpts)))
    if detection_conf is not None:
        lengths.append(len(np.asarray(detection_conf)))
    if detected is not None:
        lengths.append(len(np.asarray(detected)))
    T = int(max(lengths) if lengths else 0)
    if T <= 0:
        return np.zeros((0, 1, BARBELL_BOX_INPUT_CHANNELS), dtype=np.float32)

    if boxes is None:
        raw_boxes = np.full((T, 4), np.nan, dtype=np.float32)
        if kpts is not None:
            k = np.asarray(kpts, dtype=np.float32)
            if k.ndim == 3 and k.shape[1] >= 1 and k.shape[2] >= 2:
                n = min(T, len(k))
                raw_boxes[:n, 0] = k[:n, 0, 0]
                raw_boxes[:n, 1] = k[:n, 0, 1]
                raw_boxes[:n, 2] = k[:n, 0, 0]
                raw_boxes[:n, 3] = k[:n, 0, 1]
    else:
        raw_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(raw_boxes) < T:
            raw_boxes = np.pad(raw_boxes, ((0, T - len(raw_boxes)), (0, 0)), constant_values=np.nan)
        elif len(raw_boxes) > T:
            raw_boxes = raw_boxes[:T]

    if detection_conf is None and kpts is not None:
        k = np.asarray(kpts, dtype=np.float32)
        if k.ndim == 3 and k.shape[1] >= 1 and k.shape[2] >= 3:
            detection_conf = k[:, 0, 2]
    conf = np.zeros((T,), dtype=np.float32)
    if detection_conf is not None:
        conf_arr = np.asarray(detection_conf, dtype=np.float32).reshape(-1)
        n = min(T, len(conf_arr))
        conf[:n] = conf_arr[:n]
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.clip(conf, 0.0, 1.0)

    if detected is None:
        detected_arr = np.isfinite(raw_boxes).all(axis=1) & (conf > 0.0)
    else:
        detected_arr = np.zeros((T,), dtype=bool)
        det = np.asarray(detected, dtype=bool).reshape(-1)
        n = min(T, len(det))
        detected_arr[:n] = det[:n]
        detected_arr &= np.isfinite(raw_boxes).all(axis=1)

    boxes_filled = _interpolate_temporal_features(
        raw_boxes,
        detected_arr,
        fallback=np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )
    boxes_filled = np.clip(boxes_filled, 0.0, 1.0)
    x1 = np.minimum(boxes_filled[:, 0], boxes_filled[:, 2])
    y1 = np.minimum(boxes_filled[:, 1], boxes_filled[:, 3])
    x2 = np.maximum(boxes_filled[:, 0], boxes_filled[:, 2])
    y2 = np.maximum(boxes_filled[:, 1], boxes_filled[:, 3])
    coords = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    coords = (coords - 0.5) * 2.0

    features = np.zeros((T, 1, BARBELL_BOX_INPUT_CHANNELS), dtype=np.float32)
    features[:, 0, :4] = coords
    features[:, 0, 4] = conf
    features[:, 0, 5] = detected_arr.astype(np.float32)
    return features


def barbell_box_features_from_npz(npz: Any) -> np.ndarray:
    files = set(getattr(npz, "files", []))
    kpts = npz["kpts"] if "kpts" in files else None
    boxes = npz["boxes"] if "boxes" in files else None
    detection_conf = npz["detection_conf"] if "detection_conf" in files else None
    detected = npz["detected"] if "detected" in files else None
    return barbell_box_features_from_arrays(
        boxes=boxes,
        kpts=kpts,
        detection_conf=detection_conf,
        detected=detected,
    )


def apply_barbell_box_aug(clip: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Box-safe augmentation for ``barbell_box_mlp`` direct phase input."""

    out = clip.astype(np.float32, copy=True)
    if out.shape[-1] < 4:
        return out
    if np.random.rand() < p:
        x1 = out[..., 0].copy()
        x2 = out[..., 2].copy()
        out[..., 0] = -x2
        out[..., 2] = -x1
    if np.random.rand() < p:
        out[..., :4] += np.random.randn(*out[..., :4].shape).astype(np.float32) * 0.02
    out[..., :4] = np.clip(out[..., :4], -1.0, 1.0)
    x1 = np.minimum(out[..., 0], out[..., 2])
    y1 = np.minimum(out[..., 1], out[..., 3])
    x2 = np.maximum(out[..., 0], out[..., 2])
    y2 = np.maximum(out[..., 1], out[..., 3])
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = x1, y1, x2, y2
    return out


def build_model_input_features(
    clip: np.ndarray,
    derivative_mode: Any = "pose",
    phase_head_type: Any = PHASE_HEAD_MLP,
) -> np.ndarray:
    if uses_direct_barbell_box_head(phase_head_type):
        return clip.astype(np.float32, copy=False)
    return build_pose_input_features(clip, derivative_mode)


def slice_padded_clip(seq: np.ndarray, end: int, clip_len: int) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float32)
    T = len(arr)
    if T <= 0:
        raise ValueError("cannot slice an empty sequence")
    end = max(0, min(int(end), T - 1))
    start = end - int(clip_len) + 1
    if start >= 0:
        return arr[start : end + 1].astype(np.float32, copy=False)
    pad = -start
    return np.concatenate([np.tile(arr[0:1], (pad, 1, 1)), arr[: end + 1]], axis=0).astype(np.float32, copy=False)


def _wrist_indices_for_aux(clip: np.ndarray) -> Tuple[int, int]:
    V = int(np.asarray(clip).shape[1])
    if V >= MEDIAPIPE_BARBELL_NUM_KPT or V == NUM_KPT:
        return (15, 16)
    if V == YOLO_NUM_KPT:
        return (9, 10)
    if V == 2:
        return (0, 1)
    raise ValueError(f"wrist aux requires MediaPipe33/YOLO17/wrist2-like clip, got V={V}")


def build_phase_aux_features(
    clip: np.ndarray,
    phase_aux_inputs: Sequence[str],
    *,
    barbell_clip: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Flatten phase-head-only auxiliary signals for one causal window.

    These features are deliberately not fed through the ST-GCN backbone.  They
    are concatenated only to the phase classifier feature vector.
    """

    tokens = normalize_phase_aux_inputs(phase_aux_inputs)
    if not tokens:
        return np.zeros((0,), dtype=np.float32)
    base = np.asarray(clip, dtype=np.float32)
    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError(f"phase aux base clip must be [T,V,>=3], got {tuple(base.shape)}")
    parts: List[np.ndarray] = []
    for token in tokens:
        if token == PHASE_AUX_WRIST:
            wrist_idx = _wrist_indices_for_aux(base)
            parts.append(base[:, list(wrist_idx), :3].reshape(-1))
        elif token == PHASE_AUX_BARBELL:
            if barbell_clip is not None:
                bar = np.asarray(barbell_clip, dtype=np.float32)
            elif base.shape[1] >= MEDIAPIPE_BARBELL_NUM_KPT:
                bar = base[:, BARBELL_NODE_INDEX : BARBELL_NODE_INDEX + 1, :3]
            elif base.shape[1] == 1:
                bar = base[:, :1, :3]
            else:
                raise ValueError("barbell phase aux requires barbell_clip or a clip containing a barbell node")
            if bar.ndim != 3 or bar.shape[1] != 1 or bar.shape[2] < 3:
                raise ValueError(f"barbell phase aux must be [T,1,>=3], got {tuple(bar.shape)}")
            parts.append(bar[:, :1, :3].reshape(-1))
        elif token == PHASE_AUX_ACCELERATION:
            parts.append(temporal_xy_acceleration(base).reshape(-1))
        else:  # defensive; normalizer should catch this
            raise ValueError(f"unsupported phase aux input: {token!r}")
    return np.concatenate(parts).astype(np.float32, copy=False)


def make_phase_target(
    T: int,
    reps: Iterable[Tuple[int, int, int]],
    dtype=np.int64,
    exercise_type: Optional[str] = None,
    phase_label_scheme: str = PHASE_LABEL_SCHEME_AS_LABELED,
) -> np.ndarray:
    """Build frame-level phase targets.

    ``bar_direction`` follows physical bar travel. Squat/benchpress already match
    the legacy labels, while deadlift swaps movement segments because its first
    labeled movement is bar-up.
    """
    phases = np.zeros(int(T), dtype=dtype)
    if T <= 0:
        return phases
    scheme = normalize_phase_label_scheme(phase_label_scheme)
    first_phase = PHASE_DOWN
    second_phase = PHASE_UP
    if scheme == PHASE_LABEL_SCHEME_BAR_DIRECTION:
        exercise = normalize_exercise_type(exercise_type)
        if exercise == "deadlift":
            first_phase = PHASE_UP
            second_phase = PHASE_DOWN
    for s, b, f in reps:
        s = max(0, min(int(s), T - 1))
        b = max(0, min(int(b), T - 1))
        f = max(0, min(int(f), T - 1))
        if b > s:
            phases[s:b] = first_phase
        if f > b:
            phases[b:f] = second_phase
    return phases


MP_FLIP_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]


def aug_horizontal_flip(
    k: np.ndarray,
    pose_backend: str = POSE_BACKEND_MEDIAPIPE,
    flip_pairs: Optional[Iterable[Tuple[int, int]]] = None,
) -> np.ndarray:
    pairs = tuple(flip_pairs) if flip_pairs is not None else pose_backend_spec(pose_backend).flip_pairs
    k = k.copy()
    k[..., 0] = -k[..., 0]
    for a, b in pairs:
        k[:, a], k[:, b] = k[:, b].copy(), k[:, a].copy()
    return k


def aug_jitter(k: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    k = k.copy()
    k[..., :2] += np.random.randn(*k[..., :2].shape).astype(np.float32) * sigma
    return k


def aug_rotation(k: np.ndarray, max_deg: float = 15) -> np.ndarray:
    rad = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    k = k.copy()
    k[..., :2] = k[..., :2] @ R.T
    return k


def aug_scale(k: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    k = k.copy()
    k[..., :2] *= np.random.uniform(lo, hi)
    return k


def apply_spatial_aug(
    clip: np.ndarray,
    p: float = 0.5,
    pose_backend: str = POSE_BACKEND_MEDIAPIPE,
    flip_pairs: Optional[Iterable[Tuple[int, int]]] = None,
) -> np.ndarray:
    if np.random.rand() < p:
        clip = aug_horizontal_flip(clip, pose_backend=pose_backend, flip_pairs=flip_pairs)
    if np.random.rand() < p:
        clip = aug_rotation(clip, 15)
    if np.random.rand() < p:
        clip = aug_scale(clip)
    if np.random.rand() < p:
        clip = aug_jitter(clip, 0.02)
    return clip


class CausalWindowDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        labels: Optional[Mapping[Tuple[str, str], Dict[str, Any]]] = None,
        clip_len: int = 96,
        stride: int = 2,
        train: bool = True,
        aug: bool = True,
        derivative_mode: str = "pose",
        phase_head_type: str = PHASE_HEAD_MLP,
        pose_backend: str = POSE_BACKEND_MEDIAPIPE,
        joint_subset: str = JOINT_SUBSET_ALL,
        phase_label_scheme: str = PHASE_LABEL_SCHEME_AS_LABELED,
        barbell_edge_policy: Any = None,
        phase_aux_inputs: Any = None,
    ):
        self.meta = meta.reset_index(drop=True).copy()
        self.labels = labels if labels is not None else LABELS
        if not self.labels:
            raise RuntimeError("labels are empty; pass labels=... or call prepare_context() first")
        self.L = int(clip_len)
        self.stride = int(stride)
        self.train = bool(train)
        self.aug = bool(aug and train)
        self.derivative_mode = normalize_derivative_mode(derivative_mode)
        self.phase_head_type = normalize_phase_head_type(phase_head_type)
        self.phase_label_scheme = normalize_phase_label_scheme(phase_label_scheme)
        self.phase_aux_inputs = normalize_phase_aux_inputs(phase_aux_inputs)
        self.pose_backend = normalize_pose_backend(pose_backend)
        self.barbell_edge_policy = normalize_barbell_edge_policy(barbell_edge_policy, self.pose_backend)
        if uses_direct_barbell_box_head(self.phase_head_type) and self.pose_backend != POSE_BACKEND_BARBELL:
            raise ValueError("phase_head_type='barbell_box_mlp' requires pose_backend='barbell'")
        self.input_channels = BARBELL_BOX_INPUT_CHANNELS if uses_direct_barbell_box_head(self.phase_head_type) else derivative_mode_input_channels(self.derivative_mode)
        self.joint_subset = normalize_joint_subset(joint_subset)
        self.joint_view = joint_subset_view(self.pose_backend, self.joint_subset, barbell_edge_policy=self.barbell_edge_policy)
        self.source_num_kpt = int(pose_backend_spec(self.pose_backend).num_kpt)
        self.num_kpt = int(self.joint_view.num_kpt)
        self.samples: List[Dict[str, Any]] = []

        for row_idx, r in self.meta.iterrows():
            T = _row_int(r, "T_used", _row_int(r, "T", 0))
            reps = self.labels[(r["type"], r["name"])]["reps"]
            phases = make_phase_target(
                T,
                reps,
                exercise_type=r["type"],
                phase_label_scheme=self.phase_label_scheme,
            )
            if T <= 0:
                continue
            if T < self.L:
                end_indices = [T - 1]
            else:
                end_indices = list(range(self.L - 1, T, self.stride))
                if end_indices[-1] != T - 1:
                    end_indices.append(T - 1)
            for end in end_indices:
                self.samples.append(
                    {
                        "row_idx": row_idx,
                        "end": int(end),
                        "phase": int(phases[end]),
                        "cls": int(r["cls"]),
                        "type": r["type"],
                        "name": r["name"],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        r = self.meta.iloc[s["row_idx"]]
        load_result = load_model_keypoints_from_row(
            r,
            {
                "pose_backend": self.pose_backend,
                "joint_subset": self.joint_subset,
                "barbell_edge_policy": self.barbell_edge_policy,
                "phase_head_type": self.phase_head_type,
                "phase_label_scheme": self.phase_label_scheme,
                "derivative_mode": self.derivative_mode,
            },
        )
        k = load_result.kpts
        T = min(_row_int(r, "T_used", _row_int(r, "T", len(k))), len(k))
        k = k[:T]
        end = min(s["end"], T - 1)
        barbell_clip = None
        if (
            PHASE_AUX_BARBELL in self.phase_aux_inputs
            and self.pose_backend == POSE_BACKEND_MEDIAPIPE_BARBELL
            and self.joint_subset == JOINT_SUBSET_POSE_ONLY
        ):
            full_result = load_model_keypoints_from_row(
                r,
                {
                    "pose_backend": self.pose_backend,
                    "joint_subset": JOINT_SUBSET_ALL,
                    "barbell_edge_policy": self.barbell_edge_policy,
                    "phase_head_type": self.phase_head_type,
                    "phase_label_scheme": self.phase_label_scheme,
                    "derivative_mode": self.derivative_mode,
                },
            )
            full_k = full_result.kpts
            full_T = min(T, len(full_k))
            full_clip = slice_padded_clip(full_k[:full_T], min(end, full_T - 1), self.L)
            if self.aug and not uses_direct_barbell_box_head(self.phase_head_type):
                full_view = joint_subset_view(self.pose_backend, JOINT_SUBSET_ALL, barbell_edge_policy=self.barbell_edge_policy)
                full_clip = apply_spatial_aug(
                    full_clip,
                    p=0.5,
                    pose_backend=self.pose_backend,
                    flip_pairs=full_view.flip_pairs,
                )
            clip = select_joint_subset(
                full_clip,
                pose_backend=self.pose_backend,
                joint_subset=self.joint_subset,
                barbell_edge_policy=self.barbell_edge_policy,
            )
            barbell_clip = full_clip[:, BARBELL_NODE_INDEX : BARBELL_NODE_INDEX + 1, :3]
        else:
            clip = slice_padded_clip(k, end, self.L)
            if self.aug:
                if uses_direct_barbell_box_head(self.phase_head_type):
                    clip = apply_barbell_box_aug(clip, p=0.5)
                else:
                    clip = apply_spatial_aug(clip, p=0.5, pose_backend=self.pose_backend, flip_pairs=self.joint_view.flip_pairs)
        clip = build_model_input_features(clip, self.derivative_mode, self.phase_head_type)
        x = torch.from_numpy(clip).permute(2, 0, 1).unsqueeze(-1).contiguous()
        y_cls = torch.tensor(s["cls"], dtype=torch.long)
        y_phase = torch.tensor(s["phase"], dtype=torch.long)
        if self.phase_aux_inputs:
            aux = build_phase_aux_features(
                clip if uses_direct_barbell_box_head(self.phase_head_type) else np.asarray(clip[..., :3], dtype=np.float32),
                self.phase_aux_inputs,
                barbell_clip=barbell_clip,
            )
            return x, y_cls, y_phase, torch.from_numpy(aux).contiguous()
        return x, y_cls, y_phase


MP_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]

COCO_FLIP_PAIRS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]

COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def mediapipe_barbell_edges(policy: Any = BARBELL_EDGE_POLICY_WRISTS) -> Tuple[Tuple[int, int], ...]:
    policy = normalize_barbell_edge_policy(policy, POSE_BACKEND_MEDIAPIPE_BARBELL)
    extras: Tuple[Tuple[int, int], ...]
    if policy == BARBELL_EDGE_POLICY_WRISTS:
        extras = ((BARBELL_NODE_INDEX, 15), (BARBELL_NODE_INDEX, 16))
    elif policy == BARBELL_EDGE_POLICY_HANDS:
        extras = tuple((BARBELL_NODE_INDEX, idx) for idx in (15, 16, 17, 18, 19, 20, 21, 22))
    elif policy in {BARBELL_EDGE_POLICY_NONE, BARBELL_EDGE_POLICY_NO_POSE_EDGES}:
        extras = tuple()
    else:  # defensive; normalizer should catch this
        raise ValueError(f"unsupported barbell_edge_policy: {policy!r}")
    return tuple(MP_EDGES) + extras


def build_adjacency(num_node: int = 33, edges: Iterable[Tuple[int, int]] = MP_EDGES) -> np.ndarray:
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    A += np.eye(num_node, dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
    return D_inv_sqrt @ A @ D_inv_sqrt


A_NORM = torch.from_numpy(build_adjacency()).float()

POSE_BACKENDS: Dict[str, PoseBackendSpec] = {
    POSE_BACKEND_MEDIAPIPE: PoseBackendSpec(
        name=POSE_BACKEND_MEDIAPIPE,
        num_kpt=NUM_KPT,
        graph_id="mediapipe33",
        edges=tuple(MP_EDGES),
        flip_pairs=tuple(MP_FLIP_PAIRS),
        l_hip=23,
        r_hip=24,
        l_sho=11,
        r_sho=12,
    ),
    POSE_BACKEND_YOLO: PoseBackendSpec(
        name=POSE_BACKEND_YOLO,
        num_kpt=YOLO_NUM_KPT,
        graph_id="coco17",
        edges=tuple(COCO_EDGES),
        flip_pairs=tuple(COCO_FLIP_PAIRS),
        l_hip=11,
        r_hip=12,
        l_sho=5,
        r_sho=6,
    ),
    POSE_BACKEND_BARBELL: PoseBackendSpec(
        name=POSE_BACKEND_BARBELL,
        num_kpt=1,
        graph_id="barbell1",
        edges=tuple(),
        flip_pairs=tuple(),
        l_hip=0,
        r_hip=0,
        l_sho=0,
        r_sho=0,
    ),
    POSE_BACKEND_MEDIAPIPE_BARBELL: PoseBackendSpec(
        name=POSE_BACKEND_MEDIAPIPE_BARBELL,
        num_kpt=MEDIAPIPE_BARBELL_NUM_KPT,
        graph_id="mediapipe33_barbell34",
        edges=mediapipe_barbell_edges(BARBELL_EDGE_POLICY_WRISTS),
        flip_pairs=tuple(MP_FLIP_PAIRS),
        l_hip=23,
        r_hip=24,
        l_sho=11,
        r_sho=12,
    ),
}


def pose_backend_spec(pose_backend: str = POSE_BACKEND_MEDIAPIPE) -> PoseBackendSpec:
    backend = normalize_pose_backend(pose_backend)
    return POSE_BACKENDS[backend]


def joint_subset_view(
    pose_backend: str = POSE_BACKEND_MEDIAPIPE,
    joint_subset: str = JOINT_SUBSET_ALL,
    barbell_edge_policy: Any = None,
) -> JointSubsetView:
    backend = normalize_pose_backend(pose_backend)
    subset = normalize_joint_subset(joint_subset)
    policy = normalize_barbell_edge_policy(barbell_edge_policy, backend)
    spec = pose_backend_spec(backend)
    if backend == POSE_BACKEND_MEDIAPIPE_BARBELL:
        if subset == JOINT_SUBSET_POSE_ONLY:
            return JointSubsetView(
                name=JOINT_SUBSET_POSE_ONLY,
                indices=tuple(range(NUM_KPT)),
                names=("mediapipe33",),
                num_kpt=NUM_KPT,
                graph_id="mediapipe33",
                edges=tuple(MP_EDGES),
                flip_pairs=tuple(MP_FLIP_PAIRS),
            )
        if subset != JOINT_SUBSET_ALL:
            raise ValueError("pose_backend='mediapipe_barbell' supports joint_subset='all' or 'pose_only'.")
        graph_id = f"mediapipe33_barbell34_{policy}"
        return JointSubsetView(
            name=JOINT_SUBSET_ALL,
            indices=None,
            names=("mediapipe33", "barbell_center"),
            num_kpt=MEDIAPIPE_BARBELL_NUM_KPT,
            graph_id=graph_id,
            edges=mediapipe_barbell_edges(policy),
            flip_pairs=tuple(MP_FLIP_PAIRS),
        )
    if subset in {JOINT_SUBSET_ALL, JOINT_SUBSET_POSE_ONLY}:
        return JointSubsetView(
            name=JOINT_SUBSET_ALL if subset == JOINT_SUBSET_ALL else JOINT_SUBSET_POSE_ONLY,
            indices=None,
            names=("all",) if subset == JOINT_SUBSET_ALL else ("pose",),
            num_kpt=int(spec.num_kpt),
            graph_id=str(spec.graph_id),
            edges=tuple(spec.edges),
            flip_pairs=tuple(spec.flip_pairs),
        )
    if backend == POSE_BACKEND_BARBELL:
        raise ValueError("joint_subset='wrist_only' is not valid for pose_backend='barbell'; use joint_subset='all'.")
    if backend == POSE_BACKEND_MEDIAPIPE:
        indices = (15, 16)
        graph_id = "mediapipe_wrist2"
    else:
        indices = (9, 10)
        graph_id = "yolo_wrist2"
    return JointSubsetView(
        name=JOINT_SUBSET_WRIST_ONLY,
        indices=indices,
        names=("left_wrist", "right_wrist"),
        num_kpt=2,
        graph_id=graph_id,
        edges=((0, 1),),
        flip_pairs=((0, 1),),
    )


def select_joint_subset(
    k: np.ndarray,
    pose_backend: str = POSE_BACKEND_MEDIAPIPE,
    joint_subset: str = JOINT_SUBSET_ALL,
    barbell_edge_policy: Any = None,
) -> np.ndarray:
    view = joint_subset_view(pose_backend, joint_subset, barbell_edge_policy=barbell_edge_policy)
    if view.indices is None:
        return k
    return k[:, list(view.indices), :].copy()


def backend_adjacency(pose_backend: str = POSE_BACKEND_MEDIAPIPE) -> torch.Tensor:
    spec = pose_backend_spec(pose_backend)
    return torch.from_numpy(build_adjacency(num_node=spec.num_kpt, edges=spec.edges)).float()


def model_adjacency(
    pose_backend: str = POSE_BACKEND_MEDIAPIPE,
    joint_subset: str = JOINT_SUBSET_ALL,
    barbell_edge_policy: Any = None,
) -> torch.Tensor:
    view = joint_subset_view(pose_backend, joint_subset, barbell_edge_policy=barbell_edge_policy)
    return torch.from_numpy(build_adjacency(num_node=view.num_kpt, edges=view.edges)).float()


class GraphConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nctv,vw->nctw", self.conv(x), self.A)


class STGCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, A: torch.Tensor, kernel_t: int = 9, stride: int = 1, residual: bool = True):
        super().__init__()
        pad = (kernel_t - 1) // 2
        self.temporal_stride = int(stride)
        self.kernel_t = int(kernel_t)
        self.padding_t = int(pad)
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
            self.residual = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=(stride, 1)), nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.tcn(self.gcn(x))
        return self.relu(x + res)


def _conv1d_out_len(length: int, kernel: int = 9, stride: int = 1, padding: int = 4, dilation: int = 1) -> int:
    return math.floor((length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def stgcn_temporal_out_len(clip_len: int) -> int:
    t = int(clip_len)
    if t <= 0:
        raise ValueError(f"clip_len must be positive, got {clip_len}")
    for stride in STGCN_TEMPORAL_STRIDES:
        t = _conv1d_out_len(t, kernel=9, stride=int(stride), padding=4, dilation=1)
    return int(t)


class STGCNBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, A: torch.Tensor = A_NORM, base: int = 64):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(0))
        self.layers = nn.ModuleList(
            [
                STGCNBlock(in_channels, base, A, residual=False),
                STGCNBlock(base, base, A),
                STGCNBlock(base, base, A),
                STGCNBlock(base, base * 2, A, stride=2),
                STGCNBlock(base * 2, base * 2, A),
                STGCNBlock(base * 2, base * 4, A, stride=2),
                STGCNBlock(base * 4, base * 4, A),
            ]
        )
        self.out_channels = base * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for blk in self.layers:
            x = blk(x)
        c, t, v = x.size(1), x.size(2), x.size(3)
        return x.view(N, M, c, t, v).mean(dim=1)


class ExerciseJointAttention(nn.Module):
    """Exercise-conditioned joint attention with visibility bias."""

    def __init__(self, feat_dim: int = 256, num_exercise: int = 3, num_joints: int = 33, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_exercise = int(num_exercise)
        self.num_joints = int(num_joints)
        self.exercise_proj = nn.Linear(self.num_exercise, self.feat_dim)
        self.q_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.k_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.v_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.out_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.scale = self.feat_dim**-0.5
        self.norm = nn.LayerNorm(self.feat_dim)

    def forward(
        self,
        joint_feat: torch.Tensor,
        exercise_onehot: torch.Tensor,
        visibility: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if joint_feat.dim() != 3:
            raise ValueError(f"joint_feat must be [B,V,C], got {tuple(joint_feat.shape)}")
        if visibility.dim() != 2:
            raise ValueError(f"visibility must be [B,V], got {tuple(visibility.shape)}")
        if joint_feat.size(1) != self.num_joints:
            raise ValueError(f"joint_feat V={joint_feat.size(1)} does not match num_joints={self.num_joints}")
        if visibility.size(1) != self.num_joints:
            raise ValueError(f"visibility V={visibility.size(1)} does not match num_joints={self.num_joints}")
        exercise_onehot = exercise_onehot.to(device=joint_feat.device, dtype=joint_feat.dtype)
        visibility = visibility.to(device=joint_feat.device, dtype=joint_feat.dtype)

        ex_emb = F.relu(self.exercise_proj(exercise_onehot))
        q = self.q_proj(ex_emb).unsqueeze(1)
        k = self.k_proj(joint_feat)
        v = self.v_proj(joint_feat)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        vis_bias = torch.log(visibility.clamp(min=1e-6)).unsqueeze(1)
        attn = attn + vis_bias
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.bmm(attn_weights, v).squeeze(1)
        out = self.out_proj(out)
        out = self.norm(out + ex_emb)
        return out, attn_weights.squeeze(1)


def append_phase_aux_features(
    phase_feat: torch.Tensor,
    phase_aux: Optional[torch.Tensor],
    expected_dim: int,
) -> torch.Tensor:
    expected = int(expected_dim)
    if expected <= 0:
        return phase_feat
    if phase_aux is None:
        raise ValueError(f"phase_aux is required because phase_aux_dim={expected}")
    aux = phase_aux.to(device=phase_feat.device, dtype=phase_feat.dtype)
    aux = aux.reshape(aux.size(0), -1)
    if aux.size(1) != expected:
        raise RuntimeError(f"phase_aux dim mismatch: expected {expected}, got {aux.size(1)}")
    return torch.cat([phase_feat, aux], dim=1)


class PerExercisePhaseHead(nn.Module):
    """Separate phase classifier MLP for each exercise.

    During oracle/GT-conditioned experiments the correct exercise id selects
    one head per sample.  During predicted-action conditioning, the separate
    head logits are mixed with detached action probabilities so the phase path
    can still run without ground-truth labels.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: int,
        num_phase: int = 3,
        num_exercise: int = 3,
        dropout: float = 0.3,
        bottleneck: bool = False,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.num_phase = int(num_phase)
        self.num_exercise = int(num_exercise)
        self.bottleneck = bool(bottleneck)

        def make_head() -> nn.Sequential:
            layers: List[nn.Module] = [
                nn.Linear(self.feature_dim, self.hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
            ]
            if self.bottleneck:
                mid = max(self.hidden // 2, self.num_phase)
                layers.extend(
                    [
                        nn.Linear(self.hidden, mid),
                        nn.ReLU(inplace=True),
                        nn.Dropout(float(dropout)),
                        nn.Linear(mid, self.num_phase),
                    ]
                )
            else:
                layers.append(nn.Linear(self.hidden, self.num_phase))
            return nn.Sequential(*layers)

        self.heads = nn.ModuleList(make_head() for _ in range(self.num_exercise))

    def forward(
        self,
        features: torch.Tensor,
        action_logit: Optional[torch.Tensor] = None,
        exercise_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"features must be [B,C], got {tuple(features.shape)}")
        all_logits = torch.stack([head(features) for head in self.heads], dim=1)  # [B,E,P]
        if exercise_id is not None:
            idx = exercise_id.to(device=features.device, dtype=torch.long).clamp(0, self.num_exercise - 1)
            batch = torch.arange(features.size(0), device=features.device)
            return all_logits[batch, idx]
        if action_logit is None:
            return all_logits.mean(dim=1)
        if action_logit.size(1) != self.num_exercise:
            raise ValueError(
                f"action_logit classes={action_logit.size(1)} does not match per-exercise heads={self.num_exercise}"
            )
        weights = F.softmax(action_logit, dim=1).detach().to(dtype=all_logits.dtype)
        return torch.sum(all_logits * weights.unsqueeze(-1), dim=1)


class MultiTaskSTGCNLSTM(nn.Module):
    def __init__(
        self,
        num_action: int = 3,
        num_phase: int = 3,
        in_c: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        phase_head_type: str = PHASE_HEAD_MLP,
        derivative_mode: str = "pose",
        A: Optional[torch.Tensor] = None,
        pose_backend: str = POSE_BACKEND_MEDIAPIPE,
        joint_subset: str = JOINT_SUBSET_ALL,
        barbell_edge_policy: Any = None,
        num_kpt: Optional[int] = None,
        pose_graph_id: Optional[str] = None,
        source_num_kpt: Optional[int] = None,
        source_graph_id: Optional[str] = None,
        selected_joint_indices: Optional[Iterable[int]] = None,
        selected_joint_names: Optional[Iterable[str]] = None,
        phase_aux_inputs: Any = None,
        phase_aux_dim: int = 0,
    ):
        super().__init__()
        self.phase_head_type = normalize_phase_head_type(phase_head_type)
        if uses_direct_barbell_box_head(self.phase_head_type):
            raise ValueError("model_type='lstm' does not support phase_head_type='barbell_box_mlp'")
        self.input_channels = int(in_c)
        self.derivative_mode = normalize_derivative_mode(derivative_mode)
        self.num_action = int(num_action)
        self.num_phase = int(num_phase)
        self.pose_backend = normalize_pose_backend(pose_backend)
        self.joint_subset = normalize_joint_subset(joint_subset)
        self.barbell_edge_policy = normalize_barbell_edge_policy(barbell_edge_policy, self.pose_backend)
        self.phase_aux_inputs = tuple(normalize_phase_aux_inputs(phase_aux_inputs))
        self.phase_aux_dim = int(phase_aux_dim)
        spec = pose_backend_spec(self.pose_backend)
        view = joint_subset_view(self.pose_backend, self.joint_subset, barbell_edge_policy=self.barbell_edge_policy)
        graph = A if A is not None else model_adjacency(self.pose_backend, self.joint_subset, self.barbell_edge_policy)
        self.source_num_kpt = int(source_num_kpt if source_num_kpt is not None else spec.num_kpt)
        self.source_graph_id = str(source_graph_id or spec.graph_id)
        self.model_num_kpt = int(num_kpt if num_kpt is not None else view.num_kpt)
        self.num_kpt = self.model_num_kpt
        self.pose_graph_id = str(pose_graph_id or view.graph_id)
        self.model_graph_id = self.pose_graph_id
        self.selected_joint_indices = tuple(int(i) for i in (selected_joint_indices if selected_joint_indices is not None else (view.indices or ())))
        self.selected_joint_names = tuple(str(n) for n in (selected_joint_names if selected_joint_names is not None else view.names))
        self.backbone = STGCNBackbone(in_channels=in_c, A=graph)
        c = self.backbone.out_channels
        self.action_head = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c // 2, num_action),
        )
        self.lstm = nn.LSTM(
            input_size=c,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        if self.phase_head_type == PHASE_HEAD_EXERCISE_ATTN:
            self.joint_attn = ExerciseJointAttention(
                feat_dim=c,
                num_exercise=self.num_action,
                num_joints=self.num_kpt,
                dropout=float(dropout),
            )
            phase_feature_dim = int(lstm_hidden) + c
        else:
            self.joint_attn = None
            phase_feature_dim = int(lstm_hidden)
        phase_feature_dim += self.phase_aux_dim
        if self.phase_head_type == PHASE_HEAD_PER_EXERCISE_MLP:
            self.phase_head = PerExercisePhaseHead(
                phase_feature_dim,
                lstm_hidden,
                num_phase=num_phase,
                num_exercise=self.num_action,
                dropout=float(dropout),
                bottleneck=False,
            )
        else:
            self.phase_head = nn.Sequential(
                nn.Linear(phase_feature_dim, lstm_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(lstm_hidden, num_phase),
            )
        self.pooling_feature_dim = int(phase_feature_dim)
        self.phase_pooling = "lstm_last"

    def _exercise_vector(self, action_logit: torch.Tensor, exercise_id: Optional[torch.Tensor]) -> torch.Tensor:
        if exercise_id is None:
            return F.softmax(action_logit, dim=-1).detach()
        exercise_id = exercise_id.to(device=action_logit.device, dtype=torch.long)
        return F.one_hot(exercise_id, num_classes=self.num_action).to(dtype=action_logit.dtype)

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: Optional[torch.Tensor] = None,
        phase_aux: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        action_logit = self.action_head(f.mean(dim=(2, 3)))
        ft = f.mean(dim=-1).permute(0, 2, 1)
        lstm_out, _ = self.lstm(ft)
        phase_feat = lstm_out[:, -1, :]
        if self.phase_head_type == PHASE_HEAD_EXERCISE_ATTN:
            if x.size(1) < 3:
                raise ValueError(f"expected visibility/confidence at input channel 2, got input shape={tuple(x.shape)}")
            if self.joint_attn is None:
                raise RuntimeError("joint attention module is not initialized")
            exercise_onehot = self._exercise_vector(action_logit, exercise_id)
            visibility = x[:, 2, :, :, 0].mean(dim=1)
            joint_feat = f.mean(dim=2).permute(0, 2, 1).contiguous()
            attn_feat, attn_weights = self.joint_attn(joint_feat, exercise_onehot, visibility)
            phase_feat = torch.cat([phase_feat, attn_feat], dim=1)
            phase_feat = append_phase_aux_features(phase_feat, phase_aux, self.phase_aux_dim)
            phase_logit = self.phase_head(phase_feat)
            if return_attn:
                return action_logit, phase_logit, attn_weights
            return action_logit, phase_logit
        phase_feat = append_phase_aux_features(phase_feat, phase_aux, self.phase_aux_dim)
        if self.phase_head_type == PHASE_HEAD_PER_EXERCISE_MLP:
            phase_logit = self.phase_head(phase_feat, action_logit=action_logit, exercise_id=exercise_id)
            if return_attn:
                return action_logit, phase_logit, None
            return action_logit, phase_logit
        phase_logit = self.phase_head(phase_feat)
        if return_attn:
            return action_logit, phase_logit, None
        return action_logit, phase_logit


class MultiTaskSTGCNMLP(nn.Module):
    def __init__(
        self,
        num_action: int = 3,
        num_phase: int = 3,
        in_c: int = 3,
        mlp_hidden: int = 128,
        dropout: float = 0.3,
        phase_pooling: str = "temporal_avg",
        phase_head_type: str = PHASE_HEAD_MLP,
        clip_len: int = 16,
        derivative_mode: str = "pose",
        A: Optional[torch.Tensor] = None,
        pose_backend: str = POSE_BACKEND_MEDIAPIPE,
        joint_subset: str = JOINT_SUBSET_ALL,
        barbell_edge_policy: Any = None,
        num_kpt: Optional[int] = None,
        pose_graph_id: Optional[str] = None,
        source_num_kpt: Optional[int] = None,
        source_graph_id: Optional[str] = None,
        selected_joint_indices: Optional[Iterable[int]] = None,
        selected_joint_names: Optional[Iterable[str]] = None,
        phase_aux_inputs: Any = None,
        phase_aux_dim: int = 0,
    ):
        super().__init__()
        if phase_pooling not in ALLOWED_PHASE_POOLING:
            raise ValueError(f"unsupported phase_pooling={phase_pooling}; allowed={ALLOWED_PHASE_POOLING}")
        self.phase_head_type = normalize_phase_head_type(phase_head_type)
        self.input_channels = int(in_c)
        self.derivative_mode = normalize_derivative_mode(derivative_mode)
        self.num_action = int(num_action)
        self.num_phase = int(num_phase)
        self.pose_backend = normalize_pose_backend(pose_backend)
        self.joint_subset = normalize_joint_subset(joint_subset)
        self.barbell_edge_policy = normalize_barbell_edge_policy(barbell_edge_policy, self.pose_backend)
        self.phase_aux_inputs = tuple(normalize_phase_aux_inputs(phase_aux_inputs))
        self.phase_aux_dim = int(phase_aux_dim)
        spec = pose_backend_spec(self.pose_backend)
        view = joint_subset_view(self.pose_backend, self.joint_subset, barbell_edge_policy=self.barbell_edge_policy)
        graph = A if A is not None else model_adjacency(self.pose_backend, self.joint_subset, self.barbell_edge_policy)
        self.source_num_kpt = int(source_num_kpt if source_num_kpt is not None else spec.num_kpt)
        self.source_graph_id = str(source_graph_id or spec.graph_id)
        self.model_num_kpt = int(num_kpt if num_kpt is not None else view.num_kpt)
        self.num_kpt = self.model_num_kpt
        self.pose_graph_id = str(pose_graph_id or view.graph_id)
        self.model_graph_id = self.pose_graph_id
        self.selected_joint_indices = tuple(int(i) for i in (selected_joint_indices if selected_joint_indices is not None else (view.indices or ())))
        self.selected_joint_names = tuple(str(n) for n in (selected_joint_names if selected_joint_names is not None else view.names))
        self.backbone = STGCNBackbone(in_channels=in_c, A=graph)
        c = self.backbone.out_channels
        self.phase_pooling = phase_pooling
        self.clip_len = int(clip_len)
        self.expected_tprime = stgcn_temporal_out_len(self.clip_len)
        if self.phase_head_type == PHASE_HEAD_BARBELL_BOX_MLP:
            if self.pose_backend != POSE_BACKEND_BARBELL:
                raise ValueError("phase_head_type='barbell_box_mlp' requires pose_backend='barbell'")
            phase_feature_dim = self.clip_len * self.num_kpt * self.input_channels
        elif self.phase_head_type == PHASE_HEAD_EXERCISE_ATTN:
            phase_feature_dim = c
        elif phase_pooling in {"temporal_avg", "last"}:
            phase_feature_dim = c
        elif phase_pooling == "temporal_flatten":
            phase_feature_dim = c * self.expected_tprime
        elif phase_pooling == "avg_last_concat":
            phase_feature_dim = c * 2
        else:  # defensive; constructor validation should catch this
            raise ValueError(f"unsupported phase_pooling={phase_pooling}")
        phase_feature_dim += self.phase_aux_dim
        self.pooling_feature_dim = int(phase_feature_dim)

        self.action_head = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c // 2, num_action),
        )
        if self.phase_head_type == PHASE_HEAD_EXERCISE_ATTN:
            self.joint_attn = ExerciseJointAttention(
                feat_dim=c,
                num_exercise=self.num_action,
                num_joints=self.num_kpt,
                dropout=float(dropout),
            )
        else:
            self.joint_attn = None
        if self.phase_head_type == PHASE_HEAD_PER_EXERCISE_MLP:
            self.phase_head = PerExercisePhaseHead(
                self.pooling_feature_dim,
                mlp_hidden,
                num_phase=num_phase,
                num_exercise=self.num_action,
                dropout=float(dropout),
                bottleneck=True,
            )
        else:
            self.phase_head = nn.Sequential(
                nn.Linear(self.pooling_feature_dim, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden // 2, num_phase),
            )

    def _exercise_vector(self, action_logit: torch.Tensor, exercise_id: Optional[torch.Tensor]) -> torch.Tensor:
        if exercise_id is None:
            return F.softmax(action_logit, dim=-1).detach()
        exercise_id = exercise_id.to(device=action_logit.device, dtype=torch.long)
        return F.one_hot(exercise_id, num_classes=self.num_action).to(dtype=action_logit.dtype)

    def _phase_features(self, f: torch.Tensor) -> torch.Tensor:
        ft = f.mean(dim=-1).permute(0, 2, 1).contiguous()  # [B, T', 256]
        actual_t = int(ft.size(1))
        if actual_t != self.expected_tprime:
            raise RuntimeError(
                f"ST-GCN temporal length mismatch for phase_pooling={self.phase_pooling}: "
                f"clip_len={self.clip_len}, expected T'={self.expected_tprime}, actual T'={actual_t}."
            )
        if self.phase_pooling == "temporal_avg":
            return ft.mean(dim=1)
        if self.phase_pooling == "last":
            return ft[:, -1, :]
        if self.phase_pooling == "temporal_flatten":
            return ft.reshape(ft.size(0), -1)
        if self.phase_pooling == "avg_last_concat":
            return torch.cat([ft.mean(dim=1), ft[:, -1, :]], dim=1)
        raise ValueError(f"unsupported phase_pooling={self.phase_pooling}")

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: Optional[torch.Tensor] = None,
        phase_aux: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        f = self.backbone(x)
        action_feat = f.mean(dim=(2, 3))
        action_logit = self.action_head(action_feat)
        if self.phase_head_type == PHASE_HEAD_BARBELL_BOX_MLP:
            # Direct barbell-box phase path: bypass ST-GCN features and feed the
            # temporal bbox sequence itself into the phase MLP.
            phase_feat = x.squeeze(-1).permute(0, 2, 3, 1).contiguous().reshape(x.size(0), -1)
            phase_logit = self.phase_head(phase_feat)
            if return_attn:
                return action_logit, phase_logit, None
            return action_logit, phase_logit
        if self.phase_head_type == PHASE_HEAD_EXERCISE_ATTN:
            if x.size(1) < 3:
                raise ValueError(f"expected visibility/confidence at input channel 2, got input shape={tuple(x.shape)}")
            if self.joint_attn is None:
                raise RuntimeError("joint attention module is not initialized")
            exercise_onehot = self._exercise_vector(action_logit, exercise_id)
            visibility = x[:, 2, :, :, 0].mean(dim=1)
            joint_feat = f.mean(dim=2).permute(0, 2, 1).contiguous()
            phase_feat, attn_weights = self.joint_attn(joint_feat, exercise_onehot, visibility)
            phase_feat = append_phase_aux_features(phase_feat, phase_aux, self.phase_aux_dim)
            phase_logit = self.phase_head(phase_feat)
            if return_attn:
                return action_logit, phase_logit, attn_weights
            return action_logit, phase_logit
        phase_feat = self._phase_features(f)
        phase_feat = append_phase_aux_features(phase_feat, phase_aux, self.phase_aux_dim)
        if self.phase_head_type == PHASE_HEAD_PER_EXERCISE_MLP:
            phase_logit = self.phase_head(phase_feat, action_logit=action_logit, exercise_id=exercise_id)
            if return_attn:
                return action_logit, phase_logit, None
            return action_logit, phase_logit
        phase_logit = self.phase_head(phase_feat)
        if return_attn:
            return action_logit, phase_logit, None
        return action_logit, phase_logit


def build_model(cfg: Mapping[str, Any], device: Optional[str | torch.device] = None) -> nn.Module:
    norm = normalize_cfg(cfg)
    dev = torch.device(device or DEVICE)
    input_channels = int(norm["input_channels"])
    graph = model_adjacency(
        str(norm["pose_backend"]),
        str(norm["joint_subset"]),
        str(norm["barbell_edge_policy"]),
    )
    selected_joint_indices = tuple(int(i) for i in norm.get("selected_joint_indices", []))
    selected_joint_names = tuple(str(n) for n in norm.get("selected_joint_names", []))
    if norm["model_type"] == "lstm":
        model = MultiTaskSTGCNLSTM(
            num_action=NUM_CLASSES,
            num_phase=NUM_PHASES,
            in_c=input_channels,
            lstm_hidden=int(norm["hidden"]),
            lstm_layers=int(norm["lstm_layers"]),
            dropout=float(norm["dropout"]),
            phase_head_type=str(norm["phase_head_type"]),
            derivative_mode=str(norm["derivative_mode"]),
            A=graph,
            pose_backend=str(norm["pose_backend"]),
            joint_subset=str(norm["joint_subset"]),
            barbell_edge_policy=str(norm["barbell_edge_policy"]),
            num_kpt=int(norm["num_kpt"]),
            pose_graph_id=str(norm["pose_graph_id"]),
            source_num_kpt=int(norm["source_num_kpt"]),
            source_graph_id=str(norm["source_graph_id"]),
            selected_joint_indices=selected_joint_indices,
            selected_joint_names=selected_joint_names,
            phase_aux_inputs=norm["phase_aux_inputs"],
            phase_aux_dim=int(norm["phase_aux_dim"]),
        )
    else:
        model = MultiTaskSTGCNMLP(
            num_action=NUM_CLASSES,
            num_phase=NUM_PHASES,
            in_c=input_channels,
            mlp_hidden=int(norm["hidden"]),
            dropout=float(norm["dropout"]),
            phase_pooling=str(norm["phase_pooling"]),
            phase_head_type=str(norm["phase_head_type"]),
            clip_len=int(norm["clip_len"]),
            derivative_mode=str(norm["derivative_mode"]),
            A=graph,
            pose_backend=str(norm["pose_backend"]),
            joint_subset=str(norm["joint_subset"]),
            barbell_edge_policy=str(norm["barbell_edge_policy"]),
            num_kpt=int(norm["num_kpt"]),
            pose_graph_id=str(norm["pose_graph_id"]),
            source_num_kpt=int(norm["source_num_kpt"]),
            source_graph_id=str(norm["source_graph_id"]),
            selected_joint_indices=selected_joint_indices,
            selected_joint_names=selected_joint_names,
            phase_aux_inputs=norm["phase_aux_inputs"],
            phase_aux_dim=int(norm["phase_aux_dim"]),
        )
    setattr(model, "phase_conditioning", str(norm["phase_conditioning"]))
    setattr(model, "exercise_id_source", str(norm["exercise_id_source"]))
    return model.to(dev)


def model_param_metadata(model: nn.Module) -> Dict[str, Any]:
    joint_attn = getattr(model, "joint_attn", None)
    return {
        "num_params_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "phase_head_params": int(sum(p.numel() for p in model.phase_head.parameters() if p.requires_grad)),
        "joint_attn_params": int(sum(p.numel() for p in joint_attn.parameters() if p.requires_grad)) if joint_attn is not None else 0,
        "pooling_feature_dim": int(getattr(model, "pooling_feature_dim", -1)),
        "input_channels": int(getattr(model, "input_channels", -1)),
        "phase_aux_inputs": list(getattr(model, "phase_aux_inputs", ())),
        "phase_aux_dim": int(getattr(model, "phase_aux_dim", 0)),
        "phase_head_type": str(getattr(model, "phase_head_type", "mlp")),
        "phase_conditioning": str(getattr(model, "phase_conditioning", PHASE_CONDITIONING_NONE)),
        "exercise_id_source": str(getattr(model, "exercise_id_source", EXERCISE_ID_SOURCE_NONE)),
        "pose_backend": str(getattr(model, "pose_backend", POSE_BACKEND_MEDIAPIPE)),
        "joint_subset": str(getattr(model, "joint_subset", JOINT_SUBSET_ALL)),
        "barbell_edge_policy": str(getattr(model, "barbell_edge_policy", BARBELL_EDGE_POLICY_NONE)),
        "source_num_kpt": int(getattr(model, "source_num_kpt", NUM_KPT)),
        "source_graph_id": str(getattr(model, "source_graph_id", "mediapipe33")),
        "model_num_kpt": int(getattr(model, "model_num_kpt", getattr(model, "num_kpt", NUM_KPT))),
        "num_kpt": int(getattr(model, "num_kpt", NUM_KPT)),
        "pose_graph_id": str(getattr(model, "pose_graph_id", "mediapipe33")),
        "model_graph_id": str(getattr(model, "model_graph_id", getattr(model, "pose_graph_id", "mediapipe33"))),
        "selected_joint_indices": list(getattr(model, "selected_joint_indices", ())),
        "selected_joint_names": list(getattr(model, "selected_joint_names", ("all",))),
    }


def model_phase_conditioning(model: nn.Module) -> str:
    return normalize_phase_conditioning(
        getattr(model, "phase_conditioning", None),
        getattr(model, "phase_head_type", PHASE_HEAD_MLP),
    )


def model_forward_with_conditioning(
    model: nn.Module,
    x: torch.Tensor,
    exercise_id: Optional[torch.Tensor] = None,
    phase_aux: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head_type = normalize_phase_head_type(getattr(model, "phase_head_type", PHASE_HEAD_MLP))
    phase_aux_inputs = tuple(getattr(model, "phase_aux_inputs", ()))
    if not uses_exercise_conditioned_phase_head(head_type):
        return model(x, phase_aux=phase_aux)
    conditioning = model_phase_conditioning(model)
    if conditioning == PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL:
        if exercise_id is None:
            raise ValueError(
                f"phase_head_type={head_type!r} with phase_conditioning='ground_truth_exercise_label' "
                "requires exercise_id."
            )
        return model(x, exercise_id=exercise_id, phase_aux=phase_aux)
    if conditioning == PHASE_CONDITIONING_PREDICTED_ACTION:
        return model(x, phase_aux=phase_aux)
    raise ValueError(f"unsupported phase_conditioning for exercise_attn: {conditioning!r}")


def model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(DEVICE)


def multitask_loss(
    action_logit: torch.Tensor,
    phase_logit: torch.Tensor,
    y_cls: torch.Tensor,
    y_phase: torch.Tensor,
    alpha: float = 1.0,
    class_weight: Optional[torch.Tensor] = None,
    phase_weight: Optional[torch.Tensor] = None,
):
    ce_a = F.cross_entropy(action_logit, y_cls, weight=class_weight)
    ce_p = F.cross_entropy(phase_logit, y_phase, weight=phase_weight)
    return ce_a + alpha * ce_p, ce_a, ce_p


def smooth_phase(phase_seq: Iterable[int], window: int = 5) -> np.ndarray:
    phase_seq = np.asarray(phase_seq, dtype=np.int64)
    if window <= 1 or len(phase_seq) == 0:
        return phase_seq
    h = window // 2
    smoothed = np.zeros_like(phase_seq)
    for i in range(len(phase_seq)):
        w = phase_seq[max(0, i - h) : min(len(phase_seq), i + h + 1)]
        vals, counts = np.unique(w, return_counts=True)
        smoothed[i] = vals[counts.argmax()]
    return smoothed


def count_phases(phase_seq: Iterable[int], min_up_len: int = 3) -> Tuple[int, List[int]]:
    count, transitions = 0, []
    in_up, up_start = False, -1
    for t, p in enumerate(phase_seq):
        if p == PHASE_UP:
            if not in_up:
                in_up, up_start = True, t
        else:
            if in_up:
                if t - up_start >= min_up_len:
                    count += 1
                    transitions.append(t)
                in_up = False
    return count, transitions


@torch.no_grad()
def predict_from_kpts(
    model: nn.Module,
    kpts: np.ndarray,
    clip_len: int = 32,
    stride: int = 1,
    smooth_window: int = 5,
    min_up_len: int = 3,
    batch_size: int = 64,
    derivative_mode: Optional[str] = None,
    pose_backend: Optional[str] = None,
    joint_subset: Optional[str] = None,
    phase_head_type: Optional[str] = None,
    barbell_box_features: Optional[np.ndarray] = None,
    barbell_edge_policy: Any = None,
    phase_aux_source_kpts: Optional[np.ndarray] = None,
    model_ready_features: bool = False,
    exercise_id: Optional[int] = None,
    phase_conditioning: Optional[str] = None,
    allow_predicted_action_conditioning: bool = False,
) -> Dict[str, Any]:
    model.eval()
    device = model_device(model)
    mode = normalize_derivative_mode(
        derivative_mode if derivative_mode is not None else getattr(model, "derivative_mode", "pose")
    )
    backend = normalize_pose_backend(
        pose_backend if pose_backend is not None else getattr(model, "pose_backend", POSE_BACKEND_MEDIAPIPE)
    )
    subset = normalize_joint_subset(
        joint_subset if joint_subset is not None else getattr(model, "joint_subset", JOINT_SUBSET_ALL)
    )
    head_type = normalize_phase_head_type(
        phase_head_type if phase_head_type is not None else getattr(model, "phase_head_type", PHASE_HEAD_MLP)
    )
    conditioning = normalize_phase_conditioning(
        phase_conditioning if phase_conditioning is not None else getattr(model, "phase_conditioning", None),
        head_type,
    )
    if uses_exercise_conditioned_phase_head(head_type) and conditioning == PHASE_CONDITIONING_PREDICTED_ACTION and not allow_predicted_action_conditioning:
        raise ValueError(
            "phase_conditioning='predicted_action' is an explicit fallback mode and requires "
            "allow_predicted_action_conditioning=True."
        )
    if uses_exercise_conditioned_phase_head(head_type) and conditioning == PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL and exercise_id is None:
        raise ValueError(
            f"phase_head_type={head_type!r} with phase_conditioning='ground_truth_exercise_label' "
            "requires exercise_id. Pass exercise_id=<ground-truth class id> for the oracle label-conditioned "
            "experiment, or set phase_conditioning='predicted_action' with allow_predicted_action_conditioning=True."
        )
    edge_policy = normalize_barbell_edge_policy(
        barbell_edge_policy if barbell_edge_policy is not None else getattr(model, "barbell_edge_policy", None),
        backend,
    )
    if uses_direct_barbell_box_head(head_type):
        if backend != POSE_BACKEND_BARBELL:
            raise ValueError("phase_head_type='barbell_box_mlp' requires pose_backend='barbell'")
        if barbell_box_features is None and model_ready_features:
            k = np.asarray(kpts, dtype=np.float32)
        elif barbell_box_features is None:
            k = barbell_box_features_from_arrays(kpts=kpts.astype(np.float32))
        else:
            k = np.asarray(barbell_box_features, dtype=np.float32)
        if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] != BARBELL_BOX_INPUT_CHANNELS:
            raise ValueError(
                "barbell_box_features must have shape "
                f"[T,1,{BARBELL_BOX_INPUT_CHANNELS}], got {tuple(k.shape)}"
            )
    else:
        if model_ready_features:
            k = np.asarray(kpts, dtype=np.float32)
            expected_v = joint_subset_view(backend, subset, barbell_edge_policy=edge_policy).num_kpt
            if k.ndim != 3 or k.shape[1] != expected_v or k.shape[2] < 3:
                raise ValueError(f"model-ready keypoints must have shape [T,{expected_v},>=3], got {tuple(k.shape)}")
            k = k[..., :3]
        else:
            k = normalize_kpts(kpts.astype(np.float32), pose_backend=backend)
            k = select_joint_subset(k, pose_backend=backend, joint_subset=subset, barbell_edge_policy=edge_policy)
    T = len(k)
    if T == 0:
        raise ValueError("empty pose sequence")
    phase_aux_inputs = tuple(getattr(model, "phase_aux_inputs", ()))
    aux_source = None
    if phase_aux_inputs and phase_aux_source_kpts is not None:
        aux_source = np.asarray(phase_aux_source_kpts, dtype=np.float32)
        if len(aux_source) < T:
            T = len(aux_source)
            k = k[:T]
        else:
            aux_source = aux_source[:T]

    phase_raw = np.zeros(T, dtype=np.int64) + PHASE_READY
    phase_prob = np.zeros((T, NUM_PHASES), dtype=np.float32)
    action_raw = np.zeros(T, dtype=np.int64)
    action_prob = np.zeros((T, NUM_CLASSES), dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)
    action_valid_mask = np.zeros(T, dtype=bool)
    action_probs_all: List[np.ndarray] = []

    ends = [T - 1] if T < clip_len else list(range(clip_len - 1, T, stride))
    if T >= clip_len and ends[-1] != T - 1:
        ends.append(T - 1)
    ends = sorted(set(ends))

    clips: List[np.ndarray] = []
    aux_clips: List[Optional[np.ndarray]] = []
    clip_ends: List[int] = []

    def flush(batch_clips: List[np.ndarray], batch_aux_clips: List[Optional[np.ndarray]], batch_ends: List[int]) -> None:
        xb = torch.stack(
            [
                torch.from_numpy(build_model_input_features(c, mode, head_type))
                .permute(2, 0, 1)
                .unsqueeze(-1)
                .contiguous()
                for c in batch_clips
            ]
        ).to(device)
        phase_aux_tensor = None
        if phase_aux_inputs:
            aux_rows = []
            for c, aux_c in zip(batch_clips, batch_aux_clips):
                barbell_clip = None
                base_c = c
                if aux_c is not None:
                    if aux_c.shape[1] >= MEDIAPIPE_BARBELL_NUM_KPT:
                        barbell_clip = aux_c[:, BARBELL_NODE_INDEX : BARBELL_NODE_INDEX + 1, :3]
                    base_c = c
                aux_rows.append(build_phase_aux_features(base_c, phase_aux_inputs, barbell_clip=barbell_clip))
            phase_aux_tensor = torch.stack([torch.from_numpy(row).contiguous() for row in aux_rows]).to(device)
        exercise_tensor = None
        if conditioning == PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL and exercise_id is not None:
            exercise_tensor = torch.full((len(batch_clips),), int(exercise_id), dtype=torch.long, device=device)
        with torch.no_grad():
            a_logit, p_logit = model_forward_with_conditioning(
                model,
                xb,
                exercise_id=exercise_tensor,
                phase_aux=phase_aux_tensor,
            )
            a_prob = F.softmax(a_logit, dim=1).cpu().numpy()
            p_prob = F.softmax(p_logit, dim=1).cpu().numpy()
        for e, ap, pp in zip(batch_ends, a_prob, p_prob):
            action_probs_all.append(ap)
            action_raw[e] = int(ap.argmax())
            action_prob[e] = ap
            action_valid_mask[e] = True
            phase_raw[e] = int(pp.argmax())
            phase_prob[e] = pp
            valid_mask[e] = True

    for end in ends:
        start = end - clip_len + 1
        clip = k[start : end + 1] if start >= 0 else np.concatenate([np.tile(k[0:1], (-start, 1, 1)), k[: end + 1]], axis=0)
        aux_clip = None
        if aux_source is not None:
            aux_clip = (
                aux_source[start : end + 1]
                if start >= 0
                else np.concatenate([np.tile(aux_source[0:1], (-start, 1, 1)), aux_source[: end + 1]], axis=0)
            )
        clips.append(clip)
        aux_clips.append(aux_clip)
        clip_ends.append(end)
        if len(clips) == batch_size:
            flush(clips, aux_clips, clip_ends)
            clips, aux_clips, clip_ends = [], [], []
    if clips:
        flush(clips, aux_clips, clip_ends)

    last_action = 0
    last_action_prob = np.eye(NUM_CLASSES, dtype=np.float32)[0]
    last_phase = PHASE_READY
    last_prob = np.eye(NUM_PHASES, dtype=np.float32)[PHASE_READY]
    for t in range(T):
        if action_valid_mask[t]:
            last_action = int(action_raw[t])
            last_action_prob = action_prob[t]
        else:
            action_raw[t] = last_action
            action_prob[t] = last_action_prob
        if valid_mask[t]:
            last_phase = phase_raw[t]
            last_prob = phase_prob[t]
        else:
            phase_raw[t] = last_phase
            phase_prob[t] = last_prob

    phase_smooth = smooth_phase(phase_raw, window=smooth_window)
    pred_count, transitions = count_phases(phase_smooth, min_up_len=min_up_len)
    action_vote = np.mean(np.stack(action_probs_all), axis=0) if action_probs_all else np.eye(NUM_CLASSES, dtype=np.float32)[0]
    pred_cls = int(action_vote.argmax())
    return {
        "T": T,
        "pred_cls_id": pred_cls,
        "pred_class": ID_TO_CLASS[pred_cls],
        "action_probs": action_vote,
        "action_raw": action_raw,
        "action_prob_frame": action_prob,
        "action_valid_mask": action_valid_mask,
        "phase_conditioning": conditioning,
        "exercise_id_source": exercise_id_source_for_conditioning(conditioning),
        "phase_raw": phase_raw,
        "phase_smooth": phase_smooth,
        "phase_prob": phase_prob,
        "valid_mask": valid_mask,
        "pred_count": int(pred_count),
        "transitions": transitions,
    }


def compute_weights_from_ds(ds: CausalWindowDataset, device: Optional[str | torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_c = np.zeros(NUM_CLASSES, dtype=np.float64)
    phase_c = np.zeros(NUM_PHASES, dtype=np.float64)
    for s in ds.samples:
        cls_c[int(s["cls"])] += 1
        phase_c[int(s["phase"])] += 1
    cls_w = cls_c.sum() / (NUM_CLASSES * np.maximum(cls_c, 1))
    phase_w = 1.0 / (phase_c / max(phase_c.sum(), 1) + 1e-6)
    phase_w = phase_w / phase_w.mean()
    dev = torch.device(device or DEVICE)
    return torch.tensor(cls_w, dtype=torch.float32, device=dev), torch.tensor(phase_w, dtype=torch.float32, device=dev)


def unpack_window_batch(batch: Sequence[torch.Tensor], device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if len(batch) == 3:
        x, y_cls, y_phase = batch
        phase_aux = None
    elif len(batch) == 4:
        x, y_cls, y_phase, phase_aux = batch
    else:
        raise ValueError(f"unexpected window batch length: {len(batch)}")
    dev = torch.device(device)
    x = x.to(dev)
    y_cls = y_cls.to(dev)
    y_phase = y_phase.to(dev)
    if phase_aux is not None:
        phase_aux = phase_aux.to(dev)
    return x, y_cls, y_phase, phase_aux


@torch.no_grad()
def eval_windows(model: nn.Module, dl: DataLoader, cw: torch.Tensor, pw: torch.Tensor, alpha: float) -> Dict[str, float]:
    model.eval()
    dev = model_device(model)
    a_preds: List[int] = []
    a_gts: List[int] = []
    p_preds: List[int] = []
    p_gts: List[int] = []
    losses: List[float] = []
    for batch in dl:
        x, y_cls, y_phase, phase_aux = unpack_window_batch(batch, dev)
        a_logit, p_logit = model_forward_with_conditioning(model, x, exercise_id=y_cls, phase_aux=phase_aux)
        loss, _, _ = multitask_loss(a_logit, p_logit, y_cls, y_phase, alpha=alpha, class_weight=cw, phase_weight=pw)
        losses.append(float(loss.item()))
        a_preds.extend(a_logit.argmax(1).cpu().tolist())
        a_gts.extend(y_cls.cpu().tolist())
        p_preds.extend(p_logit.argmax(1).cpu().tolist())
        p_gts.extend(y_phase.cpu().tolist())
    if not losses:
        empty = {
            "val_loss": float("nan"),
            "action_acc": float("nan"),
            "action_f1": float("nan"),
            "phase_acc": float("nan"),
            "phase_f1": float("nan"),
        }
        empty.update({f"phase_{name}_f1": float("nan") for name in PHASE_NAMES})
        return empty
    _, _, phase_f1s, _ = precision_recall_fscore_support(
        np.asarray(p_gts, dtype=np.int64),
        np.asarray(p_preds, dtype=np.int64),
        labels=list(range(NUM_PHASES)),
        zero_division=0,
    )
    return {
        "val_loss": float(np.mean(losses)),
        "action_acc": float(accuracy_score(a_gts, a_preds)),
        "action_f1": float(f1_score(a_gts, a_preds, average="macro", zero_division=0)),
        "phase_acc": float(accuracy_score(p_gts, p_preds)),
        "phase_f1": float(f1_score(p_gts, p_preds, average="macro", zero_division=0)),
        **{f"phase_{name}_f1": float(phase_f1s[i]) for i, name in enumerate(PHASE_NAMES)},
    }


def eval_videos(
    model: nn.Module,
    val_meta_df: pd.DataFrame,
    clip_len: int,
    stride: int,
    labels: Optional[Mapping[Tuple[str, str], Dict[str, Any]]] = None,
    smooth_window: int = 5,
    min_up_len: int = 3,
    pose_backend: Optional[str] = None,
    joint_subset: Optional[str] = None,
    phase_label_scheme: str = PHASE_LABEL_SCHEME_AS_LABELED,
    barbell_edge_policy: Any = None,
    phase_conditioning: Optional[str] = None,
) -> Dict[str, float]:
    labels = labels if labels is not None else LABELS
    if labels is None or len(labels) == 0:
        raise RuntimeError("labels are empty; pass labels=... or call prepare_context() first")
    rows: List[Dict[str, Any]] = []
    all_gt: List[int] = []
    all_pred: List[int] = []
    all_action_gt: List[int] = []
    all_action_pred: List[int] = []
    phase_by_exercise: Dict[str, Dict[str, List[int]]] = {
        exercise: {"gt": [], "pred": []} for exercise in CLASS_LIST
    }
    action_by_exercise: Dict[str, Dict[str, List[int]]] = {
        exercise: {"gt": [], "pred": []} for exercise in CLASS_LIST
    }
    backend = normalize_pose_backend(
        pose_backend if pose_backend is not None else getattr(model, "pose_backend", POSE_BACKEND_MEDIAPIPE)
    )
    subset = normalize_joint_subset(
        joint_subset if joint_subset is not None else getattr(model, "joint_subset", JOINT_SUBSET_ALL)
    )
    head_type = normalize_phase_head_type(getattr(model, "phase_head_type", PHASE_HEAD_MLP))
    conditioning = normalize_phase_conditioning(
        phase_conditioning if phase_conditioning is not None else getattr(model, "phase_conditioning", None),
        head_type,
    )
    exercise_id_source = exercise_id_source_for_conditioning(conditioning)
    phase_aux_inputs = tuple(getattr(model, "phase_aux_inputs", ()))
    edge_policy = normalize_barbell_edge_policy(
        barbell_edge_policy if barbell_edge_policy is not None else getattr(model, "barbell_edge_policy", None),
        backend,
    )
    target_scheme = normalize_phase_label_scheme(phase_label_scheme)
    load_cfg = {
        "pose_backend": backend,
        "joint_subset": subset,
        "barbell_edge_policy": edge_policy,
        "phase_head_type": head_type,
        "phase_conditioning": conditioning,
        "phase_label_scheme": target_scheme,
        "derivative_mode": getattr(model, "derivative_mode", "pose"),
    }
    for _, r in val_meta_df.iterrows():
        load_result = load_model_keypoints_from_row(r, load_cfg)
        kpts = load_result.kpts
        T = min(_row_int(r, "T_used", _row_int(r, "T", len(kpts))), len(kpts))
        phase_aux_source = None
        if PHASE_AUX_BARBELL in phase_aux_inputs and backend == POSE_BACKEND_MEDIAPIPE_BARBELL:
            full_load_cfg = {**load_cfg, "joint_subset": JOINT_SUBSET_ALL}
            full_result = load_model_keypoints_from_row(r, full_load_cfg)
            phase_aux_source = full_result.kpts
            T = min(T, len(phase_aux_source))
        out = predict_from_kpts(
            model,
            kpts[:T],
            clip_len=clip_len,
            stride=stride,
            smooth_window=smooth_window,
            min_up_len=min_up_len,
            pose_backend=backend,
            joint_subset=subset,
            phase_head_type=head_type,
            barbell_box_features=kpts[:T] if uses_direct_barbell_box_head(head_type) else None,
            barbell_edge_policy=edge_policy,
            phase_aux_source_kpts=phase_aux_source[:T] if phase_aux_source is not None else None,
            model_ready_features=True,
            exercise_id=int(r["cls"]) if conditioning == PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL else None,
            phase_conditioning=conditioning,
            allow_predicted_action_conditioning=conditioning == PHASE_CONDITIONING_PREDICTED_ACTION,
        )
        gt_key = (r["type"], r["name"])
        gt_count = len(labels[gt_key]["reps"])
        gt_phase = make_phase_target(
            T,
            labels[gt_key]["reps"],
            exercise_type=r["type"],
            phase_label_scheme=target_scheme,
        )
        valid = np.arange(T) >= (clip_len - 1 if T >= clip_len else T - 1)
        valid_gt = gt_phase[valid].astype(np.int64).tolist()
        valid_pred = out["phase_smooth"][valid].astype(np.int64).tolist()
        valid_action_gt = np.full(int(valid.sum()), int(r["cls"]), dtype=np.int64).tolist()
        valid_action_pred = out["action_raw"][valid].astype(np.int64).tolist()
        all_action_gt.extend(valid_action_gt)
        all_action_pred.extend(valid_action_pred)
        all_gt.extend(valid_gt)
        all_pred.extend(valid_pred)
        action_bucket = action_by_exercise.setdefault(r["type"], {"gt": [], "pred": []})
        action_bucket["gt"].extend(valid_action_gt)
        action_bucket["pred"].extend(valid_action_pred)
        exercise_bucket = phase_by_exercise.setdefault(r["type"], {"gt": [], "pred": []})
        exercise_bucket["gt"].extend(valid_gt)
        exercise_bucket["pred"].extend(valid_pred)
        rows.append(
            {
                "type": r["type"],
                "gt_cls": r["type"],
                "pred_cls": out["pred_class"],
                "gt_count": gt_count,
                "pred_count": out["pred_count"],
                "abs_error": abs(out["pred_count"] - gt_count),
                "phase_conditioning": conditioning,
                "exercise_id_source": exercise_id_source,
            }
        )
    if not rows:
        empty = {
            "frame_eval_stride": int(stride),
            "frame_action_acc": float("nan"),
            "frame_action_f1": float("nan"),
            "frame_phase_acc": float("nan"),
            "frame_phase_f1": float("nan"),
            "video_action_acc": float("nan"),
            "video_action_f1": float("nan"),
            "video_vote_action_acc": float("nan"),
            "video_vote_action_f1": float("nan"),
            "video_phase_acc": float("nan"),
            "video_phase_f1": float("nan"),
            "phase_acc": float("nan"),
            "phase_macro_f1": float("nan"),
            "phase_conditioning": conditioning,
            "exercise_id_source": exercise_id_source,
            "video_count_mae": float("nan"),
            "video_count_obo": float("nan"),
        }
        empty.update({f"{name}_f1": float("nan") for name in PHASE_NAMES})
        for typ in CLASS_LIST:
            empty[f"frame_action_acc_{typ}"] = float("nan")
            empty[f"phase_acc_{typ}"] = float("nan")
            empty[f"phase_macro_f1_{typ}"] = float("nan")
        return empty
    vr = pd.DataFrame(rows)
    per_class: Dict[str, float] = {}
    per_exercise_phase: Dict[str, float] = {}
    for typ in CLASS_LIST:
        sub = vr[vr["type"] == typ]
        per_class[f"mae_{typ}"] = float(sub["abs_error"].mean()) if len(sub) else float("nan")
        per_class[f"obo_{typ}"] = float((sub["abs_error"] <= 1).mean()) if len(sub) else float("nan")
        action_bucket = action_by_exercise.get(typ, {"gt": [], "pred": []})
        if action_bucket["gt"]:
            per_class[f"frame_action_acc_{typ}"] = float(accuracy_score(action_bucket["gt"], action_bucket["pred"]))
        else:
            per_class[f"frame_action_acc_{typ}"] = float("nan")
        bucket = phase_by_exercise.get(typ, {"gt": [], "pred": []})
        if bucket["gt"]:
            per_exercise_phase[f"phase_acc_{typ}"] = float(accuracy_score(bucket["gt"], bucket["pred"]))
            per_exercise_phase[f"phase_macro_f1_{typ}"] = float(f1_score(bucket["gt"], bucket["pred"], average="macro", zero_division=0))
        else:
            per_exercise_phase[f"phase_acc_{typ}"] = float("nan")
            per_exercise_phase[f"phase_macro_f1_{typ}"] = float("nan")
    if all_gt:
        _, _, global_phase_f1s, _ = precision_recall_fscore_support(
            np.asarray(all_gt, dtype=np.int64),
            np.asarray(all_pred, dtype=np.int64),
            labels=list(range(NUM_PHASES)),
            zero_division=0,
        )
        per_phase = {f"{name}_f1": float(global_phase_f1s[i]) for i, name in enumerate(PHASE_NAMES)}
        global_phase_acc = float(accuracy_score(all_gt, all_pred))
        global_phase_f1 = float(f1_score(all_gt, all_pred, average="macro", zero_division=0))
    else:
        per_phase = {f"{name}_f1": float("nan") for name in PHASE_NAMES}
        global_phase_acc = float("nan")
        global_phase_f1 = float("nan")
    if all_action_gt:
        frame_action_acc = float(accuracy_score(all_action_gt, all_action_pred))
        frame_action_f1 = float(f1_score(all_action_gt, all_action_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))
    else:
        frame_action_acc = float("nan")
        frame_action_f1 = float("nan")
    video_vote_action_acc = float(accuracy_score(vr["gt_cls"], vr["pred_cls"]))
    video_vote_action_f1 = float(f1_score(vr["gt_cls"], vr["pred_cls"], average="macro", zero_division=0))
    return {
        "frame_eval_stride": int(stride),
        "frame_action_acc": frame_action_acc,
        "frame_action_f1": frame_action_f1,
        "frame_phase_acc": global_phase_acc,
        "frame_phase_f1": global_phase_f1,
        # Backward-compatible top-level action keys now report frame-level
        # validation, not video-vote validation.  The old vote metric remains
        # available under video_vote_action_*.
        "video_action_acc": frame_action_acc,
        "video_action_f1": frame_action_f1,
        "video_vote_action_acc": video_vote_action_acc,
        "video_vote_action_f1": video_vote_action_f1,
        "video_phase_acc": global_phase_acc,
        "video_phase_f1": global_phase_f1,
        "phase_acc": global_phase_acc,
        "phase_macro_f1": global_phase_f1,
        "phase_conditioning": conditioning,
        "exercise_id_source": exercise_id_source,
        "video_count_mae": float(mean_absolute_error(vr["gt_count"], vr["pred_count"])),
        "video_count_obo": float((vr["abs_error"] <= 1).mean()),
        **per_phase,
        **per_exercise_phase,
        **per_class,
    }


def make_exp_name(cfg: Mapping[str, Any]) -> str:
    norm = normalize_cfg(cfg)
    pw = _float_token(norm["phase_loss_alpha"])
    base = (
        f"{norm['pose_backend']}_{norm['pose_graph_id']}_j{norm['joint_subset']}_{norm['model_type']}_h{norm['hidden']}_c{norm['clip_len']}_pw_{pw}"
        f"_ts{norm['train_stride']}_do{_dropout_token(norm['dropout'])}_aug{int(norm['aug'])}"
    )
    if norm["model_type"] == "lstm":
        base += f"_l{norm['lstm_layers']}"
    elif norm.get("phase_pooling", "temporal_avg") != "temporal_avg":
        base += f"_pool_{norm['phase_pooling']}"
    if norm.get("phase_head_type", "mlp") != "mlp":
        base += f"_head_{norm['phase_head_type']}"
    if norm.get("phase_conditioning", PHASE_CONDITIONING_NONE) != PHASE_CONDITIONING_NONE:
        base += f"_cond_{_safe_name_token(norm['phase_conditioning'])}"
    if norm.get("derivative_mode", "pose") != "pose":
        base += f"_input_{norm['derivative_mode']}"
    if norm.get("phase_aux_inputs"):
        base += f"_phaseaux_{_safe_name_token('_'.join(norm['phase_aux_inputs']))}"
    if norm.get("phase_label_scheme", PHASE_LABEL_SCHEME_AS_LABELED) != PHASE_LABEL_SCHEME_AS_LABELED:
        base += f"_label_{_safe_name_token(norm['phase_label_scheme'])}"
    if norm.get("pose_backend") == POSE_BACKEND_MEDIAPIPE_BARBELL:
        base += f"_edge_{_safe_name_token(norm['barbell_edge_policy'])}"
    return base


def make_run_exp_name(cfg: Mapping[str, Any]) -> str:
    norm = normalize_cfg(cfg)
    base = str(norm.get("exp_name") or make_exp_name(norm))
    tag = norm.get("fresh_run_tag")
    if tag:
        return f"{base}_{_safe_name_token(tag)}"
    return base


COMPAT_KEYS = [
    "model_type",
    "phase_head_type",
    "phase_conditioning",
    "exercise_id_source",
    "phase_pooling",
    "derivative_mode",
    "phase_aux_inputs",
    "phase_aux_dim",
    "phase_label_scheme",
    "input_channels",
    "pose_backend",
    "joint_subset",
    "barbell_edge_policy",
    "source_num_kpt",
    "source_graph_id",
    "model_num_kpt",
    "num_kpt",
    "pose_graph_id",
    "model_graph_id",
    "selected_joint_indices",
    "selected_joint_names",
    "hidden",
    "lstm_layers",
    "clip_len",
    "train_stride",
    "dropout",
    "aug",
    "phase_loss_alpha",
    "batch",
    "epochs",
]


def checkpoint_cfg_compatible(saved_cfg: Mapping[str, Any], current_cfg: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    saved = normalize_cfg(saved_cfg)
    current = normalize_cfg(current_cfg)
    mismatches = [key for key in COMPAT_KEYS if saved.get(key) != current.get(key)]
    return not mismatches, mismatches


def assert_single_pose_backend(configs: Iterable[Mapping[str, Any]]) -> str:
    backends = {normalize_cfg(cfg)["pose_backend"] for cfg in configs}
    if len(backends) > 1:
        raise ValueError(
            "Mixed pose backends are not supported: one pose backend per invocation is required. "
            f"Got {sorted(backends)}."
        )
    return next(iter(backends), POSE_BACKEND_MEDIAPIPE)


def _checkpoint_payload(
    model: nn.Module,
    cfg: Mapping[str, Any],
    epoch: int,
    best_score: float,
    best_epoch: int,
    history: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    metadata = model_param_metadata(model)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "cfg": normalize_cfg(cfg),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "clip_len": int(normalize_cfg(cfg)["clip_len"]),
        "classes": CLASS_LIST,
        "phase_names": PHASE_NAMES,
        **metadata,
    }
    if history is not None:
        payload["history"] = history
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def run_one_experiment(cfg: Mapping[str, Any], context: Optional[ExperimentContext] = None) -> Dict[str, Any]:
    norm = normalize_cfg(cfg)
    if context is None:
        context = prepare_context(norm, verbose=True, update_globals=True)
    exp_name = make_run_exp_name(norm)
    exp_dir = context.ablation_root / exp_name

    has_existing_checkpoint = (exp_dir / "latest.pt").exists() or any(exp_dir.glob("best_ep*.pt"))
    if (
        bool(norm["force_retrain"])
        and not bool(norm["resume"])
        and has_existing_checkpoint
        and not norm.get("fresh_run_tag")
        and not bool(norm["overwrite_existing"])
    ):
        norm["fresh_run_tag"] = f"rerun_{time.strftime('%Y%m%dT%H%M%S')}"
        exp_name = make_run_exp_name(norm)
        exp_dir = context.ablation_root / exp_name

    exp_dir.mkdir(parents=True, exist_ok=True)
    latest_path = exp_dir / "latest.pt"
    hist_path = exp_dir / "history.json"
    config_path = exp_dir / "config.json"
    config_path.write_text(json.dumps(norm, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"[EXP] {exp_name}")
    print(
        f"  pose={norm['pose_backend']}({norm['pose_graph_id']}, source_V={norm['source_num_kpt']} model_V={norm['model_num_kpt']}) "
        f"joint_subset={norm['joint_subset']} edge_policy={norm['barbell_edge_policy']} "
        f"model={norm['model_type']} phase_head={norm['phase_head_type']} pooling={norm.get('phase_pooling')} "
        f"input={norm['derivative_mode']} phase_aux={norm['phase_aux_inputs']} "
        f"label_scheme={norm['phase_label_scheme']} "
        f"channels={norm['input_channels']} hidden={norm['hidden']} "
        f"clip={norm['clip_len']} stride={norm['train_stride']} drop={norm['dropout']} aug={norm['aug']} "
        f"epochs={norm['epochs']} batch={norm['batch']}",
        flush=True,
    )

    target_epochs = int(norm["epochs"])
    existing_bests = sorted(exp_dir.glob("best_ep*.pt"))
    latest_epoch = 0
    if latest_path.exists() and bool(norm["resume"]):
        latest_probe = torch.load(latest_path, map_location="cpu")
        ok, mismatches = checkpoint_cfg_compatible(latest_probe.get("cfg", {}), norm)
        if not ok:
            raise RuntimeError(f"checkpoint cfg mismatch for {latest_path}: {mismatches}")
        latest_epoch = int(latest_probe.get("epoch", 0))

    if latest_path.exists() and bool(norm["skip_completed"]) and latest_epoch >= target_epochs:
        best_path = existing_bests[-1] if existing_bests else latest_path
        print(f"  -> completed checkpoint found ({best_path.name}); evaluating only", flush=True)
        ckpt = torch.load(best_path, map_location=DEVICE)
        model = build_model(norm)
        model.load_state_dict(ckpt["model"])
        model.eval()
        metadata = model_param_metadata(model)
        vm = eval_videos(
            model,
            context.val_meta,
            int(norm["clip_len"]),
            1,
            labels=context.labels,
            pose_backend=str(norm["pose_backend"]),
            joint_subset=str(norm["joint_subset"]),
            phase_label_scheme=str(norm["phase_label_scheme"]),
            barbell_edge_policy=str(norm["barbell_edge_policy"]),
        )
        return {
            "exp_name": exp_name,
            **norm,
            **metadata,
            "num_params": metadata["num_params_total"],
            "best_epoch": int(ckpt.get("best_epoch", -1)),
            "best_val_score": float(ckpt.get("best_score", float("nan"))),
            "elapsed_min": 0.0,
            "skipped": True,
            "completion_status": "skipped",
            "checkpoint_path": str(best_path),
            "latest_path": str(latest_path),
            "history_path": str(hist_path),
            "exp_dir": str(exp_dir),
            **vm,
        }

    train_ds = CausalWindowDataset(
        context.train_meta,
        labels=context.labels,
        clip_len=int(norm["clip_len"]),
        stride=int(norm["train_stride"]),
        train=True,
        aug=bool(norm["aug"]),
        derivative_mode=str(norm["derivative_mode"]),
        phase_head_type=str(norm["phase_head_type"]),
        pose_backend=str(norm["pose_backend"]),
        joint_subset=str(norm["joint_subset"]),
        phase_label_scheme=str(norm["phase_label_scheme"]),
        barbell_edge_policy=str(norm["barbell_edge_policy"]),
        phase_aux_inputs=norm["phase_aux_inputs"],
    )
    val_ds = CausalWindowDataset(
        context.val_meta,
        labels=context.labels,
        clip_len=int(norm["clip_len"]),
        stride=int(norm["train_stride"]),
        train=False,
        aug=False,
        derivative_mode=str(norm["derivative_mode"]),
        phase_head_type=str(norm["phase_head_type"]),
        pose_backend=str(norm["pose_backend"]),
        joint_subset=str(norm["joint_subset"]),
        phase_label_scheme=str(norm["phase_label_scheme"]),
        barbell_edge_policy=str(norm["barbell_edge_policy"]),
        phase_aux_inputs=norm["phase_aux_inputs"],
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"empty dataset windows: train={len(train_ds)} val={len(val_ds)}")
    batch = int(norm["batch"])
    pin_memory = bool(norm["pin_memory"]) and DEVICE == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=int(norm["num_workers"]),
        drop_last=len(train_ds) >= batch,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=int(norm["num_workers"]), pin_memory=pin_memory)
    print(f"  train={len(train_ds)} | val={len(val_ds)} windows", flush=True)

    cw, pw = compute_weights_from_ds(train_ds, device=DEVICE)
    model = build_model(norm)
    metadata = model_param_metadata(model)
    print(
        f"  params={metadata['num_params_total']:,} | phase_head={metadata['phase_head_params']:,} "
        f"| joint_attn={metadata['joint_attn_params']:,} | pooling_dim={metadata['pooling_feature_dim']}",
        flush=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=float(norm["lr"]), weight_decay=float(norm["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(target_epochs, 1))
    history: Dict[str, List[float]] = {"train_loss": [], "val_action_acc": [], "val_action_f1": [], "val_phase_acc": [], "val_phase_f1": []}
    best_score = -1e9
    best_epoch = -1
    start_epoch = 1

    if latest_path.exists() and bool(norm["resume"]):
        ck = torch.load(latest_path, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_score = float(ck.get("best_score", -1e9))
        best_epoch = int(ck.get("best_epoch", -1))
        history = ck.get("history", history)
        print(f"  -> resume from epoch {start_epoch}", flush=True)

    t0 = time.time()
    best_path: Path = existing_bests[-1] if existing_bests else exp_dir / "best_ep000.pt"

    for ep in range(start_epoch, target_epochs + 1):
        model.train()
        losses: List[float] = []
        for batch_data in tqdm(train_dl, desc=f"ep{ep:02d}", leave=False):
            x, y_cls, y_phase, phase_aux = unpack_window_batch(batch_data, DEVICE)
            a_logit, p_logit = model_forward_with_conditioning(model, x, exercise_id=y_cls, phase_aux=phase_aux)
            loss, _, _ = multitask_loss(a_logit, p_logit, y_cls, y_phase, alpha=float(norm["phase_loss_alpha"]), class_weight=cw, phase_weight=pw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        vm = eval_windows(model, val_dl, cw, pw, float(norm["phase_loss_alpha"]))
        score = float(vm["action_f1"] + vm["phase_f1"])
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_action_acc"].append(float(vm["action_acc"]))
        history["val_action_f1"].append(float(vm["action_f1"]))
        history["val_phase_acc"].append(float(vm["phase_acc"]))
        history["val_phase_f1"].append(float(vm["phase_f1"]))

        print(
            f"  [ep {ep:02d}/{target_epochs}] loss={train_loss:.3f} | "
            f"a_acc={vm['action_acc']:.3f} a_f1={vm['action_f1']:.3f} "
            f"p_acc={vm['phase_acc']:.3f} p_f1={vm['phase_f1']:.3f} score={score:.3f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_epoch = ep
            for old in exp_dir.glob("best_ep*.pt"):
                old.unlink()
            best_path = exp_dir / f"best_ep{ep:03d}.pt"
            torch.save(_checkpoint_payload(model, norm, ep, best_score, best_epoch), best_path)

        torch.save(_checkpoint_payload(model, norm, ep, best_score, best_epoch, history=history, optimizer=opt, scheduler=sched), latest_path)
        hist_path.write_text(json.dumps(history, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"  -> complete | best_score={best_score:.4f} | {elapsed / 60:.1f}min", flush=True)

    if not best_path.exists():
        best_path = latest_path
    ck = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    vm = eval_videos(
        model,
        context.val_meta,
        int(norm["clip_len"]),
        1,
        labels=context.labels,
        pose_backend=str(norm["pose_backend"]),
        joint_subset=str(norm["joint_subset"]),
        phase_label_scheme=str(norm["phase_label_scheme"]),
        barbell_edge_policy=str(norm["barbell_edge_policy"]),
    )
    print(f"  -> best: {best_path.name}", flush=True)

    return {
        "exp_name": exp_name,
        **norm,
        **metadata,
        "num_params": metadata["num_params_total"],
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "elapsed_min": round(elapsed / 60, 3),
        "skipped": False,
        "completion_status": "complete",
        "checkpoint_path": str(best_path),
        "latest_path": str(latest_path),
        "history_path": str(hist_path),
        "exp_dir": str(exp_dir),
        **vm,
    }


def env_list(name: str, default: List[Any], cast=str) -> List[Any]:
    raw = os.environ.get(name)
    if not raw:
        return default
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return [cast(v) for v in vals] if vals else default


def build_ablation_configs_from_env() -> List[Dict[str, Any]]:
    force_train = _coerce_bool(os.environ.get("PHASE_FORCE_TRAIN", str(int(DIRECT_RUN_FORCE_TRAIN))))
    fresh_rerun = _coerce_bool(os.environ.get("PHASE_FRESH_RERUN", str(int(DIRECT_RUN_FRESH_RERUN))))
    run_tag = os.environ.get("PHASE_RUN_TAG")
    if force_train and fresh_rerun and not run_tag:
        run_tag = f"rerun_{time.strftime('%Y%m%dT%H%M%S')}"
    env_head_type = os.environ.get("PHASE_HEAD_TYPE", DEFAULT_EXPERIMENT_CONFIG["phase_head_type"])
    if "PHASE_USE_ATTN" in os.environ:
        env_head_type = "exercise_attn" if _coerce_bool(os.environ["PHASE_USE_ATTN"]) else "mlp"

    base = copy.deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    base.update(
        {
            "output_root": os.environ.get("PHASE_ABLATION_ROOT", str(ABLATION_ROOT)),
            "pose_backend": os.environ.get("PHASE_POSE_BACKEND", DEFAULT_EXPERIMENT_CONFIG["pose_backend"]),
            "joint_subset": os.environ.get("PHASE_JOINT_SUBSET", DEFAULT_EXPERIMENT_CONFIG["joint_subset"]),
            "barbell_edge_policy": os.environ.get("PHASE_BARBELL_EDGE_POLICY", DEFAULT_EXPERIMENT_CONFIG["barbell_edge_policy"]),
            "yolo_pose_dir": os.environ.get("PHASE_YOLO_POSE_DIR", str(YOLO_POSE_DIR)),
            "barbell_dir": os.environ.get("PHASE_BARBELL_DIR", str(BARBELL_DIR)),
            "yolo_missing_policy": os.environ.get("PHASE_YOLO_MISSING_POLICY", DEFAULT_EXPERIMENT_CONFIG["yolo_missing_policy"]),
            "barbell_missing_policy": os.environ.get(
                "PHASE_BARBELL_MISSING_POLICY",
                DEFAULT_EXPERIMENT_CONFIG["barbell_missing_policy"],
            ),
            "phase_head_type": env_head_type,
            "phase_aux_inputs": os.environ.get(
                "PHASE_AUX_INPUTS",
                ",".join(DEFAULT_EXPERIMENT_CONFIG.get("phase_aux_inputs", [])),
            ),
            "phase_label_scheme": os.environ.get(
                "PHASE_LABEL_SCHEME",
                DEFAULT_EXPERIMENT_CONFIG["phase_label_scheme"],
            ),
            "epochs": int(os.environ.get("PHASE_EPOCHS", str(DEFAULT_EXPERIMENT_CONFIG["epochs"]))),
            "phase_loss_alpha": float(os.environ.get("PHASE_LOSS_ALPHA", str(DEFAULT_EXPERIMENT_CONFIG["phase_loss_alpha"]))),
            "extract_missing_pose": _coerce_bool(os.environ.get("PHASE_EXTRACT_POSE", "0")),
            "force_retrain": force_train,
            "resume": _coerce_bool(os.environ.get("PHASE_RESUME", "0" if force_train else "1")),
            "skip_completed": _coerce_bool(os.environ.get("PHASE_SKIP_COMPLETED", "0" if force_train else "1")),
            "fresh_run_tag": run_tag if force_train and fresh_rerun else None,
            "overwrite_existing": _coerce_bool(os.environ.get("PHASE_OVERWRITE_EXISTING", "0")),
            "max_videos_per_split_type": os.environ.get(
                "PHASE_MAX_VIDEOS_PER_SPLIT_TYPE",
                DEFAULT_EXPERIMENT_CONFIG["max_videos_per_split_type"],
            ),
            "run_kind": "legacy",
        }
    )
    grid = {
        "pose_backend": env_list("PHASE_POSE_BACKENDS", [base["pose_backend"]], str),
        "joint_subset": env_list("PHASE_JOINT_SUBSETS", [base["joint_subset"]], str),
        "barbell_edge_policy": env_list("PHASE_BARBELL_EDGE_POLICIES", [base["barbell_edge_policy"]], str),
        "phase_head_type": env_list("PHASE_HEAD_TYPES", [base["phase_head_type"]], str),
        "model_type": env_list("PHASE_MODEL_TYPES", ["mlp"], str),
        "phase_pooling": env_list("PHASE_POOLINGS", ["temporal_avg"], str),
        "derivative_mode": env_list("PHASE_DERIVATIVE_MODES", ["pose"], str),
        "phase_aux_inputs": env_list("PHASE_AUX_INPUTS_LIST", [base["phase_aux_inputs"]], str),
        "phase_label_scheme": env_list("PHASE_LABEL_SCHEMES", [base["phase_label_scheme"]], str),
        "hidden": [128],
        "lstm_layers": [1],
        "clip_len": env_list("PHASE_CLIP_LENS", [16], int),
        "train_stride": env_list("PHASE_TRAIN_STRIDES", [2], int),
        "dropout": [0.3],
        "aug": env_list("PHASE_AUGS", [True], lambda v: str(v).lower() in {"1", "true", "yes", "y"}),
    }
    keys = list(grid.keys())
    raw_configs = []
    for values in itertools.product(*grid.values()):
        cfg = copy.deepcopy(base)
        cfg.update(dict(zip(keys, values)))
        raw_configs.append(normalize_cfg(cfg))
    seen, unique = set(), []
    for c in raw_configs:
        k = make_exp_name(c)
        if k not in seen:
            seen.add(k)
            unique.append(c)
    return unique


def append_result_logs(results: List[Dict[str, Any]], context: ExperimentContext) -> None:
    context.ablation_log_json.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    pd.DataFrame(results).to_csv(context.ablation_log_csv, index=False, encoding="utf-8-sig")


def main() -> int:
    ablation_configs = build_ablation_configs_from_env()
    if not ablation_configs:
        print("No experiments configured")
        return 0
    assert_single_pose_backend(ablation_configs)
    context = prepare_context(ablation_configs[0], verbose=True, update_globals=True)

    print(f"total experiments: {len(ablation_configs)}")
    for i, c in enumerate(ablation_configs):
        print(f"  [{i + 1:02d}] {make_run_exp_name(c)}")

    all_results: List[Dict[str, Any]] = []
    if context.ablation_log_json.exists():
        all_results = json.loads(context.ablation_log_json.read_text(encoding="utf-8"))
        print(f"loaded existing results: {len(all_results)}")
    done_names = {r.get("exp_name") for r in all_results}

    for i, cfg in enumerate(ablation_configs):
        exp_name = make_run_exp_name(cfg)
        if exp_name in done_names and any((context.ablation_root / exp_name).glob("best_ep*.pt")):
            print(f"[{i + 1}/{len(ablation_configs)}] {exp_name} -> already in result log; checking run state")
        result = run_one_experiment(cfg, context=context)
        all_results = [r for r in all_results if r.get("exp_name") != result["exp_name"]]
        all_results.append(result)
        done_names.add(result["exp_name"])
        append_result_logs(all_results, context)

    results_df = pd.DataFrame(all_results)
    sort_col = "video_count_obo" if "video_count_obo" in results_df.columns else "best_val_score"
    results_df = results_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    show = [
        "exp_name",
        "pose_backend",
        "joint_subset",
        "source_num_kpt",
        "source_graph_id",
        "model_num_kpt",
        "num_kpt",
        "pose_graph_id",
        "model_graph_id",
        "model_type",
        "phase_head_type",
        "phase_conditioning",
        "exercise_id_source",
        "phase_pooling",
        "derivative_mode",
        "phase_aux_inputs",
        "phase_aux_dim",
        "phase_label_scheme",
        "input_channels",
        "hidden",
        "lstm_layers",
        "clip_len",
        "train_stride",
        "dropout",
        "aug",
        "num_params_total",
        "phase_head_params",
        "joint_attn_params",
        "pooling_feature_dim",
        "best_epoch",
        "best_val_score",
        "phase_acc",
        "phase_macro_f1",
        "ready_f1",
        "down_f1",
        "up_f1",
        "frame_eval_stride",
        "frame_action_acc",
        "frame_action_f1",
        "frame_phase_acc",
        "frame_phase_f1",
        "video_action_acc",
        "video_action_f1",
        "video_vote_action_acc",
        "video_vote_action_f1",
        "video_phase_acc",
        "video_phase_f1",
        "video_count_mae",
        "video_count_obo",
        "elapsed_min",
    ]
    print("\n" + "=" * 70)
    print("Ablation results")
    print("=" * 70)
    print(results_df[[c for c in show if c in results_df.columns]].to_string())
    print(f"\nsaved: {context.ablation_log_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
