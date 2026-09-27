"""Exercise-conditioned joint attention phase experiment runner.

This runner is intentionally standalone: it reuses the repository's ST-GCN
backbone, causal-window dataset, derivative input construction, loss, and
metric helpers, but keeps the attention model and y_cls-conditioned train/eval
loop local to avoid changing existing ablation or realtime paths.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:  # `import model.phase_exercise_attn_experiment`
    from . import phase_metrics
    from . import train_ablation as ta
except ImportError:  # `python model/phase_exercise_attn_experiment.py`
    import phase_metrics  # type: ignore
    import train_ablation as ta  # type: ignore


RUN_KIND = os.environ.get("EXERCISE_ATTN_RUN_KIND", "full").strip().lower() or "full"
SMOKE_EPOCHS = 1
FULL_EPOCHS = 60
SMOKE_MAX_VIDEOS_PER_SPLIT_TYPE = 2
OUTPUT_ROOT = Path("phase_experiments") / "exercise_attn"
SMOOTH_WINDOW = 5
EXERCISE_SOURCE = "gt_metadata"

ATTN_VARIANTS = ("pose_attn", "vel_attn")
VARIANT_TO_DERIVATIVE = {
    "pose_attn": "pose",
    "vel_attn": "velocity",
}
VARIANT_ALIASES = {
    "pose": "pose_attn",
    "p": "pose_attn",
    "pose-attn": "pose_attn",
    "pose_attention": "pose_attn",
    "velocity": "vel_attn",
    "vel": "vel_attn",
    "v": "vel_attn",
    "vel-attn": "vel_attn",
    "velocity_attn": "vel_attn",
    "velocity_attention": "vel_attn",
}

FORCE_RETRAIN = os.environ.get("EXERCISE_ATTN_FORCE_RETRAIN", "0").strip().lower() in {"1", "true", "yes", "on"}
FRESH_RERUN = os.environ.get("EXERCISE_ATTN_FRESH_RERUN", "0").strip().lower() in {"1", "true", "yes", "on"}
RUN_TAG = os.environ.get("EXERCISE_ATTN_RUN_TAG") or None

NOTE = (
    "Exercise-conditioned joint attention phase head with log visibility bias. "
    "This runner executes only pose_attn and vel_attn; baseline MLP temporal_avg is external."
)

COMPAT_KEYS = [
    "architecture",
    "variant",
    "phase_head_type",
    "exercise_source",
    "derivative_mode",
    "phase_label_scheme",
    "input_channels",
    "hidden",
    "clip_len",
    "train_stride",
    "dropout",
    "aug",
    "phase_loss_alpha",
    "batch",
    "epochs",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def normalize_variant(value: Any) -> str:
    variant = str(value or "pose_attn").strip().lower().replace(" ", "_")
    variant = VARIANT_ALIASES.get(variant, variant)
    if variant not in ATTN_VARIANTS:
        raise ValueError(f"unsupported exercise-attention variant: {value!r}; allowed={ATTN_VARIANTS}")
    return variant


def env_variants(default: Iterable[str] = ATTN_VARIANTS) -> List[str]:
    raw = os.environ.get("EXERCISE_ATTN_MODE") or os.environ.get("EXERCISE_ATTN_VARIANTS")
    values = [v.strip() for v in raw.split(",") if v.strip()] if raw else list(default)
    return [normalize_variant(v) for v in values]


def checkpoint_cfg_compatible(saved_cfg: Mapping[str, Any], current_cfg: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    saved = dict(saved_cfg)
    current = dict(current_cfg)
    mismatches = [key for key in COMPAT_KEYS if saved.get(key) != current.get(key)]
    return not mismatches, mismatches


def build_cfg(variant: str, run_kind: str = RUN_KIND, run_tag: Optional[str] = None) -> Dict[str, Any]:
    variant = normalize_variant(variant)
    run_kind = str(run_kind).strip().lower()
    if run_kind not in {"smoke", "full"}:
        raise ValueError(f"run_kind must be smoke or full, got {run_kind!r}")
    derivative_mode = ta.normalize_derivative_mode(VARIANT_TO_DERIVATIVE[variant])
    cfg = copy.deepcopy(ta.DEFAULT_EXPERIMENT_CONFIG)
    cfg.update(
        {
            "architecture": "exercise_joint_attention_visibility",
            "variant": variant,
            "model_type": "mlp",  # Keep train_ablation.normalize_cfg compatibility; model is built locally.
            "phase_pooling": "temporal_avg",  # Compatibility field for train_ablation.normalize_cfg().
            "phase_head_type": "exercise_joint_attention",
            "exercise_source": EXERCISE_SOURCE,
            "derivative_mode": derivative_mode,
            "input_channels": ta.derivative_mode_input_channels(derivative_mode),
            "hidden": 128,
            "clip_len": 16,
            "train_stride": 2,
            "dropout": 0.3,
            "aug": True,
            "epochs": SMOKE_EPOCHS if run_kind == "smoke" else FULL_EPOCHS,
            "batch": int(ta.DEFAULT_EXPERIMENT_CONFIG["batch"]),
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "phase_loss_alpha": 2.0,
            "num_workers": 0,
            "run_kind": run_kind,
            "output_root": str(OUTPUT_ROOT / run_kind),
            "write_meta_csv": False,
            "extract_missing_pose": False,
            "resume": not FORCE_RETRAIN,
            "skip_completed": not FORCE_RETRAIN,
            "force_retrain": FORCE_RETRAIN,
            "fresh_run_tag": run_tag if FORCE_RETRAIN and FRESH_RERUN else None,
            "overwrite_existing": False,
            "max_videos_per_split_type": SMOKE_MAX_VIDEOS_PER_SPLIT_TYPE if run_kind == "smoke" else None,
            "pin_memory": True,
        }
    )
    if run_tag and FORCE_RETRAIN and FRESH_RERUN:
        cfg["fresh_run_tag"] = run_tag
    norm = ta.normalize_cfg(cfg)
    norm["architecture"] = cfg["architecture"]
    norm["variant"] = variant
    norm["phase_head_type"] = cfg["phase_head_type"]
    norm["exercise_source"] = EXERCISE_SOURCE
    norm["input_channels"] = int(cfg["input_channels"])
    return norm


class ExerciseJointAttention(nn.Module):
    """Single-query joint attention where exercise identity supplies the query."""

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
        if joint_feat.size(1) != visibility.size(1):
            raise ValueError(
                f"joint/visibility mismatch: joint_feat V={joint_feat.size(1)} visibility V={visibility.size(1)}"
            )
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


class MultiTaskSTGCNExerciseAttn(nn.Module):
    """ST-GCN multitask model with exercise-conditioned joint attention phase head."""

    def __init__(
        self,
        num_action: int = ta.NUM_CLASSES,
        num_phase: int = ta.NUM_PHASES,
        in_c: int = 3,
        mlp_hidden: int = 128,
        dropout: float = 0.3,
        derivative_mode: str = "pose",
    ):
        super().__init__()
        self.input_channels = int(in_c)
        self.derivative_mode = ta.normalize_derivative_mode(derivative_mode)
        self.num_action = int(num_action)
        self.num_phase = int(num_phase)
        self.backbone = ta.STGCNBackbone(in_channels=self.input_channels)
        c = int(self.backbone.out_channels)
        hidden = int(mlp_hidden)
        self.pooling_feature_dim = c
        self.phase_feature_dim = c
        self.phase_pooling = "exercise_joint_attention"
        self.action_head = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(c // 2, self.num_action),
        )
        self.joint_attn = ExerciseJointAttention(
            feat_dim=c,
            num_exercise=self.num_action,
            num_joints=33,
            dropout=float(dropout),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(c, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden // 2, self.num_phase),
        )

    def exercise_vector(self, action_logit: torch.Tensor, exercise_id: Optional[torch.Tensor]) -> torch.Tensor:
        if exercise_id is None:
            return F.softmax(action_logit, dim=-1).detach()
        exercise_id = exercise_id.to(device=action_logit.device, dtype=torch.long)
        return F.one_hot(exercise_id, num_classes=self.num_action).to(dtype=action_logit.dtype)

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        if x.size(1) < 3:
            raise ValueError(f"expected visibility at input channel 2, got input shape={tuple(x.shape)}")
        f = self.backbone(x)
        action_feat = f.mean(dim=(2, 3))
        action_logit = self.action_head(action_feat)

        ex_onehot = self.exercise_vector(action_logit, exercise_id)
        visibility = x[:, 2, :, :, 0].mean(dim=1)
        joint_feat = f.mean(dim=2).permute(0, 2, 1).contiguous()
        phase_feat, attn_weights = self.joint_attn(joint_feat, ex_onehot, visibility)
        phase_logit = self.phase_head(phase_feat)
        if return_attn:
            return action_logit, phase_logit, attn_weights
        return action_logit, phase_logit


def build_model(cfg: Mapping[str, Any], device: Optional[str | torch.device] = None) -> MultiTaskSTGCNExerciseAttn:
    model = MultiTaskSTGCNExerciseAttn(
        num_action=ta.NUM_CLASSES,
        num_phase=ta.NUM_PHASES,
        in_c=int(cfg["input_channels"]),
        mlp_hidden=int(cfg["hidden"]),
        dropout=float(cfg["dropout"]),
        derivative_mode=str(cfg["derivative_mode"]),
    )
    return model.to(torch.device(device or ta.DEVICE))


def model_metadata(model: MultiTaskSTGCNExerciseAttn) -> Dict[str, int]:
    return {
        "num_params_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "phase_head_params": int(sum(p.numel() for p in model.phase_head.parameters() if p.requires_grad)),
        "joint_attn_params": int(sum(p.numel() for p in model.joint_attn.parameters() if p.requires_grad)),
        "pooling_feature_dim": int(model.pooling_feature_dim),
        "phase_feature_dim": int(model.phase_feature_dim),
        "input_channels": int(model.input_channels),
    }


def exercise_id_for_row(row: pd.Series, source: str) -> Optional[int]:
    source = str(source).lower()
    if source in {"gt", "ground_truth", "metadata", "gt_metadata", "selected", "true"}:
        return int(row["cls"])
    if source in {"pred", "predicted", "action_softmax"}:
        return None
    raise ValueError(f"unsupported exercise_source={source!r}")


def compute_loss_and_logits(
    model: MultiTaskSTGCNExerciseAttn,
    x: torch.Tensor,
    y_cls: torch.Tensor,
    y_phase: torch.Tensor,
    class_weight: torch.Tensor,
    phase_weight: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_logit, phase_logit = model(x, exercise_id=y_cls)
    loss, _, _ = ta.multitask_loss(
        action_logit,
        phase_logit,
        y_cls,
        y_phase,
        alpha=alpha,
        class_weight=class_weight,
        phase_weight=phase_weight,
    )
    return loss, action_logit, phase_logit


@torch.no_grad()
def eval_windows(
    model: MultiTaskSTGCNExerciseAttn,
    dl: DataLoader,
    class_weight: torch.Tensor,
    phase_weight: torch.Tensor,
    alpha: float,
) -> Dict[str, float]:
    model.eval()
    dev = next(model.parameters()).device
    a_preds: List[int] = []
    a_gts: List[int] = []
    p_preds: List[int] = []
    p_gts: List[int] = []
    losses: List[float] = []
    for x, y_cls, y_phase in dl:
        x, y_cls, y_phase = x.to(dev), y_cls.to(dev), y_phase.to(dev)
        loss, action_logit, phase_logit = compute_loss_and_logits(
            model,
            x,
            y_cls,
            y_phase,
            class_weight,
            phase_weight,
            alpha,
        )
        losses.append(float(loss.item()))
        a_preds.extend(action_logit.argmax(1).cpu().tolist())
        a_gts.extend(y_cls.cpu().tolist())
        p_preds.extend(phase_logit.argmax(1).cpu().tolist())
        p_gts.extend(y_phase.cpu().tolist())
    if not losses:
        empty = {
            "val_loss": float("nan"),
            "action_acc": float("nan"),
            "action_f1": float("nan"),
            "phase_acc": float("nan"),
            "phase_f1": float("nan"),
        }
        empty.update({f"phase_{name}_f1": float("nan") for name in ta.PHASE_NAMES})
        return empty
    _, _, phase_f1s, _ = precision_recall_fscore_support(
        np.asarray(p_gts, dtype=np.int64),
        np.asarray(p_preds, dtype=np.int64),
        labels=list(range(ta.NUM_PHASES)),
        zero_division=0,
    )
    return {
        "val_loss": float(np.mean(losses)),
        "action_acc": float(accuracy_score(a_gts, a_preds)),
        "action_f1": float(f1_score(a_gts, a_preds, average="macro", zero_division=0)),
        "phase_acc": float(accuracy_score(p_gts, p_preds)),
        "phase_f1": float(f1_score(p_gts, p_preds, average="macro", zero_division=0)),
        **{f"phase_{name}_f1": float(phase_f1s[i]) for i, name in enumerate(ta.PHASE_NAMES)},
    }


def boundary_distance(frame_idx: int, reps: Iterable[Tuple[int, int, int]]) -> float:
    points: List[int] = []
    for start, bottom, finish in reps:
        points.extend([int(start), int(bottom), int(finish)])
    return float(min(abs(int(frame_idx) - point) for point in points)) if points else float("nan")


@torch.no_grad()
def predict_from_kpts(
    model: MultiTaskSTGCNExerciseAttn,
    kpts: np.ndarray,
    cfg: Mapping[str, Any],
    exercise_id: Optional[int],
    smooth_window: int = SMOOTH_WINDOW,
    min_up_len: int = 3,
) -> Dict[str, Any]:
    model.eval()
    dev = next(model.parameters()).device
    k = ta.normalize_kpts(kpts.astype(np.float32))
    total_frames = len(k)
    if total_frames == 0:
        raise ValueError("empty pose sequence")

    clip_len = int(cfg["clip_len"])
    stride = int(cfg["train_stride"])
    batch_size = int(cfg["batch"])
    derivative_mode = str(cfg["derivative_mode"])

    phase_raw = np.zeros(total_frames, dtype=np.int64) + ta.PHASE_READY
    phase_prob = np.zeros((total_frames, ta.NUM_PHASES), dtype=np.float32)
    valid_mask = np.zeros(total_frames, dtype=bool)
    attn_by_frame = np.full((total_frames, 33), np.nan, dtype=np.float32)
    action_probs_all: List[np.ndarray] = []

    ends = [total_frames - 1] if total_frames < clip_len else list(range(clip_len - 1, total_frames, stride))
    if total_frames >= clip_len and ends[-1] != total_frames - 1:
        ends.append(total_frames - 1)
    ends = sorted(set(ends))

    clips: List[np.ndarray] = []
    clip_ends: List[int] = []

    def flush(batch_clips: List[np.ndarray], batch_ends: List[int]) -> None:
        xb = torch.stack(
            [
                torch.from_numpy(ta.build_pose_input_features(clip, derivative_mode))
                .permute(2, 0, 1)
                .unsqueeze(-1)
                .contiguous()
                for clip in batch_clips
            ]
        ).to(dev)
        ex_tensor = None
        if exercise_id is not None:
            ex_tensor = torch.full((len(batch_clips),), int(exercise_id), dtype=torch.long, device=dev)
        action_logit, phase_logit, attn_weights = model(xb, exercise_id=ex_tensor, return_attn=True)
        a_prob = F.softmax(action_logit, dim=1).cpu().numpy()
        p_prob = F.softmax(phase_logit, dim=1).cpu().numpy()
        attn_np = attn_weights.cpu().numpy().astype(np.float32)
        for end, ap, pp, aw in zip(batch_ends, a_prob, p_prob, attn_np):
            action_probs_all.append(ap)
            phase_raw[end] = int(pp.argmax())
            phase_prob[end] = pp
            valid_mask[end] = True
            attn_by_frame[end] = aw

    for end in ends:
        start = end - clip_len + 1
        if start >= 0:
            clip = k[start : end + 1]
        else:
            clip = np.concatenate([np.tile(k[0:1], (-start, 1, 1)), k[: end + 1]], axis=0)
        clips.append(clip)
        clip_ends.append(end)
        if len(clips) == batch_size:
            flush(clips, clip_ends)
            clips, clip_ends = [], []
    if clips:
        flush(clips, clip_ends)

    last_phase = ta.PHASE_READY
    last_prob = np.eye(ta.NUM_PHASES, dtype=np.float32)[ta.PHASE_READY]
    for frame_idx in range(total_frames):
        if valid_mask[frame_idx]:
            last_phase = phase_raw[frame_idx]
            last_prob = phase_prob[frame_idx]
        else:
            phase_raw[frame_idx] = last_phase
            phase_prob[frame_idx] = last_prob

    phase_smooth = ta.smooth_phase(phase_raw, window=smooth_window)
    pred_count, transitions = ta.count_phases(phase_smooth, min_up_len=min_up_len)
    if action_probs_all:
        action_vote = np.mean(np.stack(action_probs_all), axis=0)
    else:
        action_vote = np.eye(ta.NUM_CLASSES, dtype=np.float32)[0]
    pred_cls = int(action_vote.argmax())
    return {
        "T": total_frames,
        "pred_cls_id": pred_cls,
        "pred_class": ta.ID_TO_CLASS[pred_cls],
        "action_probs": action_vote,
        "phase_raw": phase_raw,
        "phase_smooth": phase_smooth,
        "phase_prob": phase_prob,
        "valid_mask": valid_mask,
        "attn_weights": attn_by_frame,
        "pred_count": int(pred_count),
        "transitions": transitions,
    }


def eval_videos(model: MultiTaskSTGCNExerciseAttn, context: ta.ExperimentContext, cfg: Mapping[str, Any]) -> Dict[str, float]:
    rows: List[Dict[str, Any]] = []
    all_gt: List[int] = []
    all_pred: List[int] = []
    phase_by_exercise: Dict[str, Dict[str, List[int]]] = {
        exercise: {"gt": [], "pred": []} for exercise in ta.CLASS_LIST
    }
    for _, row in context.val_meta.iterrows():
        kpts = np.load(row["npz"])["kpts"].astype(np.float32)
        total_frames = min(int(row["T"]), len(kpts))
        out = predict_from_kpts(
            model,
            kpts[:total_frames],
            cfg,
            exercise_id_for_row(row, str(cfg["exercise_source"])),
            smooth_window=SMOOTH_WINDOW,
        )
        reps = context.labels[(row["type"], row["name"])]["reps"]
        gt_count = len(reps)
        gt_phase = ta.make_phase_target(
            total_frames,
            reps,
            exercise_type=row["type"],
            phase_label_scheme=str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)),
        )
        valid = np.arange(total_frames) >= (
            int(cfg["clip_len"]) - 1 if total_frames >= int(cfg["clip_len"]) else total_frames - 1
        )
        valid_gt = gt_phase[valid].astype(np.int64).tolist()
        valid_pred = out["phase_smooth"][valid].astype(np.int64).tolist()
        all_gt.extend(valid_gt)
        all_pred.extend(valid_pred)
        bucket = phase_by_exercise.setdefault(str(row["type"]), {"gt": [], "pred": []})
        bucket["gt"].extend(valid_gt)
        bucket["pred"].extend(valid_pred)
        rows.append(
            {
                "type": row["type"],
                "gt_cls": row["type"],
                "pred_cls": out["pred_class"],
                "gt_count": gt_count,
                "pred_count": out["pred_count"],
                "abs_error": abs(out["pred_count"] - gt_count),
            }
        )
    if not rows:
        empty = {
            "video_action_acc": float("nan"),
            "video_action_f1": float("nan"),
            "video_phase_acc": float("nan"),
            "video_phase_f1": float("nan"),
            "phase_acc": float("nan"),
            "phase_macro_f1": float("nan"),
            "video_count_mae": float("nan"),
            "video_count_obo": float("nan"),
        }
        empty.update({f"{name}_f1": float("nan") for name in ta.PHASE_NAMES})
        for typ in ta.CLASS_LIST:
            empty[f"phase_acc_{typ}"] = float("nan")
            empty[f"phase_macro_f1_{typ}"] = float("nan")
            empty[f"mae_{typ}"] = float("nan")
            empty[f"obo_{typ}"] = float("nan")
        return empty

    video_df = pd.DataFrame(rows)
    per_class: Dict[str, float] = {}
    per_exercise_phase: Dict[str, float] = {}
    for typ in ta.CLASS_LIST:
        sub = video_df[video_df["type"] == typ]
        per_class[f"mae_{typ}"] = float(sub["abs_error"].mean()) if len(sub) else float("nan")
        per_class[f"obo_{typ}"] = float((sub["abs_error"] <= 1).mean()) if len(sub) else float("nan")
        bucket = phase_by_exercise.get(typ, {"gt": [], "pred": []})
        if bucket["gt"]:
            per_exercise_phase[f"phase_acc_{typ}"] = float(accuracy_score(bucket["gt"], bucket["pred"]))
            per_exercise_phase[f"phase_macro_f1_{typ}"] = float(
                f1_score(bucket["gt"], bucket["pred"], average="macro", zero_division=0)
            )
        else:
            per_exercise_phase[f"phase_acc_{typ}"] = float("nan")
            per_exercise_phase[f"phase_macro_f1_{typ}"] = float("nan")
    if all_gt:
        _, _, global_phase_f1s, _ = precision_recall_fscore_support(
            np.asarray(all_gt, dtype=np.int64),
            np.asarray(all_pred, dtype=np.int64),
            labels=list(range(ta.NUM_PHASES)),
            zero_division=0,
        )
        per_phase = {f"{name}_f1": float(global_phase_f1s[i]) for i, name in enumerate(ta.PHASE_NAMES)}
        global_phase_acc = float(accuracy_score(all_gt, all_pred))
        global_phase_f1 = float(f1_score(all_gt, all_pred, average="macro", zero_division=0))
    else:
        per_phase = {f"{name}_f1": float("nan") for name in ta.PHASE_NAMES}
        global_phase_acc = float("nan")
        global_phase_f1 = float("nan")
    return {
        "video_action_acc": float(accuracy_score(video_df["gt_cls"], video_df["pred_cls"])),
        "video_action_f1": float(f1_score(video_df["gt_cls"], video_df["pred_cls"], average="macro", zero_division=0)),
        "video_phase_acc": global_phase_acc,
        "video_phase_f1": global_phase_f1,
        "phase_acc": global_phase_acc,
        "phase_macro_f1": global_phase_f1,
        "video_count_mae": float(mean_absolute_error(video_df["gt_count"], video_df["pred_count"])),
        "video_count_obo": float((video_df["abs_error"] <= 1).mean()),
        **per_phase,
        **per_exercise_phase,
        **per_class,
    }


def prediction_records_and_attention(
    model: MultiTaskSTGCNExerciseAttn,
    context: ta.ExperimentContext,
    cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, Any]] = []
    attn_sum: Dict[str, np.ndarray] = {}
    attn_count: Dict[str, int] = {}
    for _, row in context.val_meta.iterrows():
        kpts = np.load(row["npz"])["kpts"].astype(np.float32)
        total_frames = min(int(row["T"]), len(kpts))
        reps = context.labels[(row["type"], row["name"])]["reps"]
        gt_phase = ta.make_phase_target(
            total_frames,
            reps,
            exercise_type=row["type"],
            phase_label_scheme=str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)),
        )
        out = predict_from_kpts(
            model,
            kpts[:total_frames],
            cfg,
            exercise_id_for_row(row, str(cfg["exercise_source"])),
            smooth_window=SMOOTH_WINDOW,
        )
        valid_indices = np.where(np.asarray(out["valid_mask"], dtype=bool))[0]
        valid_attn = np.asarray(out["attn_weights"], dtype=np.float32)[valid_indices]
        exercise = str(row["type"])
        finite_attn = valid_attn[np.isfinite(valid_attn).all(axis=1)]
        if len(finite_attn):
            attn_sum[exercise] = attn_sum.get(exercise, np.zeros(33, dtype=np.float64)) + finite_attn.sum(axis=0)
            attn_count[exercise] = attn_count.get(exercise, 0) + int(len(finite_attn))
        for frame_idx in valid_indices:
            probs = np.asarray(out["phase_prob"][frame_idx], dtype=float)
            records.append(
                {
                    "type": row["type"],
                    "name": row["name"],
                    "split": row.get("split", "val"),
                    "frame_idx": int(frame_idx),
                    "clip_len": int(cfg["clip_len"]),
                    "stride": int(cfg["train_stride"]),
                    "variant": cfg["variant"],
                    "derivative_mode": cfg["derivative_mode"],
                    "phase_label_scheme": cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED),
                    "exercise_source": cfg["exercise_source"],
                    "gt_phase": int(gt_phase[frame_idx]),
                    "gt_phase_name": ta.PHASE_NAMES[int(gt_phase[frame_idx])],
                    "pred_phase_raw": int(out["phase_raw"][frame_idx]),
                    "pred_phase_raw_name": ta.PHASE_NAMES[int(out["phase_raw"][frame_idx])],
                    "pred_phase_offline_smooth": int(out["phase_smooth"][frame_idx]),
                    "pred_phase_offline_smooth_name": ta.PHASE_NAMES[int(out["phase_smooth"][frame_idx])],
                    "phase_prob_ready": float(probs[ta.PHASE_READY]),
                    "phase_prob_down": float(probs[ta.PHASE_DOWN]),
                    "phase_prob_up": float(probs[ta.PHASE_UP]),
                    "boundary_distance": boundary_distance(int(frame_idx), reps),
                    "action_pred": out["pred_class"],
                    "action_prob_squat": float(out["action_probs"][ta.CLASS_TO_ID["squat"]]),
                    "action_prob_benchpress": float(out["action_probs"][ta.CLASS_TO_ID["benchpress"]]),
                    "action_prob_deadlift": float(out["action_probs"][ta.CLASS_TO_ID["deadlift"]]),
                }
            )
    if not records:
        raise RuntimeError("no diagnostic prediction records generated")

    attn_rows: List[Dict[str, Any]] = []
    for exercise, sums in attn_sum.items():
        means = sums / max(attn_count.get(exercise, 0), 1)
        order = np.argsort(-means)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        for joint_index, mean_attention in enumerate(means):
            rank = int(ranks[joint_index])
            attn_rows.append(
                {
                    "variant": cfg["variant"],
                    "exercise": exercise,
                    "joint_index": int(joint_index),
                    "mean_attention": float(mean_attention),
                    "rank": rank,
                    "is_top5": bool(rank <= 5),
                    "num_windows": int(attn_count.get(exercise, 0)),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(attn_rows)


def write_diagnostics(
    model: MultiTaskSTGCNExerciseAttn,
    context: ta.ExperimentContext,
    cfg: Mapping[str, Any],
    exp_dir: Path,
) -> Dict[str, Path]:
    preds, attention = prediction_records_and_attention(model, context, cfg)
    paths = phase_metrics.write_metric_artifacts(preds, ta.PHASE_NAMES, exp_dir)
    attention_path = exp_dir / "attn_weights_by_exercise.csv"
    attention.to_csv(attention_path, index=False, encoding="utf-8-sig")
    paths["attention_summary"] = attention_path
    return paths


def diagnostic_metric_row(paths: Mapping[str, Path]) -> Dict[str, Any]:
    metrics_path = Path(paths["metrics"])
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return {}
    out: Dict[str, Any] = {}
    raw = metrics[metrics["mode"] == "raw"]
    if not raw.empty:
        out.update(raw.iloc[0].drop(labels=["mode"]).to_dict())
    smooth = metrics[metrics["mode"] == "offline_smooth"]
    if not smooth.empty:
        out.update({f"offline_smooth_{key}": value for key, value in smooth.iloc[0].drop(labels=["mode"]).to_dict().items()})
    return out


def checkpoint_payload(
    model: MultiTaskSTGCNExerciseAttn,
    cfg: Mapping[str, Any],
    epoch: int,
    best_score: float,
    best_epoch: int,
    history: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "cfg": dict(cfg),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "clip_len": int(cfg["clip_len"]),
        "classes": ta.CLASS_LIST,
        "phase_names": ta.PHASE_NAMES,
        **model_metadata(model),
    }
    if history is not None:
        payload["history"] = history
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def started_manifest(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": f"{cfg['run_kind']}_{cfg['variant']}_{utc_stamp()}",
        "created_at_utc": utc_stamp(),
        "completion_status": "started",
        "architecture": cfg["architecture"],
        "variant": cfg["variant"],
        "exercise_source": cfg["exercise_source"],
        "derivative_mode": cfg["derivative_mode"],
        "phase_label_scheme": cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED),
        "input_channels": int(cfg["input_channels"]),
        "config": dict(cfg),
        "note": NOTE,
    }


def completed_manifest(
    cfg: Mapping[str, Any],
    result: Mapping[str, Any],
    context: ta.ExperimentContext,
    diagnostic_paths: Mapping[str, Path],
) -> Dict[str, Any]:
    return {
        **started_manifest(cfg),
        "completion_status": result.get("completion_status", "complete"),
        "completed_at_utc": utc_stamp(),
        "checkpoint_path": result.get("checkpoint_path"),
        "latest_path": result.get("latest_path"),
        "history_path": result.get("history_path"),
        "exp_dir": result.get("exp_dir"),
        "data_split_fingerprint": ta.data_split_fingerprint(context.meta_df),
        "artifacts": {key: str(value) for key, value in diagnostic_paths.items()},
        "metrics": {key: value for key, value in result.items() if key.endswith("_f1") or key.endswith("_acc")},
        "note": NOTE,
    }


def failed_manifest(cfg: Mapping[str, Any], exc: BaseException) -> Dict[str, Any]:
    manifest = started_manifest(cfg)
    manifest.update({"completion_status": "failed", "failed_at_utc": utc_stamp(), "error": f"{type(exc).__name__}: {exc}"})
    return manifest


def aggregate_row_from_result(
    cfg: Mapping[str, Any],
    result: Optional[Mapping[str, Any]] = None,
    diagnostic_paths: Optional[Mapping[str, Path]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "variant": cfg["variant"],
        "exercise_source": cfg["exercise_source"],
        "derivative_mode": cfg["derivative_mode"],
        "input_channels": int(cfg["input_channels"]),
        "run_kind": cfg["run_kind"],
        "architecture": cfg["architecture"],
        "completion_status": "failed" if error else None,
        "error": error,
        "checkpoint_path": None,
        "attention_summary_path": None,
    }
    if result:
        row.update({key: value for key, value in result.items() if key != "diagnostic_paths"})
    if diagnostic_paths:
        row.update(diagnostic_metric_row(diagnostic_paths))
        row["attention_summary_path"] = str(diagnostic_paths.get("attention_summary", ""))
    if row.get("completion_status") is None:
        row["completion_status"] = result.get("completion_status", "complete") if result else "failed"
    return row


def update_aggregates(
    output_root: Path,
    rows: List[Dict[str, Any]],
    per_exercise_frames: List[pd.DataFrame],
    attention_frames: List[pd.DataFrame],
    print_summary: bool = True,
) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "exercise_attn_results.csv"
    results_json = output_root / "exercise_attn_results.json"
    per_ex_csv = output_root / "exercise_attn_per_exercise.csv"
    attn_dir = output_root / "attn_weights"
    attn_dir.mkdir(parents=True, exist_ok=True)
    attn_csv = attn_dir / "attn_weights_by_exercise.csv"

    result_df = pd.DataFrame(rows)
    if not result_df.empty and "phase_macro_f1" in result_df.columns:
        result_df = result_df.sort_values("phase_macro_f1", ascending=False, na_position="last")
    result_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    results_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    per_ex = pd.concat(per_exercise_frames, ignore_index=True) if per_exercise_frames else pd.DataFrame()
    per_ex.to_csv(per_ex_csv, index=False, encoding="utf-8-sig")
    attn = pd.concat(attention_frames, ignore_index=True) if attention_frames else pd.DataFrame()
    attn.to_csv(attn_csv, index=False, encoding="utf-8-sig")

    if print_summary and not result_df.empty:
        show = [
            "variant",
            "phase_label_scheme",
            "completion_status",
            "best_epoch",
            "phase_macro_f1",
            "video_phase_f1",
            "attention_summary_path",
        ]
        print("\n[exercise_attn_results summary]")
        print(result_df[[col for col in show if col in result_df.columns]].to_string(index=False))
    return {
        "results_csv": results_csv,
        "results_json": results_json,
        "per_exercise_csv": per_ex_csv,
        "attention_summary_csv": attn_csv,
    }


def run_one_experiment(cfg: Dict[str, Any], context: ta.ExperimentContext) -> Dict[str, Any]:
    exp_name = str(cfg["variant"])
    exp_dir = Path(cfg["output_root"]) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    latest_path = exp_dir / "latest.pt"
    hist_path = exp_dir / "history.json"
    manifest_path = exp_dir / "run_manifest.json"
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    write_manifest(manifest_path, started_manifest(cfg))

    print("\n" + "=" * 60)
    print(f"[EXP] {exp_name}")
    print(
        f"  architecture={cfg['architecture']} variant={cfg['variant']} input={cfg['derivative_mode']} "
        f"label_scheme={cfg.get('phase_label_scheme', ta.PHASE_LABEL_SCHEME_AS_LABELED)} "
        f"channels={cfg['input_channels']} exercise_source={cfg['exercise_source']} "
        f"epochs={cfg['epochs']} batch={cfg['batch']}",
        flush=True,
    )

    target_epochs = int(cfg["epochs"])
    existing_bests = sorted(exp_dir.glob("best_ep*.pt"))
    latest_epoch = 0
    if latest_path.exists() and bool(cfg["resume"]):
        latest_probe = torch.load(latest_path, map_location="cpu")
        ok, mismatches = checkpoint_cfg_compatible(latest_probe.get("cfg", {}), cfg)
        if not ok:
            raise RuntimeError(f"checkpoint cfg mismatch for {latest_path}: {mismatches}")
        latest_epoch = int(latest_probe.get("epoch", 0))

    if latest_path.exists() and bool(cfg["skip_completed"]) and latest_epoch >= target_epochs:
        best_path = existing_bests[-1] if existing_bests else latest_path
        print(f"  -> completed checkpoint found ({best_path.name}); evaluating only", flush=True)
        ckpt = torch.load(best_path, map_location=ta.DEVICE)
        model = build_model(cfg, device=ta.DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        metadata = model_metadata(model)
        video_metrics = eval_videos(model, context, cfg)
        diagnostic_paths = write_diagnostics(model, context, cfg, exp_dir)
        result = {
            "exp_name": exp_name,
            **cfg,
            **metadata,
            "best_epoch": int(ckpt.get("best_epoch", -1)),
            "best_val_score": float(ckpt.get("best_score", float("nan"))),
            "elapsed_min": 0.0,
            "skipped": True,
            "completion_status": "skipped",
            "checkpoint_path": str(best_path),
            "latest_path": str(latest_path),
            "history_path": str(hist_path),
            "exp_dir": str(exp_dir),
            **video_metrics,
        }
        write_manifest(manifest_path, completed_manifest(cfg, result, context, diagnostic_paths))
        result["diagnostic_paths"] = {key: str(value) for key, value in diagnostic_paths.items()}
        return result

    train_ds = ta.CausalWindowDataset(
        context.train_meta,
        labels=context.labels,
        clip_len=int(cfg["clip_len"]),
        stride=int(cfg["train_stride"]),
        train=True,
        aug=bool(cfg["aug"]),
        derivative_mode=str(cfg["derivative_mode"]),
        phase_label_scheme=str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)),
    )
    val_ds = ta.CausalWindowDataset(
        context.val_meta,
        labels=context.labels,
        clip_len=int(cfg["clip_len"]),
        stride=int(cfg["train_stride"]),
        train=False,
        aug=False,
        derivative_mode=str(cfg["derivative_mode"]),
        phase_label_scheme=str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)),
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"empty dataset windows: train={len(train_ds)} val={len(val_ds)}")
    batch = int(cfg["batch"])
    pin_memory = bool(cfg["pin_memory"]) and ta.DEVICE == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        drop_last=len(train_ds) >= batch,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=int(cfg["num_workers"]), pin_memory=pin_memory)
    print(f"  train={len(train_ds)} | val={len(val_ds)} windows", flush=True)

    class_weight, phase_weight = ta.compute_weights_from_ds(train_ds, device=ta.DEVICE)
    model = build_model(cfg, device=ta.DEVICE)
    metadata = model_metadata(model)
    print(
        f"  params={metadata['num_params_total']:,} | phase_head={metadata['phase_head_params']:,} "
        f"| joint_attn={metadata['joint_attn_params']:,}",
        flush=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(target_epochs, 1))
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_action_acc": [],
        "val_action_f1": [],
        "val_phase_acc": [],
        "val_phase_f1": [],
    }
    best_score = -1e9
    best_epoch = -1
    start_epoch = 1
    best_path = existing_bests[-1] if existing_bests else exp_dir / "best_ep000.pt"

    if latest_path.exists() and bool(cfg["resume"]):
        ckpt = torch.load(latest_path, map_location=ta.DEVICE)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", -1e9))
        best_epoch = int(ckpt.get("best_epoch", -1))
        history = ckpt.get("history", history)
        print(f"  -> resume from epoch {start_epoch}", flush=True)

    started = time.time()
    for ep in range(start_epoch, target_epochs + 1):
        model.train()
        losses: List[float] = []
        for x, y_cls, y_phase in tqdm(train_dl, desc=f"ep{ep:02d}", leave=False):
            x, y_cls, y_phase = x.to(ta.DEVICE), y_cls.to(ta.DEVICE), y_phase.to(ta.DEVICE)
            loss, _, _ = compute_loss_and_logits(
                model,
                x,
                y_cls,
                y_phase,
                class_weight,
                phase_weight,
                float(cfg["phase_loss_alpha"]),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        val_metrics = eval_windows(model, val_dl, class_weight, phase_weight, float(cfg["phase_loss_alpha"]))
        score = float(val_metrics["action_f1"] + val_metrics["phase_f1"])
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_action_acc"].append(float(val_metrics["action_acc"]))
        history["val_action_f1"].append(float(val_metrics["action_f1"]))
        history["val_phase_acc"].append(float(val_metrics["phase_acc"]))
        history["val_phase_f1"].append(float(val_metrics["phase_f1"]))
        print(
            f"  [ep {ep:02d}/{target_epochs}] loss={train_loss:.3f} | "
            f"a_f1={val_metrics['action_f1']:.3f} p_f1={val_metrics['phase_f1']:.3f} score={score:.3f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_epoch = ep
            for old in exp_dir.glob("best_ep*.pt"):
                old.unlink()
            best_path = exp_dir / f"best_ep{ep:03d}.pt"
            torch.save(checkpoint_payload(model, cfg, ep, best_score, best_epoch), best_path)

        torch.save(
            checkpoint_payload(model, cfg, ep, best_score, best_epoch, history=history, optimizer=opt, scheduler=sched),
            latest_path,
        )
        hist_path.write_text(json.dumps(history, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    elapsed_min = round((time.time() - started) / 60, 3)
    if not best_path.exists():
        best_path = latest_path
    ckpt = torch.load(best_path, map_location=ta.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    video_metrics = eval_videos(model, context, cfg)
    diagnostic_paths = write_diagnostics(model, context, cfg, exp_dir)
    result = {
        "exp_name": exp_name,
        **cfg,
        **metadata,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "elapsed_min": elapsed_min,
        "skipped": False,
        "completion_status": "complete",
        "checkpoint_path": str(best_path),
        "latest_path": str(latest_path),
        "history_path": str(hist_path),
        "exp_dir": str(exp_dir),
        **video_metrics,
    }
    write_manifest(manifest_path, completed_manifest(cfg, result, context, diagnostic_paths))
    result["diagnostic_paths"] = {key: str(value) for key, value in diagnostic_paths.items()}
    print(f"  -> complete | best_ep={best_epoch} best_score={best_score:.4f} | {elapsed_min:.2f}min")
    return result


def main(run_kind: str = RUN_KIND, variants: Optional[Iterable[str]] = None) -> Dict[str, Path]:
    run_kind = str(run_kind).strip().lower()
    selected_variants = [normalize_variant(v) for v in (variants or env_variants())]
    run_tag = RUN_TAG
    if FORCE_RETRAIN and FRESH_RERUN and not run_tag:
        run_tag = f"rerun_{utc_stamp()}"
    cfgs = [build_cfg(variant, run_kind, run_tag=run_tag) for variant in selected_variants]
    output_root = Path(cfgs[0]["output_root"])
    print(
        f"[phase_exercise_attn_experiment] run_kind={run_kind} epochs={cfgs[0]['epochs']} "
        f"variants={selected_variants} force_retrain={FORCE_RETRAIN} resume={cfgs[0]['resume']} "
        f"output_root={output_root}",
        flush=True,
    )
    context = ta.prepare_context(cfgs[0], verbose=True, update_globals=True)

    aggregate_rows: List[Dict[str, Any]] = []
    per_exercise_frames: List[pd.DataFrame] = []
    attention_frames: List[pd.DataFrame] = []
    failures: List[str] = []

    for cfg in cfgs:
        result: Optional[Dict[str, Any]] = None
        diagnostic_paths: Optional[Dict[str, Path]] = None
        try:
            result = run_one_experiment(cfg, context)
            diagnostic_paths = {key: Path(value) for key, value in result.get("diagnostic_paths", {}).items()}
            aggregate_rows.append(aggregate_row_from_result(cfg, result=result, diagnostic_paths=diagnostic_paths))
            per_ex_path = Path(diagnostic_paths["per_exercise"])
            if per_ex_path.exists():
                per_ex = pd.read_csv(per_ex_path)
                per_ex.insert(0, "variant", cfg["variant"])
                per_ex.insert(1, "exercise_source", cfg["exercise_source"])
                per_ex.insert(2, "derivative_mode", cfg["derivative_mode"])
                per_ex.insert(3, "input_channels", int(cfg["input_channels"]))
                per_ex.insert(4, "run_kind", run_kind)
                per_exercise_frames.append(per_ex)
            attn_path = Path(diagnostic_paths["attention_summary"])
            if attn_path.exists():
                attention_frames.append(pd.read_csv(attn_path))
        except Exception as exc:
            exp_dir = Path(cfg["output_root"]) / str(cfg["variant"])
            write_manifest(exp_dir / "run_manifest.json", failed_manifest(cfg, exc))
            failures.append(f"{cfg['variant']}: {type(exc).__name__}: {exc}")
            aggregate_rows.append(aggregate_row_from_result(cfg, result=result, error=f"{type(exc).__name__}: {exc}"))
            print(f"[FAILED] {cfg['variant']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

        update_aggregates(output_root, aggregate_rows, per_exercise_frames, attention_frames, print_summary=False)

    artifacts = update_aggregates(output_root, aggregate_rows, per_exercise_frames, attention_frames, print_summary=True)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError("Exercise attention experiment failed for required variant(s): " + "; ".join(failures))
    return artifacts


if __name__ == "__main__":
    main()
