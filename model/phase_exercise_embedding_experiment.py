"""ST-GCN + MLP phase head with exercise one-hot concatenation.

Architecture:
    shared ST-GCN backbone
      -> action head: exercise/action classification
      -> single phase head: concat(phase_feature, exercise_one_hot)

This is the lightweight alternative to exercise-specific phase heads. In a
service where the user selects the exercise, keep EXERCISE_SOURCE="selected".
If exercise is not user-selected, set EXERCISE_SOURCE="pred" to route phase
prediction with the model's predicted action id.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from . import phase_metrics
    from . import train_ablation as ta
except ImportError:
    import phase_metrics  # type: ignore
    import train_ablation as ta  # type: ignore


RUN_KIND = "full"  # "full" = 60 epochs, "smoke" = 1 epoch
SMOKE_EPOCHS = 1
FULL_EPOCHS = 60
BATCH = 256  # try 128/256 for speed if VRAM allows; 64 keeps baseline comparability
PHASE_LOSS_ALPHA = 2.0
PHASE_POOLINGS = ["last"]  # or ["temporal_flatten", "last", "avg_last_concat"]
EXERCISE_SOURCE = "pred"  # "selected"/"gt" or "pred"
SMOKE_MAX_VIDEOS_PER_SPLIT_TYPE = 2
OUTPUT_ROOT = Path("phase_experiments") / "exercise_embedding"
SMOOTH_WINDOW = 5

FORCE_RETRAIN = True
FRESH_RERUN = True
RUN_TAG = None

NOTE = (
    "Single shared phase head receives concat(phase_feature, exercise_one_hot). "
    "EXERCISE_SOURCE='selected' simulates a service where the user chooses the exercise."
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def clean_token(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")


def phase_base_dim(phase_pooling: str, clip_len: int) -> int:
    c = 256
    t_prime = ta.stgcn_temporal_out_len(int(clip_len))
    if phase_pooling in {"temporal_avg", "last"}:
        return c
    if phase_pooling == "temporal_flatten":
        return c * t_prime
    if phase_pooling == "avg_last_concat":
        return c * 2
    raise ValueError(f"unsupported phase_pooling={phase_pooling}")


class MultiTaskSTGCNMLPExerciseEmbedding(nn.Module):
    def __init__(
        self,
        num_action: int = ta.NUM_CLASSES,
        num_phase: int = ta.NUM_PHASES,
        in_c: int = 3,
        mlp_hidden: int = 128,
        dropout: float = 0.3,
        phase_pooling: str = "last",
        clip_len: int = 16,
    ):
        super().__init__()
        if phase_pooling not in ta.ALLOWED_PHASE_POOLING:
            raise ValueError(f"unsupported phase_pooling={phase_pooling}")
        self.num_action = int(num_action)
        self.phase_pooling = str(phase_pooling)
        self.clip_len = int(clip_len)
        self.expected_tprime = ta.stgcn_temporal_out_len(self.clip_len)
        self.backbone = ta.STGCNBackbone(in_channels=in_c)
        c = self.backbone.out_channels
        self.pooling_feature_dim = phase_base_dim(self.phase_pooling, self.clip_len)
        self.exercise_embedding_dim = self.num_action
        self.phase_feature_dim = self.pooling_feature_dim + self.exercise_embedding_dim
        self.action_head = nn.Sequential(
            nn.Linear(c, c // 2), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(c // 2, num_action)
        )
        self.phase_head = nn.Sequential(
            nn.Linear(self.phase_feature_dim, mlp_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, num_phase),
        )

    def _phase_base_features(self, f: torch.Tensor) -> torch.Tensor:
        ft = f.mean(dim=-1).permute(0, 2, 1).contiguous()
        actual_t = int(ft.size(1))
        if actual_t != self.expected_tprime:
            raise RuntimeError(
                f"temporal length mismatch: clip_len={self.clip_len}, expected T'={self.expected_tprime}, actual T'={actual_t}"
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

    def forward(self, x: torch.Tensor, exercise_id: Optional[torch.Tensor] = None):
        f = self.backbone(x)
        action_logit = self.action_head(f.mean(dim=(2, 3)))
        if exercise_id is None:
            exercise_id = action_logit.argmax(dim=1)
        exercise_id = exercise_id.to(device=x.device, dtype=torch.long)
        exercise_one_hot = F.one_hot(exercise_id, num_classes=self.num_action).to(dtype=f.dtype)
        phase_feat = torch.cat([self._phase_base_features(f), exercise_one_hot], dim=1)
        return action_logit, self.phase_head(phase_feat)


def build_cfg(phase_pooling: str, run_kind: str, run_tag: Optional[str] = None) -> Dict[str, Any]:
    run_kind = str(run_kind).lower()
    if run_kind not in {"smoke", "full"}:
        raise ValueError(f"run_kind must be smoke or full, got {run_kind}")
    if FORCE_RETRAIN and FRESH_RERUN and not run_tag:
        run_tag = f"rerun_{utc_stamp()}"
    cfg = copy.deepcopy(ta.DEFAULT_POOLING_EXPERIMENT)
    cfg.update(
        {
            "architecture": "mlp_exercise_embedding_concat",
            "model_type": "mlp",
            "phase_pooling": phase_pooling,
            "exercise_source": EXERCISE_SOURCE,
            "hidden": 128,
            "clip_len": 16,
            "train_stride": 2,
            "dropout": 0.3,
            "aug": True,
            "epochs": SMOKE_EPOCHS if run_kind == "smoke" else FULL_EPOCHS,
            "batch": BATCH,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "phase_loss_alpha": PHASE_LOSS_ALPHA,
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
    return ta.normalize_cfg(cfg)


def make_exp_name(cfg: Mapping[str, Any]) -> str:
    base = (
        f"mlp_exemb_pool_{cfg['phase_pooling']}_h{cfg['hidden']}_c{cfg['clip_len']}"
        f"_pw_{float(cfg['phase_loss_alpha'])}_ts{cfg['train_stride']}"
        f"_do{str(float(cfg['dropout'])).replace('.', '')}_aug{int(bool(cfg['aug']))}"
        f"_src_{cfg.get('exercise_source', EXERCISE_SOURCE)}"
    )
    tag = cfg.get("fresh_run_tag")
    return f"{base}_{clean_token(tag)}" if tag else base


def build_model(cfg: Mapping[str, Any], device: Optional[str | torch.device] = None) -> MultiTaskSTGCNMLPExerciseEmbedding:
    model = MultiTaskSTGCNMLPExerciseEmbedding(
        num_action=ta.NUM_CLASSES,
        num_phase=ta.NUM_PHASES,
        in_c=3,
        mlp_hidden=int(cfg["hidden"]),
        dropout=float(cfg["dropout"]),
        phase_pooling=str(cfg["phase_pooling"]),
        clip_len=int(cfg["clip_len"]),
    )
    return model.to(torch.device(device or ta.DEVICE))


def model_metadata(model: MultiTaskSTGCNMLPExerciseEmbedding) -> Dict[str, int]:
    return {
        "num_params_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "phase_head_params": int(sum(p.numel() for p in model.phase_head.parameters() if p.requires_grad)),
        "pooling_feature_dim": int(model.pooling_feature_dim),
        "exercise_embedding_dim": int(model.exercise_embedding_dim),
        "phase_feature_dim": int(model.phase_feature_dim),
    }


def exercise_ids_for_batch(y_cls: torch.Tensor, source: str) -> Optional[torch.Tensor]:
    source = str(source).lower()
    if source in {"selected", "gt", "ground_truth", "true"}:
        return y_cls
    if source == "pred":
        return None
    raise ValueError(f"unsupported EXERCISE_SOURCE={source}")


def compute_loss_and_logits(model, x, y_cls, y_phase, class_weight, phase_weight, alpha, source):
    exercise_id = exercise_ids_for_batch(y_cls, source)
    action_logit, phase_logit = model(x, exercise_id=exercise_id)
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
def eval_windows(model, dl, class_weight, phase_weight, alpha: float, source: str) -> Dict[str, float]:
    model.eval()
    a_preds: List[int] = []
    a_gts: List[int] = []
    p_preds: List[int] = []
    p_gts: List[int] = []
    losses: List[float] = []
    for x, y_cls, y_phase in dl:
        x, y_cls, y_phase = x.to(ta.DEVICE), y_cls.to(ta.DEVICE), y_phase.to(ta.DEVICE)
        loss, action_logit, phase_logit = compute_loss_and_logits(
            model, x, y_cls, y_phase, class_weight, phase_weight, alpha, source
        )
        losses.append(float(loss.item()))
        a_preds.extend(action_logit.argmax(1).cpu().tolist())
        a_gts.extend(y_cls.cpu().tolist())
        p_preds.extend(phase_logit.argmax(1).cpu().tolist())
        p_gts.extend(y_phase.cpu().tolist())
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "action_acc": float(accuracy_score(a_gts, a_preds)) if a_gts else float("nan"),
        "action_f1": float(f1_score(a_gts, a_preds, average="macro", zero_division=0)) if a_gts else float("nan"),
        "phase_acc": float(accuracy_score(p_gts, p_preds)) if p_gts else float("nan"),
        "phase_f1": float(f1_score(p_gts, p_preds, average="macro", zero_division=0)) if p_gts else float("nan"),
    }


@torch.no_grad()
def predict_from_kpts(model, kpts, cfg, exercise_id: Optional[int], smooth_window: int = 5, min_up_len: int = 3):
    model.eval()
    k = ta.normalize_kpts(kpts.astype(np.float32))
    T = len(k)
    if T == 0:
        raise ValueError("empty pose sequence")
    clip_len = int(cfg["clip_len"])
    stride = int(cfg["train_stride"])
    batch_size = int(cfg["batch"])
    phase_raw = np.zeros(T, dtype=np.int64) + ta.PHASE_READY
    phase_prob = np.zeros((T, ta.NUM_PHASES), dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)
    action_probs_all: List[np.ndarray] = []
    ends = [T - 1] if T < clip_len else list(range(clip_len - 1, T, stride))
    if T >= clip_len and ends[-1] != T - 1:
        ends.append(T - 1)
    ends = sorted(set(ends))
    clips: List[np.ndarray] = []
    clip_ends: List[int] = []

    def flush(batch_clips: List[np.ndarray], batch_ends: List[int]) -> None:
        xb = torch.stack(
            [torch.from_numpy(c).permute(2, 0, 1).unsqueeze(-1).contiguous() for c in batch_clips]
        ).to(ta.DEVICE)
        ex = None
        if exercise_id is not None:
            ex = torch.full((len(batch_clips),), int(exercise_id), dtype=torch.long, device=ta.DEVICE)
        action_logit, phase_logit = model(xb, exercise_id=ex)
        a_prob = F.softmax(action_logit, dim=1).cpu().numpy()
        p_prob = F.softmax(phase_logit, dim=1).cpu().numpy()
        for e, ap, pp in zip(batch_ends, a_prob, p_prob):
            action_probs_all.append(ap)
            phase_raw[e] = int(pp.argmax())
            phase_prob[e] = pp
            valid_mask[e] = True

    for end in ends:
        start = end - clip_len + 1
        clip = k[start : end + 1] if start >= 0 else np.concatenate([np.tile(k[0:1], (-start, 1, 1)), k[: end + 1]], axis=0)
        clips.append(clip)
        clip_ends.append(end)
        if len(clips) == batch_size:
            flush(clips, clip_ends)
            clips, clip_ends = [], []
    if clips:
        flush(clips, clip_ends)
    last_phase = ta.PHASE_READY
    last_prob = np.eye(ta.NUM_PHASES, dtype=np.float32)[ta.PHASE_READY]
    for t in range(T):
        if valid_mask[t]:
            last_phase = phase_raw[t]
            last_prob = phase_prob[t]
        else:
            phase_raw[t] = last_phase
            phase_prob[t] = last_prob
    phase_smooth = ta.smooth_phase(phase_raw, window=smooth_window)
    pred_count, transitions = ta.count_phases(phase_smooth, min_up_len=min_up_len)
    action_vote = np.mean(np.stack(action_probs_all), axis=0) if action_probs_all else np.eye(ta.NUM_CLASSES, dtype=np.float32)[0]
    pred_cls = int(action_vote.argmax())
    return {
        "T": T,
        "pred_cls_id": pred_cls,
        "pred_class": ta.ID_TO_CLASS[pred_cls],
        "action_probs": action_vote,
        "phase_raw": phase_raw,
        "phase_smooth": phase_smooth,
        "phase_prob": phase_prob,
        "valid_mask": valid_mask,
        "pred_count": int(pred_count),
        "transitions": transitions,
    }


def selected_exercise_id(row: pd.Series, source: str) -> Optional[int]:
    if str(source).lower() in {"selected", "gt", "ground_truth", "true"}:
        return int(row["cls"])
    if str(source).lower() == "pred":
        return None
    raise ValueError(f"unsupported exercise source: {source}")


def eval_videos(model, context: ta.ExperimentContext, cfg: Mapping[str, Any]) -> Dict[str, float]:
    rows: List[Dict[str, Any]] = []
    all_gt: List[int] = []
    all_pred: List[int] = []
    phase_label_scheme = str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED))
    for _, row in context.val_meta.iterrows():
        kpts = np.load(row["npz"])["kpts"].astype(np.float32)
        T = min(int(row["T"]), len(kpts))
        out = predict_from_kpts(model, kpts[:T], cfg, selected_exercise_id(row, cfg["exercise_source"]), smooth_window=SMOOTH_WINDOW)
        reps = context.labels[(row["type"], row["name"])]["reps"]
        gt_count = len(reps)
        gt_phase = ta.make_phase_target(
            T,
            reps,
            exercise_type=row["type"],
            phase_label_scheme=phase_label_scheme,
        )
        valid = np.arange(T) >= (int(cfg["clip_len"]) - 1 if T >= int(cfg["clip_len"]) else T - 1)
        all_gt.extend(gt_phase[valid].tolist())
        all_pred.extend(out["phase_smooth"][valid].tolist())
        rows.append(
            {"type": row["type"], "gt_cls": row["type"], "pred_cls": out["pred_class"], "gt_count": gt_count,
             "pred_count": out["pred_count"], "abs_error": abs(out["pred_count"] - gt_count)}
        )
    vr = pd.DataFrame(rows)
    per_class: Dict[str, float] = {}
    for typ in ta.CLASS_LIST:
        sub = vr[vr["type"] == typ]
        per_class[f"mae_{typ}"] = float(sub["abs_error"].mean()) if len(sub) else float("nan")
        per_class[f"obo_{typ}"] = float((sub["abs_error"] <= 1).mean()) if len(sub) else float("nan")
    return {
        "video_action_acc": float(accuracy_score(vr["gt_cls"], vr["pred_cls"])),
        "video_action_f1": float(f1_score(vr["gt_cls"], vr["pred_cls"], average="macro", zero_division=0)),
        "video_phase_acc": float(accuracy_score(all_gt, all_pred)) if all_gt else float("nan"),
        "video_phase_f1": float(f1_score(all_gt, all_pred, average="macro", zero_division=0)) if all_gt else float("nan"),
        "video_count_mae": float(mean_absolute_error(vr["gt_count"], vr["pred_count"])),
        "video_count_obo": float((vr["abs_error"] <= 1).mean()),
        **per_class,
    }


def boundary_distance(frame_idx: int, reps: Iterable[tuple[int, int, int]]) -> float:
    points: List[int] = []
    for s, b, f in reps:
        points.extend([int(s), int(b), int(f)])
    return float(min(abs(int(frame_idx) - p) for p in points)) if points else float("nan")


def prediction_records(model, context: ta.ExperimentContext, cfg: Mapping[str, Any]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    phase_label_scheme = str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED))
    for _, row in context.val_meta.iterrows():
        kpts = np.load(row["npz"])["kpts"].astype(np.float32)
        T = min(int(row["T"]), len(kpts))
        reps = context.labels[(row["type"], row["name"])]["reps"]
        gt_phase = ta.make_phase_target(
            T,
            reps,
            exercise_type=row["type"],
            phase_label_scheme=phase_label_scheme,
        )
        out = predict_from_kpts(model, kpts[:T], cfg, selected_exercise_id(row, cfg["exercise_source"]), smooth_window=SMOOTH_WINDOW)
        for frame_idx in np.where(np.asarray(out["valid_mask"], dtype=bool))[0]:
            probs = np.asarray(out["phase_prob"][frame_idx], dtype=float)
            records.append(
                {
                    "type": row["type"], "name": row["name"], "split": row.get("split", "val"),
                    "frame_idx": int(frame_idx), "clip_len": int(cfg["clip_len"]), "stride": int(cfg["train_stride"]),
                    "exercise_source": cfg["exercise_source"], "phase_label_scheme": phase_label_scheme,
                    "gt_phase": int(gt_phase[frame_idx]),
                    "gt_phase_name": ta.PHASE_NAMES[int(gt_phase[frame_idx])],
                    "pred_phase_raw": int(out["phase_raw"][frame_idx]),
                    "pred_phase_raw_name": ta.PHASE_NAMES[int(out["phase_raw"][frame_idx])],
                    "pred_phase_offline_smooth": int(out["phase_smooth"][frame_idx]),
                    "pred_phase_offline_smooth_name": ta.PHASE_NAMES[int(out["phase_smooth"][frame_idx])],
                    "phase_prob_ready": float(probs[ta.PHASE_READY]), "phase_prob_down": float(probs[ta.PHASE_DOWN]),
                    "phase_prob_up": float(probs[ta.PHASE_UP]), "boundary_distance": boundary_distance(int(frame_idx), reps),
                    "action_pred": out["pred_class"],
                    "action_prob_squat": float(out["action_probs"][ta.CLASS_TO_ID["squat"]]),
                    "action_prob_benchpress": float(out["action_probs"][ta.CLASS_TO_ID["benchpress"]]),
                    "action_prob_deadlift": float(out["action_probs"][ta.CLASS_TO_ID["deadlift"]]),
                }
            )
    if not records:
        raise RuntimeError("no diagnostic prediction records generated")
    return pd.DataFrame(records)


def write_diagnostics(model, context: ta.ExperimentContext, cfg: Mapping[str, Any], exp_dir: Path) -> Dict[str, Path]:
    preds = prediction_records(model, context, cfg)
    artifacts = phase_metrics.combined_metric_artifacts(preds, ta.PHASE_NAMES)
    paths = {
        "raw_predictions": exp_dir / "raw_phase_predictions.csv",
        "metrics": exp_dir / "phase_diagnostics.csv",
        "per_exercise": exp_dir / "phase_diagnostics_by_exercise.csv",
        "confusion_matrix": exp_dir / "phase_confusion_matrix.csv",
        "boundary_summary": exp_dir / "phase_boundary_summary.csv",
    }
    preds.to_csv(paths["raw_predictions"], index=False, encoding="utf-8-sig")
    artifacts["summaries"].to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    artifacts["per_exercise"].to_csv(paths["per_exercise"], index=False, encoding="utf-8-sig")
    artifacts["confusion"].to_csv(paths["confusion_matrix"], index=True, encoding="utf-8-sig")
    artifacts["boundary"].to_csv(paths["boundary_summary"], index=False, encoding="utf-8-sig")
    return paths


def checkpoint_payload(model, cfg, epoch, best_score, best_epoch, history=None, optimizer=None, scheduler=None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model.state_dict(), "cfg": dict(cfg), "epoch": int(epoch), "best_score": float(best_score),
        "best_epoch": int(best_epoch), "classes": ta.CLASS_LIST, "phase_names": ta.PHASE_NAMES, **model_metadata(model),
    }
    if history is not None:
        payload["history"] = history
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def run_one_experiment(cfg: Dict[str, Any], context: ta.ExperimentContext) -> Dict[str, Any]:
    exp_name = make_exp_name(cfg)
    exp_dir = Path(cfg["output_root"]) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    latest_path = exp_dir / "latest.pt"
    hist_path = exp_dir / "history.json"
    manifest_path = exp_dir / "run_manifest.json"
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"[EXP] {exp_name}")
    print(
        f"  architecture=exercise_embedding_concat pooling={cfg['phase_pooling']} "
        f"exercise_source={cfg['exercise_source']} epochs={cfg['epochs']} batch={cfg['batch']}",
        flush=True,
    )
    train_ds = ta.CausalWindowDataset(
        context.train_meta,
        labels=context.labels,
        clip_len=int(cfg["clip_len"]),
        stride=int(cfg["train_stride"]),
        train=True,
        aug=bool(cfg["aug"]),
        phase_label_scheme=str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)),
    )
    val_ds = ta.CausalWindowDataset(
        context.val_meta,
        labels=context.labels,
        clip_len=int(cfg["clip_len"]),
        stride=int(cfg["train_stride"]),
        train=False,
        aug=False,
        phase_label_scheme=str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)),
    )
    batch = int(cfg["batch"])
    pin_memory = bool(cfg["pin_memory"]) and ta.DEVICE == "cuda"
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=int(cfg["num_workers"]), drop_last=len(train_ds) >= batch, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=int(cfg["num_workers"]), pin_memory=pin_memory)
    print(f"  train={len(train_ds)} | val={len(val_ds)} windows", flush=True)
    class_weight, phase_weight = ta.compute_weights_from_ds(train_ds, device=ta.DEVICE)
    model = build_model(cfg, device=ta.DEVICE)
    metadata = model_metadata(model)
    print(
        f"  params={metadata['num_params_total']:,} | phase_head={metadata['phase_head_params']:,} "
        f"| pooling_dim={metadata['pooling_feature_dim']} | phase_feature_dim={metadata['phase_feature_dim']}",
        flush=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(cfg["epochs"]), 1))
    history: Dict[str, List[float]] = {"train_loss": [], "val_action_acc": [], "val_action_f1": [], "val_phase_acc": [], "val_phase_f1": []}
    best_score = -1e9
    best_epoch = -1
    best_path = exp_dir / "best_ep000.pt"
    started = time.time()
    try:
        for ep in range(1, int(cfg["epochs"]) + 1):
            model.train()
            losses: List[float] = []
            for x, y_cls, y_phase in tqdm(train_dl, desc=f"ep{ep:02d}", leave=False):
                x, y_cls, y_phase = x.to(ta.DEVICE), y_cls.to(ta.DEVICE), y_phase.to(ta.DEVICE)
                loss, _, _ = compute_loss_and_logits(model, x, y_cls, y_phase, class_weight, phase_weight, float(cfg["phase_loss_alpha"]), str(cfg["exercise_source"]))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.item()))
            sched.step()
            vm = eval_windows(model, val_dl, class_weight, phase_weight, float(cfg["phase_loss_alpha"]), str(cfg["exercise_source"]))
            score = float(vm["action_f1"] + vm["phase_f1"])
            train_loss = float(np.mean(losses)) if losses else float("nan")
            history["train_loss"].append(train_loss)
            history["val_action_acc"].append(float(vm["action_acc"]))
            history["val_action_f1"].append(float(vm["action_f1"]))
            history["val_phase_acc"].append(float(vm["phase_acc"]))
            history["val_phase_f1"].append(float(vm["phase_f1"]))
            print(f"  [ep {ep:02d}/{cfg['epochs']}] loss={train_loss:.3f} | a_f1={vm['action_f1']:.3f} p_f1={vm['phase_f1']:.3f} score={score:.3f}", flush=True)
            if score > best_score:
                best_score = score
                best_epoch = ep
                for old in exp_dir.glob("best_ep*.pt"):
                    old.unlink()
                best_path = exp_dir / f"best_ep{ep:03d}.pt"
                torch.save(checkpoint_payload(model, cfg, ep, best_score, best_epoch), best_path)
            torch.save(checkpoint_payload(model, cfg, ep, best_score, best_epoch, history, opt, sched), latest_path)
            hist_path.write_text(json.dumps(history, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
        ckpt = torch.load(best_path, map_location=ta.DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        video_metrics = eval_videos(model, context, cfg)
        diag_paths = write_diagnostics(model, context, cfg, exp_dir)
        elapsed_min = round((time.time() - started) / 60, 3)
        result = {
            "exp_name": exp_name, **cfg, **metadata, "best_epoch": int(best_epoch), "best_val_score": float(best_score),
            "elapsed_min": elapsed_min, "completion_status": "complete", "checkpoint_path": str(best_path),
            "latest_path": str(latest_path), "history_path": str(hist_path), "exp_dir": str(exp_dir), **video_metrics,
        }
        manifest = {
            "run_id": exp_name, "created_at_utc": utc_stamp(), "completion_status": "complete",
            "architecture": cfg["architecture"], "phase_pooling": cfg["phase_pooling"], "exercise_source": cfg["exercise_source"],
            "phase_label_scheme": cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED),
            "checkpoint_path": str(best_path), "config": cfg, "note": NOTE,
            "artifacts": {
                "latest": str(latest_path), "history": str(hist_path), "phase_diagnostics": str(diag_paths["metrics"]),
                "phase_diagnostics_by_exercise": str(diag_paths["per_exercise"]), "phase_confusion_matrix": str(diag_paths["confusion_matrix"]),
                "raw_predictions": str(diag_paths["raw_predictions"]),
            },
            **metadata,
        }
        write_manifest(manifest_path, manifest)
        print(f"  -> complete | best_ep={best_epoch} best_score={best_score:.4f} | {elapsed_min:.2f}min")
        return result
    except Exception as exc:
        write_manifest(
            manifest_path,
            {"run_id": exp_name, "created_at_utc": utc_stamp(), "completion_status": "failed", "architecture": cfg["architecture"],
             "phase_pooling": cfg["phase_pooling"], "exercise_source": cfg["exercise_source"],
             "phase_label_scheme": cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED), "config": cfg,
             "error": f"{type(exc).__name__}: {exc}", "note": NOTE},
        )
        raise


def update_aggregates(
    output_root: Path,
    rows: List[Dict[str, Any]],
    per_exercise_frames: List[pd.DataFrame],
    *,
    print_summary: bool = True,
) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "exercise_embedding_results.csv"
    results_json = output_root / "exercise_embedding_results.json"
    per_ex_csv = output_root / "exercise_embedding_per_exercise.csv"
    result_df = pd.DataFrame(rows)
    if not result_df.empty and "video_phase_f1" in result_df.columns:
        result_df = result_df.sort_values("video_phase_f1", ascending=False, na_position="last")
    result_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    results_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    per_ex = pd.concat(per_exercise_frames, ignore_index=True) if per_exercise_frames else pd.DataFrame()
    per_ex.to_csv(per_ex_csv, index=False, encoding="utf-8-sig")
    if print_summary and not result_df.empty:
        show = [
            "exp_name",
            "phase_pooling",
            "exercise_source",
            "phase_label_scheme",
            "best_epoch",
            "best_val_score",
            "video_phase_f1",
            "phase_feature_dim",
        ]
        print("\n[exercise_embedding_results summary]")
        print(result_df[[c for c in show if c in result_df.columns]].to_string(index=False))
    return {"results_csv": results_csv, "results_json": results_json, "per_exercise_csv": per_ex_csv}


def main(run_kind: str = RUN_KIND, poolings: Optional[Iterable[str]] = None) -> Dict[str, Path]:
    run_kind = str(run_kind).lower()
    poolings = list(poolings or PHASE_POOLINGS)
    run_tag = RUN_TAG
    if FORCE_RETRAIN and FRESH_RERUN and not run_tag:
        run_tag = f"rerun_{utc_stamp()}"
    cfgs = [build_cfg(pooling, run_kind, run_tag=run_tag) for pooling in poolings]
    print(
        f"[phase_exercise_embedding_experiment] run_kind={run_kind} epochs={cfgs[0]['epochs']} "
        f"poolings={poolings} exercise_source={EXERCISE_SOURCE} "
        f"label_scheme={cfgs[0].get('phase_label_scheme', ta.PHASE_LABEL_SCHEME_AS_LABELED)} force_retrain={FORCE_RETRAIN} "
        f"output_root={cfgs[0]['output_root']}",
        flush=True,
    )
    context = ta.prepare_context(cfgs[0], verbose=True, update_globals=True)
    rows: List[Dict[str, Any]] = []
    per_exercise_frames: List[pd.DataFrame] = []
    for cfg in cfgs:
        result = run_one_experiment(cfg, context)
        rows.append(result)
        per_ex_path = Path(result["exp_dir"]) / "phase_diagnostics_by_exercise.csv"
        if per_ex_path.exists():
            frame = pd.read_csv(per_ex_path)
            frame.insert(0, "exp_name", result["exp_name"])
            frame.insert(1, "phase_pooling", cfg["phase_pooling"])
            frame.insert(2, "exercise_source", cfg["exercise_source"])
            per_exercise_frames.append(frame)
        update_aggregates(Path(cfg["output_root"]), rows, per_exercise_frames, print_summary=False)
    artifacts = update_aggregates(Path(cfgs[0]["output_root"]), rows, per_exercise_frames, print_summary=True)
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, ensure_ascii=False, indent=2))
    return artifacts


if __name__ == "__main__":
    main()
