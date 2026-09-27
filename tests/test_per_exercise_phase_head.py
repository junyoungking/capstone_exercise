import torch

from model import train_ablation as ta


def _cfg(model_type: str):
    return ta.normalize_cfg(
        {
            "model_type": model_type,
            "phase_head_type": "per_exercise_mlp",
            "phase_conditioning": "ground_truth_exercise_label",
            "pose_backend": "mediapipe_barbell",
            "joint_subset": "all",
            "barbell_edge_policy": "wrists",
            "phase_label_scheme": "bar_direction",
            "clip_len": 16,
            "train_stride": 2,
            "batch": 2,
            "epochs": 1,
        }
    )


def test_per_exercise_head_normalizes_conditioning():
    cfg = _cfg("mlp")

    assert cfg["phase_head_type"] == ta.PHASE_HEAD_PER_EXERCISE_MLP
    assert cfg["phase_conditioning"] == ta.PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL
    assert cfg["exercise_id_source"] == ta.EXERCISE_ID_SOURCE_GROUND_TRUTH_LABEL


def test_per_exercise_head_forward_mlp_and_lstm():
    for model_type in ("mlp", "lstm"):
        cfg = _cfg(model_type)
        model = ta.build_model(cfg, device="cpu")
        x = torch.randn(2, int(cfg["input_channels"]), int(cfg["clip_len"]), int(cfg["num_kpt"]), 1)
        y_cls = torch.tensor([0, 2], dtype=torch.long)

        action_logit, phase_logit = ta.model_forward_with_conditioning(model, x, exercise_id=y_cls)

        assert tuple(action_logit.shape) == (2, ta.NUM_CLASSES)
        assert tuple(phase_logit.shape) == (2, ta.NUM_PHASES)
