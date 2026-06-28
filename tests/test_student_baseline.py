import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.student_model import StudentModel
import src.training.student_train as student_train
from src.training.student_train import compute_student_batch_losses
from src.utils.rank_logging import rank0_print


def test_student_forward_outputs_normalized_512d_embedding():
    torch.manual_seed(7)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(3, 3, 224, 224)

    with torch.no_grad():
        embedding = model(x)

    assert embedding.shape == (3, 512)
    torch.testing.assert_close(
        embedding.norm(p=2, dim=1),
        torch.ones(3),
        atol=1e-5,
        rtol=1e-5,
    )


def test_student_forward_matches_f4_gap_bn_l2_path():
    torch.manual_seed(13)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        embedding = model(x)
        f4 = model.backbone(x)[-1]
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        expected = F.normalize(model.neck(desc), dim=1)

    torch.testing.assert_close(embedding, expected, atol=1e-6, rtol=1e-6)


def test_student_batch_loss_is_only_symmetric_infonce():
    class IdentityFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(x.float(), dim=1)

    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
        [0.1, 0.9],
    ])
    model = IdentityFeatureModel()
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)

    losses = compute_student_batch_losses(
        model,
        features,
        pair_batch_size=2,
        criterion=criterion,
    )
    expected = student_train.sample4geo_loss(
        model,
        F.normalize(features, dim=1),
        criterion,
        pair_batch_size=2,
    )

    assert set(losses) == {
        "loss",
        "main_loss",
        "global_pair_batch_size",
    }
    torch.testing.assert_close(losses["loss"], expected)
    torch.testing.assert_close(losses["main_loss"], expected)


def test_cli_defaults_to_clean_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["student_train.py"])
    args = student_train.parse_args()

    removed_attrs = [
        "use_" + "pro" + "xy_loss",
        "pro" + "xy_loss_weight",
        "pro" + "xy_scale",
        "pro" + "xy_label_smoothing",
        "num_" + "train_" + "ids",
    ]
    for attr in removed_attrs:
        assert not hasattr(args, attr)


def test_removed_student_experiment_flags_are_rejected(monkeypatch):
    removed_flags = [
        "--use_" + "soft" + "_" + "orth" + "_fusion",
        "--" + "soft" + "_" + "orth" + "_apply_views",
        "--" + "soft" + "_" + "orth" + "_lambda_init",
        "--" + "soft" + "_" + "orth" + "_gamma_init",
        "--" + "soft" + "_" + "orth" + "_proj_init",
        "--use_" + "local" + "_align",
        "--use_" + "pro" + "xy_loss",
        "--" + "pro" + "xy_loss_weight",
        "--" + "pro" + "xy_scale",
        "--" + "pro" + "xy_label_smoothing",
        "--num_" + "train_" + "ids",
    ]
    for flag in removed_flags:
        monkeypatch.setattr(sys, "argv", ["student_train.py", flag])
        with pytest.raises(SystemExit):
            student_train.parse_args()


def test_student_source_has_no_removed_experiment_strings():
    forbidden = [
        "soft" + "_orth",
        "local" + "_align",
        "F4" + "LocalAlignmentLoss",
        "loss_" + "local" + "_align",
        "gamma" + "_raw",
        "lambda" + "_raw",
        "f3" + "_proj",
        "dual" + "_path",
        "preserve" + "_loss",
        "pro" + "xy",
        "ViewSharedIdentity" + "Pro" + "xyLoss",
        "distill" + "_projection",
        "training-only " + "distillation",
    ]
    paths = [
        os.path.join(ROOT, "src", "models", "student_model.py"),
        os.path.join(ROOT, "src", "training", "student_train.py"),
        os.path.join(ROOT, "src", "inference", "student_eval.py"),
    ]

    assert not os.path.exists(
        os.path.join(ROOT, "src", "loss", "pro" + "xy_loss.py")
    )
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        assert not any(needle in text for needle in forbidden), path


def test_rank0_print_uses_launcher_rank_before_dist_init(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "1")
    rank0_print("hidden")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("RANK", "0")
    rank0_print("visible")
    assert capsys.readouterr().out.strip() == "visible"
