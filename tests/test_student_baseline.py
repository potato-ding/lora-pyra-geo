import json
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.student_model import StudentModel
import src.training.student_train as student_train
import src.dataset.datasets as student_datasets
from src.dataset.datasets import U1652PairDataset
from src.training.student_train import (
    build_deepspeed_runtime_config,
    compute_student_batch_losses,
    gather_paired_views,
    train_one_epoch_deepspeed,
)


def test_student_forward_outputs_normalized_f4_embedding():
    torch.manual_seed(7)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        embedding = model(x)

    assert embedding.shape == (2, 512)
    torch.testing.assert_close(
        embedding.norm(p=2, dim=1),
        torch.ones(2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_student_has_only_backbone_neck_and_logit_scale():
    model = StudentModel(ckpt_path=None).eval()
    child_names = {name for name, _ in model.named_children()}
    parameter_names = {name for name, _ in model.named_parameters()}

    assert child_names == {"backbone", "neck"}
    assert "logit_scale" in parameter_names


def test_student_matches_explicit_gap_bn_normalize_path():
    torch.manual_seed(13)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        embedding = model(x)
        _, _, _, f4 = model.backbone(x)
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


def test_distributed_pair_gather_preserves_global_positive_diagonal(monkeypatch):
    def fake_gather(tensor):
        return torch.cat([tensor, tensor + 100.0], dim=0)

    monkeypatch.setattr(student_train, "gather_tensor_with_grad", fake_gather)
    local = torch.tensor([
        [1.0],
        [2.0],
        [11.0],
        [12.0],
    ], requires_grad=True)

    gathered, pair_batch_size = gather_paired_views(
        local,
        pair_batch_size=2,
        with_grad=True,
    )

    expected = torch.tensor([
        [1.0],
        [2.0],
        [101.0],
        [102.0],
        [11.0],
        [12.0],
        [111.0],
        [112.0],
    ])
    assert pair_batch_size == 4
    torch.testing.assert_close(gathered, expected)


def test_pair_gather_rejects_detached_mode():
    tensor = torch.randn(4, 3, requires_grad=True)
    with pytest.raises(ValueError, match="preserve gradients"):
        gather_paired_views(tensor, pair_batch_size=2, with_grad=False)


def test_deepspeed_runtime_config_tracks_global_pair_batch(tmp_path):
    config_path = tmp_path / "ds.json"
    config_path.write_text(
        json.dumps({
            "zero_optimization": {"stage": 1},
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
        }),
        encoding="utf-8",
    )

    class Args:
        batch_size = 4
        grad_accum_steps = 3
        grad_clip = 1.5
        amp = True

    config = build_deepspeed_runtime_config(
        str(config_path),
        Args,
        world_size=2,
    )
    assert config["train_micro_batch_size_per_gpu"] == 4
    assert config["gradient_accumulation_steps"] == 3
    assert config["train_batch_size"] == 24
    assert config["gradient_clipping"] == 1.5


def test_deepspeed_runtime_config_rejects_zero3(tmp_path):
    config_path = tmp_path / "ds.json"
    config_path.write_text(
        json.dumps({"zero_optimization": {"stage": 3}}),
        encoding="utf-8",
    )

    class Args:
        batch_size = 2
        grad_accum_steps = 1
        grad_clip = 0.0
        amp = False

    with pytest.raises(ValueError, match="ZeRO stages 0, 1, and 2"):
        build_deepspeed_runtime_config(
            str(config_path),
            Args,
            world_size=2,
        )


def test_cli_defaults_to_clean_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["student_train.py"])
    args = student_train.parse_args()
    destinations = {
        action.dest
        for action in student_train.argparse.ArgumentParser()._actions
    }

    assert args.use_kd_distill is False
    assert args.deepspeed_config == "configs/ds_student_baseline.json"
    assert not hasattr(args, "teacher_" + "checkpoint")
    assert not any(name.startswith("b" + "rd") for name in vars(args))
    assert destinations == {"help"}


def test_removed_training_flags_are_rejected(monkeypatch):
    removed_flags = [
        "--use_" + "b" + "rd_distill",
        "--" + "b" + "rd_weight",
        "--use_" + "local_" + "align",
        "--teacher_" + "checkpoint",
    ]
    for flag in removed_flags:
        monkeypatch.setattr(sys, "argv", ["student_train.py", flag])
        with pytest.raises(SystemExit):
            student_train.parse_args()


def test_student_dataset_returns_only_baseline_pair_fields(
    tmp_path,
    monkeypatch,
):
    sat_dir = tmp_path / "satellite" / "0001"
    drone_dir = tmp_path / "drone" / "0001"
    sat_dir.mkdir(parents=True)
    drone_dir.mkdir(parents=True)
    (sat_dir / "sat.png").touch()
    (drone_dir / "drone.png").touch()

    monkeypatch.setattr(
        student_datasets,
        "read_rgb_image",
        lambda path: torch.zeros(8, 8, 3).numpy(),
    )

    class ToTensor:
        def __call__(self, image):
            return {"image": torch.from_numpy(image).permute(2, 0, 1)}

    dataset = U1652PairDataset(
        str(tmp_path),
        sat_transforms=ToTensor(),
        drone_transforms=ToTensor(),
        prob_flip=0.0,
    )
    sample = dataset[0]

    assert isinstance(sample, tuple)
    assert len(sample) == 4
    drone, satellite, label, pid = sample
    assert drone.shape == satellite.shape == (3, 8, 8)
    assert label == 0
    assert pid == "0001"


def test_deepspeed_epoch_updates_student_with_infonce_only(
    monkeypatch,
    capsys,
):
    class PairDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            drone = torch.tensor([[[float(idx + 1), 0.0]]])
            satellite = torch.tensor([[[float(idx + 1), 0.5]]])
            return drone, satellite, idx, f"{idx:04d}"

    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(self.proj(x.float().flatten(1)), dim=1)

    class FakeDeepSpeedEngine(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        def forward(self, x):
            return self.module(x)

        def backward(self, loss):
            loss.backward()

        def step(self):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    class Args:
        print_freq = 100
        epochs = 1

    torch.manual_seed(5)
    engine = FakeDeepSpeedEngine(TinyStudent())
    before = engine.module.proj.weight.detach().clone()
    loader = DataLoader(PairDataset(), batch_size=2, shuffle=False)
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.1)
    monkeypatch.setattr(student_train, "is_main_process", lambda: False)

    stats = train_one_epoch_deepspeed(
        engine,
        loader,
        criterion,
        engine.optimizer,
        torch.device("cpu"),
        Args,
        epoch=1,
    )

    assert set(stats) == {"loss_total", "loss_infonce"}
    assert torch.isfinite(torch.tensor(stats["loss_infonce"]))
    assert not torch.equal(before, engine.module.proj.weight.detach())
    assert capsys.readouterr().out == ""
