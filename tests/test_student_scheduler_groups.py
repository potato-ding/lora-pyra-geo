"""Regression: every BF16/FP32 Student group receives warmup and cosine LR."""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.student.optimizer import build_student_optimizer
from src.student.part2 import MLPResidual, ScalarGate
from src.student.formal_top import prepare_student_precision_groups
from src.student.scheduler import (
    build_student_scheduler, assert_student_scheduler_groups,
)


class TrainingProbe(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.neck = nn.BatchNorm1d(8)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        if method != "S0":
            self.top = nn.Linear(8, 128)
            self.residual = MLPResidual()
            self.alpha = ScalarGate()
        if method in ("S2", "S3", "S4"):
            self.random = nn.Linear(8, 32)


def args():
    return SimpleNamespace(epochs=30, warmup_epochs=.1, min_lr_ratio=.01)


@pytest.mark.parametrize("method", ["S0", "S1", "S2", "S3", "S4"])
def test_all_final_groups_follow_full_schedule(method):
    model = TrainingProbe(method)
    optimizer = build_student_optimizer(model)
    prepare_student_precision_groups(model, optimizer)
    scheduler = build_student_scheduler(optimizer, args(), steps_per_epoch=100)
    assert len(optimizer.param_groups) == (2 if method == "S0" else 4)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    for step in range(3001):
        assert_student_scheduler_groups(scheduler)
        factor = ((step + 1) / 10 if step < 10 else
                  .01 + .99 * .5 * (1 + math.cos(math.pi * (step - 10) / 2990)))
        for group in optimizer.param_groups:
            assert group["lr"] == pytest.approx(1e-4 * factor, rel=1e-12)
            if group["name"].startswith("no_decay"):
                assert group["weight_decay"] == 0
        if step < 3000:
            optimizer.step()
            scheduler.step()


def test_old_scheduler_before_dtype_grouping_is_rejected():
    model = TrainingProbe("S1")
    optimizer = build_student_optimizer(model)
    scheduler = build_student_scheduler(optimizer, args(), steps_per_epoch=100)
    prepare_student_precision_groups(model, optimizer)
    with pytest.raises(RuntimeError, match="parameter-group mismatch"):
        assert_student_scheduler_groups(scheduler)


def test_stuck_group_learning_rate_is_rejected():
    model = TrainingProbe("S1")
    optimizer = build_student_optimizer(model)
    prepare_student_precision_groups(model, optimizer)
    scheduler = build_student_scheduler(optimizer, args(), steps_per_epoch=100)
    optimizer.param_groups[-1]["lr"] = 1e-4
    with pytest.raises(RuntimeError, match="learning rate mismatch"):
        assert_student_scheduler_groups(scheduler)
