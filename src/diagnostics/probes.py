"""Diagnostic-only frozen probe modules (never used by StudentModel)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenStudentProbe(nn.Module):
    def __init__(self, student, probe_type):
        super().__init__()
        self.student = student
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)
        self.student.eval()
        self.probe_type = probe_type
        if probe_type == "P1":
            self.feature_key = "f3_gap"
            self.probe = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512))
        elif probe_type == "P2":
            self.feature_key = "f4_gap"
            self.probe = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512))
        elif probe_type == "P3":
            self.feature_key = "f4_gap"
            self.probe = nn.Sequential(
                nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512), nn.BatchNorm1d(512)
            )
        else:
            raise ValueError(probe_type)

    @property
    def backbone(self):
        return self.student.backbone

    def train(self, mode=True):
        super().train(mode)
        self.student.eval()
        self.probe.train(mode)
        return self

    def forward(self, images):
        self.student.eval()
        with torch.no_grad():
            features = self.student(images, return_audit_features=True)[self.feature_key]
        probe_dtype = next(self.probe.parameters()).dtype
        return F.normalize(self.probe(features.to(dtype=probe_dtype)), dim=1)


def parameter_audit(model):
    backbone = model.student
    return {
        "backbone_total_params": sum(p.numel() for p in backbone.parameters()),
        "backbone_trainable_params": sum(p.numel() for p in backbone.parameters() if p.requires_grad),
        "probe_total_params": sum(p.numel() for p in model.probe.parameters()),
        "probe_trainable_params": sum(p.numel() for p in model.probe.parameters() if p.requires_grad),
        "frozen_backbone_eval_mode": not backbone.training,
        "frozen_backbone_bn_buffer_policy": "student.eval() on every train()/forward; no_grad feature extraction",
    }
