import torch
import torch.nn as nn

from src.models.teacher.peft_lora import LoRALayer
from src.utils.teacher_experiment_audit import audit_teacher_runtime_structure


class FakeDinoModel(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for block_idx in range(40):
            base = nn.Linear(4, 4)
            if block_idx < 20:
                for param in base.parameters():
                    param.requires_grad = False
                block = nn.Sequential(base)
            elif block_idx < 36:
                for param in base.parameters():
                    param.requires_grad = False
                block = nn.Sequential(LoRALayer(base, r=2, alpha=4, dropout=0.0))
            else:
                block = nn.Sequential(base)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeDinoModel()


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.logit_scale = nn.Parameter(torch.tensor(1.0))


def test_runtime_structure_audit_classifies_real_requires_grad_state():
    model = FakeTeacher()
    lines = []

    report = audit_teacher_runtime_structure(model, print_fn=lines.append)

    assert report["valid"] is True
    assert report["num_blocks"] == 40
    assert [item["mode"] for item in report["blocks"][:20]] == ["frozen"] * 20
    assert [item["mode"] for item in report["blocks"][20:36]] == ["LoRA"] * 16
    assert [item["mode"] for item in report["blocks"][36:]] == ["full_finetune"] * 4
    assert len(report["lora_module_names"]) == 16
    assert report["parameters"]["logit_scale_trainable_params"] == 1
    assert report["parameters"]["head_descriptor_trainable_params"] == 0
    assert any("Block 20 | mode=LoRA" in line for line in lines)
    assert any("teacher_structure_matches_T0_3090=True" in line for line in lines)


def test_runtime_structure_audit_reports_mismatch_without_repairing_model():
    model = FakeTeacher()
    first_block_param = next(model.backbone.model.blocks[0].parameters())
    first_block_param.requires_grad = True
    lines = []

    report = audit_teacher_runtime_structure(model, print_fn=lines.append)

    assert report["valid"] is False
    assert first_block_param.requires_grad is True
    assert any("Block 00" in error for error in report["errors"])
    assert any("[EXPERIMENT_AUDIT][ERROR]" in line for line in lines)
