"""Single-GPU Student training wrapper and DeepSpeed BF16 protocol."""
from torch import nn


class StudentTrainingModel(nn.Module):
    def __init__(self, student, supervision=None):
        super().__init__()
        self.student = student
        self.supervision = supervision

    def forward(self, images):
        return self.student(images)


def deepspeed_config(cfg):
    batch = cfg["batch_size"]
    world = cfg["world_size"]
    accumulation = cfg["grad_accum_steps"]
    if world != 1 or accumulation != 1 or cfg["precision"] != "bfloat16":
        raise ValueError("Formal Student requires one GPU, no accumulation and BF16")
    return {
        "train_batch_size": batch,
        "train_micro_batch_size_per_gpu": batch,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 1},
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "gradient_clipping": 0.0,
        "steps_per_print": 1000000,
    }
