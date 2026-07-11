import torch
import torch.nn as nn


class LoRAInject:
    """Inject LoRA layers into selected linear modules."""

    def __init__(
        self,
        module: nn.Module,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        target_names=("q_proj", "v_proj"),
        block_range=None,
        task_type=None,
    ):
        self.module = module
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_names = target_names
        self.lora_layers = []
        self.block_range = block_range
        self.task_type = task_type

    def inject(self):
        targets = []
        for name, submodule in self.module.named_modules():
            if not any(target_name in name for target_name in self.target_names):
                continue
            if not isinstance(submodule, nn.Linear):
                continue
            if self.block_range is not None and name.startswith("blocks."):
                try:
                    block_idx = int(name.split(".")[1])
                except (IndexError, ValueError):
                    block_idx = None
                if block_idx is not None and not (
                    self.block_range[0] <= block_idx < self.block_range[1]
                ):
                    continue
            targets.append((name, submodule))

        for name, submodule in targets:
            lora = LoRALayer(submodule, self.r, self.alpha, self.dropout)
            if self.task_type is not None:
                lora.task_type = self.task_type

            parent = self.module
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name_parts[-1], lora)
            self.lora_layers.append((name, lora))


class LoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        device = base_layer.weight.device
        self.lora_A = self.lora_A.to(device=device, dtype=torch.float32)
        self.lora_B = self.lora_B.to(device=device, dtype=torch.float32)

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

    def forward(self, x):
        base_dtype = self.base.weight.dtype
        if x.dtype != base_dtype:
            x = x.to(base_dtype)

        lora_dtype = self.lora_A.weight.dtype
        x_lora = x.to(lora_dtype)
        # Runtime evidence for T0 checks. Do not alter this compute path without
        # declaring a new experiment variable / precision contract.
        self._runtime_input_dtype = x_lora.dtype
        lora_out = self.lora_B(self.lora_A(self.dropout(x_lora))) * self.scaling
        return self.base(x) + lora_out.to(base_dtype)


__all__ = ["LoRAInject", "LoRALayer"]
