import torch
import torch.nn as nn

class LoRAInject:
	"""
	独立的 LoRA 注入工具类，可用于为任意 nn.Module 注入 LoRA 层。
	用法：
		lora = LoRAInject(target_module, r=8, alpha=16, dropout=0.05)
		lora.inject()
	"""
	def __init__(self, module: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05, target_names=("q_proj", "v_proj"), block_range=None, task_type=None):
		self.module = module
		self.r = r
		self.alpha = alpha
		self.dropout = dropout
		self.target_names = target_names
		self.lora_layers = []
		self.block_range = block_range  # (start, end) or None
		self.task_type = task_type  # e.g. "feature_extraction"

	def inject(self):
		# 先收集所有目标，再批量注入，避免遍历时修改结构
		targets = []
		for name, submodule in self.module.named_modules():
			if any(tn in name for tn in self.target_names) and isinstance(submodule, nn.Linear):
				# 支持block范围过滤（如blocks.0.attn.qkv）
				if self.block_range is not None and name.startswith("blocks."):
					try:
						block_idx = int(name.split(".")[1])
					except Exception:
						block_idx = None
					if block_idx is not None:
						if not (self.block_range[0] <= block_idx < self.block_range[1]):
							continue
				targets.append((name, submodule))
		for name, submodule in targets:
			lora = LoRALayer(submodule, self.r, self.alpha, self.dropout)
			# 注入自定义task_type属性
			if self.task_type is not None:
				lora.task_type = self.task_type
			# 只支持一级属性注入
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
		nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
		nn.init.zeros_(self.lora_B.weight)
		# LoRA权重强制float32
		device = base_layer.weight.device
		self.lora_A = self.lora_A.to(device=device, dtype=torch.float32)
		self.lora_B = self.lora_B.to(device=device, dtype=torch.float32)
		# 兼容主干直接访问in_features等属性
		self.in_features = base_layer.in_features
		self.out_features = base_layer.out_features

	def forward(self, x):
		# 保证输入与权重dtype一致，避免bfloat16/float32混用
		dtype = self.base.weight.dtype
		if x.dtype != dtype:
			x = x.to(dtype)
			
		lora_out = self.dropout(self.lora_B(self.lora_A(x))) * self.scaling
		
		return self.base(x) + lora_out.to(dtype)

class DoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        self.scaling = alpha / r
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        # ------------------- DoRA 核心修改 -------------------
        # 1. 提取初始权重幅度 (Magnitude m)
        # 计算 base_layer weight 沿 dim=1 (输入维度) 的 L2 范数
        device = base_layer.weight.device
        with torch.no_grad():
            m_init = torch.linalg.norm(base_layer.weight.data.to(torch.float32), dim=1, keepdim=True)
        # 注册 m 为可训练参数
        self.m = nn.Parameter(m_init)
        # -----------------------------------------------------

        # LORA 权重强制 float32，与原代码保持一致
        self.lora_A.to(device=device, dtype=torch.float32)
        self.lora_B.to(device=device, dtype=torch.float32)
        self.m.data = self.m.data.to(device=device, dtype=torch.float32)

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

    def forward(self, x):
        # 保证输入与权重 dtype 一致
        dtype = self.base.weight.dtype
        if x.dtype != dtype:
            x = x.to(dtype)

        # 1. 基础分支输出 (包含 base bias，如果有的话)
        base_out = self.base(x)

        # 2. LoRA 分支输出 (将其转为 float32 计算)
        # x_f32 = x.to(torch.float32)
        # lora_out = self.lora_B(self.lora_A(self.dropout(x_f32))) * self.scaling
        lora_dtype = self.lora_A.weight.dtype
        x_lora = x.to(lora_dtype)
        lora_out = self.lora_B(self.lora_A(self.dropout(x_lora))) * self.scaling

        # 3. 动态计算组合权重 (W + dW) 的方向范数
        W = self.base.weight.to(torch.float32)
        # dW = B @ A * scaling
        dW = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        W_new = W + dW
        
        # 计算新方向矩阵的 L2 范数，shape: (out_features, 1)
        norm = torch.linalg.norm(W_new, dim=1, keepdim=True)

        # 4. 计算 DoRA 缩放因子: m / ||W + dW||
        # shape 从 (out_features, 1) squeeze 成 (out_features,)，方便利用 broadcasting 应用到 batch 输出上
        scale_factor = (self.m / norm).squeeze(-1).to(dtype)

        # 5. 应用缩放
        # DoRA 的缩放仅作用于权重矩阵带来的输出。
        # base_out = x @ W^T + bias
        # 总输出应为: (x @ W^T + x @ dW^T) * scale + bias
        # 等价于: (base_out - bias + lora_out) * scale + bias
        if self.base.bias is not None:
            base_out_no_bias = base_out - self.base.bias
            out = (base_out_no_bias + lora_out.to(dtype)) * scale_factor + self.base.bias
        else:
            out = (base_out + lora_out.to(dtype)) * scale_factor

        return out

class DoRAInject:
    """
    独立的 DoRA 注入工具。
    """
    def __init__(self, module: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05, 
                 target_names=("q_proj", "v_proj"), block_range=None, task_type=None):
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
            if any(tn in name for tn in self.target_names) and isinstance(submodule, nn.Linear):
                if self.block_range is not None and name.startswith("blocks."):
                    try:
                        block_idx = int(name.split(".")[1])
                    except Exception:
                        block_idx = None
                    if block_idx is not None:
                        if not (self.block_range[0] <= block_idx < self.block_range[1]):
                            continue
                targets.append((name, submodule))

        for name, submodule in targets:
            lora = DoRALayer(submodule, self.r, self.alpha, self.dropout)
            
            if self.task_type is not None:
                lora.task_type = self.task_type
                
            parent = self.module
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name_parts[-1], lora)
            self.lora_layers.append((name, lora))
