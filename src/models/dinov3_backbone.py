import torch
import torch.nn as nn
from pathlib import Path
import sys, os
from torch.nn.attention import sdpa_kernel, SDPBackend
class DINOv3Backbone(nn.Module):
	"""
	DINOv3-7B 主干网络加载与推理基类。
	支持加载官方 dinov3-main 仓库权重，并可设置为 bfloat16 推理。
	"""
	def __init__(self, repo_dir: str, ckpt_path: str, device: str = 'cuda', dtype: str = 'bfloat16'):
		super().__init__()
		# 现在 dinov3 代码和权重都在 src/models/DINOV3 下
		self.repo_dir = Path(repo_dir) if repo_dir else Path(__file__).parent / 'dinov3_main'
		self.ckpt_path = Path(ckpt_path) if ckpt_path else self.repo_dir / 'dinov3-pth/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'
		self.device = torch.device(device)
		self.dtype = dtype
		self.model = self._load_model()
 
	def _load_model(self):
		dinov3_main_dir = os.path.join(str(self.repo_dir), 'dinov3_main')
		dinov3_main_parent = os.path.dirname(dinov3_main_dir)
		
		if dinov3_main_parent not in sys.path:
			sys.path.insert(0, dinov3_main_parent)
		if dinov3_main_dir not in sys.path:
			sys.path.insert(0, dinov3_main_dir)
			
		# 把定义模型的函数导进来
		from dinov3_main.dinov3.hub.backbones import dinov3_vit7b16
		model = dinov3_vit7b16(pretrained=True)
		checkpoint = torch.load(self.ckpt_path, map_location='cpu')
		# 扒出真正的权重字典
		if 'model' in checkpoint:
			state_dict = checkpoint['model']
		elif 'teacher' in checkpoint:
			state_dict = checkpoint['teacher']
		else:
			state_dict = checkpoint
			
		msg = model.load_state_dict(state_dict, strict=True)
		# 先改成 strict=False，让它能塞多少塞多少，并把没塞进去的清单返回
		
		if self.device.type == "cuda":
			model = model.to(device=self.device, dtype=torch.bfloat16)
		else:
			model = model.to(device=self.device)
			
		model.eval()
		for p in model.parameters():
			p.requires_grad_(False)
			
		return model

	def forward(self, x):
		# 输入需为已转到 self.device 且 dtype 匹配的 tensor
		with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
			return self.model(x)
