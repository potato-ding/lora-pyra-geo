import torch
import torch.nn as nn
from pathlib import Path
import sys, os
import inspect
from torch.nn.attention import sdpa_kernel, SDPBackend

from src.utils.rank_logging import rank0_print


def _safe_torch_load(path, map_location):
	load_kwargs = {"map_location": map_location}
	if "weights_only" in inspect.signature(torch.load).parameters:
		load_kwargs["weights_only"] = True
	return torch.load(path, **load_kwargs)


class DINOv3Backbone(nn.Module):
	"""
	DINOv3-7B backbone loader.

	Loads the local dinov3-main implementation and official checkpoint.
	"""
	def __init__(self, repo_dir: str, ckpt_path: str, device: str = 'cuda', dtype: str = 'bfloat16'):
		super().__init__()
		# DINOv3 code and weights live under src/models by default.
		self.repo_dir = Path(repo_dir) if repo_dir else Path(__file__).parent / 'dinov3_main'
		model_root = Path(__file__).resolve().parents[1]
		self.repo_dir = Path(repo_dir) if repo_dir else model_root
		self.ckpt_path = Path(ckpt_path) if ckpt_path else model_root / 'dinov3-pth/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'
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
			
		# Import the local DINOv3 model definition.
		from dinov3_main.dinov3.hub.backbones import dinov3_vit7b16
		# Use pretrained=False because weights are loaded from the local checkpoint.
		model = dinov3_vit7b16(pretrained=False)
		rank0_print(f"[DINOv3Base] loading checkpoint: {self.ckpt_path}")
		checkpoint = _safe_torch_load(self.ckpt_path, map_location='cpu')
		# Extract the actual state dict from common checkpoint wrappers.
		if 'model' in checkpoint:
			state_dict = checkpoint['model']
		elif 'teacher' in checkpoint:
			state_dict = checkpoint['teacher']
		else:
			state_dict = checkpoint
			
		msg = model.load_state_dict(state_dict, strict=True)
		# Strict loading catches any mismatch between the model and checkpoint.
		
		if self.device.type == "cuda":
			model = model.to(device=self.device, dtype=torch.bfloat16)
		else:
			model = model.to(device=self.device)
			
		model.eval()
		for p in model.parameters():
			p.requires_grad_(False)
			
		return model

	def forward(self, x):
		# Input should already be on self.device with a compatible dtype.
		with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
			return self.model(x)
