import torch
import torch.nn as nn
import torch.nn.functional as F


class U1652ResnetBottleBlock(nn.Module):
	"""
	残差瓶颈层（ResNet Block 风格）
	"""
	def __init__(self, in_dim=4096, num_classes=701, expansion=4, dtype=None):
		super().__init__()
		bottleneck_dim = in_dim // expansion
		self.conv1 = nn.Linear(in_dim, bottleneck_dim)
		self.conv2 = nn.Linear(bottleneck_dim, bottleneck_dim)
		self.conv3 = nn.Linear(bottleneck_dim, in_dim)
		self.relu = nn.ReLU(inplace=True)
		self.fc = nn.Linear(in_dim, num_classes)

	def forward(self, x):
		residual = x
		out = self.relu(self.conv1(x))
		out = self.relu(self.conv2(out))
		out = self.conv3(out)
		out += residual  # 残差连接
		out = self.fc(out)  # 确保输出是float32
		return out.float()

class U1652TransBottleBlock(nn.Module):
	"""
	基于注意力的瓶颈层分类头
	"""
	def __init__(self, in_dim=4096, num_classes=701, bottleneck_dim=512, num_heads=8, dtype=None):
		super().__init__()
		# 1. 线性降维到bottleneck
		self.proj = nn.Linear(in_dim, bottleneck_dim)
		# 2. 多头自注意力
		self.attn = nn.MultiheadAttention(embed_dim=bottleneck_dim, num_heads=num_heads, batch_first=True)
		self.bn = nn.BatchNorm1d(bottleneck_dim)
		self.relu = nn.ReLU(inplace=True)
		# 3. 分类层
		self.fc = nn.Linear(bottleneck_dim, num_classes)
		# 强制float32

	def forward(self, x):
		# x: (batch, in_dim)
		x = self.proj(x)  # (batch, bottleneck_dim)
		# 变成(batch, seq_len=1, bottleneck_dim)以适配MultiheadAttention
		x = x.unsqueeze(1)
		attn_out, _ = self.attn(x, x, x)  # (batch, 1, bottleneck_dim)
		x = attn_out.squeeze(1)  # (batch, bottleneck_dim)
		x = self.bn(x)
		x = self.relu(x)
		x = self.fc(x)
		return x.float()

class U1652NormalBottleBlock(nn.Module):
	"""
	归一化增强的瓶颈层分类头
	"""
	def __init__(self, in_dim=4096, num_classes=701, bottleneck_dim=512, dropout=0.2, dtype=None):
		super().__init__()
		self.bottleneck = nn.Linear(in_dim, bottleneck_dim)
		self.norm = nn.LayerNorm(bottleneck_dim)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU(inplace=True)
		self.fc = nn.Linear(bottleneck_dim, num_classes)

	def forward(self, x):
		x = self.bottleneck(x)
		x = self.norm(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc(x)
		return x.float()

# 准备用作baseline的简单瓶颈分类头
class U1652ClassifierHead(nn.Module):
	def __init__(self, in_dim=4096, num_classes=701, dtype=None):
		super().__init__()
		# 先降一半维度，再ReLU，再降到num_classes
		half_dim = in_dim // 2
		self.fc1 = nn.Linear(in_dim, half_dim)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Linear(half_dim, num_classes)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x.float()

