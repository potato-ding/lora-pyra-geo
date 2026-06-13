import torch
import torch.nn as nn

class PYRAModule(nn.Module):
    def __init__(self, dim, sigma=0.02):
        super().__init__()
        self.dim = dim
        
        # 直接在初始化时指定 dtype=torch.bfloat16
        self.W_r = nn.Parameter(
            torch.randn(dim, 1, dtype=torch.bfloat16) * sigma
        )
        self.W_D = nn.Parameter(
            torch.zeros(1, dim, dtype=torch.bfloat16)
        )
        
        # LayerNorm 绝对不指定 bfloat16，让它保持默认的 float32 以确保数值稳定
        self.ln = nn.LayerNorm(dim)

    def forward(self, M_s, M_t):
        # M_s 和 M_t 进来大概率已经是 bfloat16 了
        
        # LayerNorm 内部会自动把输入转为 fp32 算方差，再安全转回 bf16 输出 (Autocast 特性)
        M_info = self.ln(M_s + M_t) 
        
        # 此时大家都是 bf16，直接无缝丝滑相乘，没有任何额外开销
        delta_D = torch.matmul(M_info, self.W_r) 
        
        W_D_expand = self.W_D.expand(M_info.size(0), -1, -1)
        delta_r = torch.matmul(W_D_expand, M_info.transpose(1, 2)) 

        delta_D_broadcast = delta_D.expand(-1, -1, self.dim)
        delta_r_broadcast = delta_r.transpose(1, 2).expand(-1, -1, self.dim)

        # PyTorch 的 sigmoid 和乘法运算会自动处理这些同精度张量
        hat_M_s = 2 * torch.sigmoid(delta_D_broadcast) * M_s
        M_s_mod = M_s + (2 * torch.sigmoid(delta_r_broadcast) - 1) * hat_M_s
        
        return M_s_mod