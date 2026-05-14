import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenMergingBipartite:
    """
    ToMe（二分图匹配）Token合并逻辑。
    支持基于相似度的二分图匹配与Token平均池化。
    """
    def __init__(self, merge_ratio=0.5):
        self.merge_ratio = merge_ratio  # 合并比例（0~1），如0.5表示合并一半token

    def bipartite_matching(self, sim_matrix):
        """
        输入: sim_matrix (N, M) 表示N个token与M个token的相似度
        输出: 匹配对列表 [(i, j), ...]
        """
        # 贪心最大匹配（可替换为匈牙利算法等）
        N, M = sim_matrix.shape
        matched = set()
        pairs = []
        sim_flat = sim_matrix.flatten()
        idx_sorted = torch.argsort(sim_flat, descending=True)
        for idx in idx_sorted:
            i = idx // M
            j = idx % M
            if i not in matched and j not in matched:
                pairs.append((i.item(), j.item()))
                matched.add(i)
                matched.add(j)
            if len(pairs) >= int(self.merge_ratio * min(N, M)):
                break
        return pairs

    def merge_tokens(self, tokens, pairs):
        """
        tokens: (B, N, C)
        pairs: [(i, j), ...]
        返回合并后的tokens (B, N', C)
        """
        B, N, C = tokens.shape
        mask = torch.ones(N, dtype=torch.bool, device=tokens.device)
        merged_tokens = []
        for i, j in pairs:
            merged = (tokens[:, i, :] + tokens[:, j, :]) / 2
            merged_tokens.append(merged.unsqueeze(1))
            mask[i] = False
            mask[j] = False
        # 保留未合并的token
        remain_tokens = tokens[:, mask, :]
        if merged_tokens:
            merged_tokens = torch.cat(merged_tokens, dim=1)
            out = torch.cat([remain_tokens, merged_tokens], dim=1)
        else:
            out = remain_tokens
        return out

    def __call__(self, tokens):
        # tokens: (B, N, C)
        # 计算token间相似度
        B, N, C = tokens.shape
        sim_matrix = torch.matmul(tokens[0], tokens[0].T)  # (N, N)
        pairs = self.bipartite_matching(sim_matrix)
        return self.merge_tokens(tokens, pairs)
