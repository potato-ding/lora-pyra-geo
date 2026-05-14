import numpy as np
import torch

def compute_mAP(index, good_index, junk_index):
    """
    这是你截图里的标准函数，100% 还原。
    用于计算单行 Query 的 AP 和 CMC 命中情况。
    """
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    
    # 1. 移除干扰项 (Junk Index)
    # mask 找出那些既不是正确项也不是干扰项的索引
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # 2. 寻找正确项 (Good Index)
    if good_index.size == 0:
        return -1, cmc # 如果没有正确项，跳过

    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()

    # 3. 计算 CMC (Recall)
    if rows_good.size > 0:
        cmc[rows_good[0]:] = 1

    # 4. 计算 AP (梯形积分面积法)
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        # 核心：使用 (新精度 + 旧精度) / 2 进行积分
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc