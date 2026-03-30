import torch
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        # 把属于当前卡的梯度切出来传回去
        return all_gradients[dist.get_rank()]

@torch.no_grad()
def concat_all_gather(tensor):
    """专门用来跨卡搬运标签和视角（无梯度，极度安全，防止整数报错）"""
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)

def gather_features_and_labels_and_views(feats, labels, views):
    """
    将所有显卡上的特征、标签汇总，并保留跨卡满血梯度！
    """
    if not dist.is_initialized():
        return feats, labels, views
    
    # 1. 特征(feats)：带小数点，需要算梯度，走带 backward 的魔法通道
    all_feats = torch.cat(GatherLayer.apply(feats), dim=0)
    
    # 2. 标签和视角(labels, views)：纯整数，不需要梯度，走普通安全通道
    all_labels = concat_all_gather(labels)
    all_views = concat_all_gather(views)
    
    return all_feats, all_labels, all_views