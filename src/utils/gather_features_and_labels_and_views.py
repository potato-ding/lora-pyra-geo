"""Distributed gather helpers."""

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
        return all_gradients[dist.get_rank()]


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


def gather_features_and_labels_and_views(feats, labels, views):
    if not dist.is_available() or not dist.is_initialized():
        return feats, labels, views

    all_feats = torch.cat(GatherLayer.apply(feats), dim=0)
    all_labels = concat_all_gather(labels)
    all_views = concat_all_gather(views)
    return all_feats, all_labels, all_views


__all__ = [
    "GatherLayer",
    "concat_all_gather",
    "gather_features_and_labels_and_views",
]
