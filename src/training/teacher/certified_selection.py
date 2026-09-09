"""Teacher selection wrapper matching the R224 certified final inference path.

Fixed groups of eight images preserve BF16 kernel batch composition across
distributed and single-GPU evaluation. Metric mathematics remain shared.
"""
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.evaluation.model_loader import EvaluationEncoder
from src.evaluation.metrics import getdist_1652_val_and_get_recall


class CanonicalSelectionEncoder(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        # Shared extraction chooses its input dtype from the first parameter.
        # The final loader has FP32 logit_scale; DeepSpeed stores it in BF16.
        # This nontrainable wrapper-only anchor preserves the final FP32 input.
        self.input_dtype_anchor = nn.Parameter(torch.zeros((), dtype=torch.float32,
            device=next(teacher.parameters()).device), requires_grad=False)
        self.encoder = EvaluationEncoder(teacher, 4096)

    def forward(self, images):
        return self.encoder(images)


def canonical_batch_groups(size, rank, world_size, batch_size=8):
    groups = [list(range(i, min(i + batch_size, size))) for i in range(0, size, batch_size)]
    if not groups:
        raise ValueError('Empty certified evaluation dataset')
    count = math.ceil(len(groups) / world_size) * world_size
    groups = groups + [groups[-1]] * (count - len(groups))
    return groups[rank::world_size]


def canonical_loader(loader):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world = torch.distributed.get_world_size() if distributed else 1
    return DataLoader(loader.dataset,
        batch_sampler=canonical_batch_groups(len(loader.dataset), rank, world),
        num_workers=loader.num_workers, pin_memory=loader.pin_memory,
        collate_fn=loader.collate_fn)


@torch.no_grad()
def certified_teacher_selection(model, query_loader, gallery_loader, device, task_name=None):
    teacher = model.module if hasattr(model, 'module') else model
    training = teacher.training
    try:
        encoder = CanonicalSelectionEncoder(teacher).eval()
        return getdist_1652_val_and_get_recall(encoder,
            canonical_loader(query_loader), canonical_loader(gallery_loader),
            device, task_name=task_name)
    finally:
        teacher.train(training)
