import random
import numpy as np
import torch
import torch.distributed as dist
def _rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def _world():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _seed_stst_worker(worker_id):
    """Make paired TOP/RANDOM augmentation streams reproducible per worker."""
    info = torch.utils.data.get_worker_info()
    worker_seed = int(torch.initial_seed() % 2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    dataset = info.dataset
    if hasattr(dataset.sat_transforms, 'set_random_seed'):
        dataset.sat_transforms.set_random_seed(worker_seed + 17)
    if hasattr(dataset.drone_transforms, 'set_random_seed'):
        dataset.drone_transforms.set_random_seed(worker_seed + 31)

def _gather_grad(tensor):
    if _world() == 1:
        return tensor
    from src.utils.gather_features_and_labels_and_views import GatherLayer
    return torch.cat(GatherLayer.apply(tensor), dim=0)
