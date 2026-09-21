"""One single-process Teacher U1652 path for live selection and reload."""
from torch.utils.data import DataLoader
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
from src.utils.train_eval_utils import single_process_evaluation, getdist_1652_val_and_get_recall

CANONICAL_TEACHER_EVAL_BATCH = 8

def canonical_loader(loader):
    return DataLoader(loader.dataset, batch_size=CANONICAL_TEACHER_EVAL_BATCH,
                      shuffle=False, drop_last=False, num_workers=loader.num_workers,
                      pin_memory=loader.pin_memory, collate_fn=loader.collate_fn)

def evaluate_u1652_single_gpu_canonical(model, *, image_size, device,
                                       data_dir='data/U1652', num_workers=4,
                                       loaders=None, cached_features=None):
    if image_size <= 0:
        raise ValueError('Explicit positive Teacher image_size required')
    # Context-local suppression applies only to evaluation helpers, never to the
    # training process group or DeepSpeed. No library functions are monkeypatched.
    with single_process_evaluation():
        if loaders is None:
            loaders = build_1652_val_dataloaders(data_dir=data_dir,
                img_size=[image_size, image_size], batch_size=8,
                num_workers=num_workers, distributed=False)
        results = {}
        for direction in ('D2S', 'S2D'):
            pair = tuple(canonical_loader(x) for x in loaders[direction])
            values = getdist_1652_val_and_get_recall(model, *pair, device,
                task_name=direction,
                precomputed_features=None if cached_features is None else cached_features[direction])
            results[direction] = dict(zip(('R@1','R@5','R@10','AP'), values))
        return results
