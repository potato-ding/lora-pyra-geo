"""Shared complete single-GPU Middle selection/reload evaluation (batch 32)."""
from torch.utils.data import DataLoader
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
from src.utils.train_eval_utils import single_process_evaluation, getdist_1652_val_and_get_recall

CANONICAL_MIDDLE_EVAL_BATCH = 32

def evaluate_middle_u1652_canonical(model, *, image_size, device, data_dir='data/U1652', num_workers=4, loaders=None):
    if image_size != 224:
        raise ValueError('Formal Middle requires image_size=224')
    with single_process_evaluation():
        if loaders is None:
            loaders = build_1652_val_dataloaders(data_dir=data_dir, img_size=[image_size]*2,
                batch_size=CANONICAL_MIDDLE_EVAL_BATCH, num_workers=num_workers, distributed=False)
        results = {}
        for direction in ('D2S','S2D'):
            pair = tuple(DataLoader(x.dataset,batch_size=CANONICAL_MIDDLE_EVAL_BATCH,
                shuffle=False,drop_last=False,num_workers=x.num_workers,
                pin_memory=x.pin_memory,collate_fn=x.collate_fn) for x in loaders[direction])
            values = getdist_1652_val_and_get_recall(model,*pair,device,task_name=direction)
            results[direction] = dict(zip(('R@1','R@5','R@10','AP'),values))
        return results
