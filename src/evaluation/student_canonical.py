"""Shared complete single-GPU Student selection/reload evaluation."""
from torch.utils.data import DataLoader
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
from src.utils.train_eval_utils import single_process_evaluation, getdist_1652_val_and_get_recall

CANONICAL_STUDENT_EVAL_BATCH = 16

def evaluate_student_u1652_canonical(model, *, image_size, device, data_dir='data/U1652', num_workers=4, loaders=None, batch_size=CANONICAL_STUDENT_EVAL_BATCH):
    if type(image_size) is not int or image_size not in (224,256):
        raise ValueError('Formal Student requires image_size in (224,256)')
    if batch_size != 16:
        raise ValueError('Formal Student U1652 evaluation requires batch 16')
    with single_process_evaluation():
        if loaders is None:
            loaders = build_1652_val_dataloaders(data_dir=data_dir, img_size=[image_size]*2,
                batch_size=batch_size, num_workers=num_workers, distributed=False)
        results = {}
        for direction in ('D2S','S2D'):
            pair = tuple(DataLoader(x.dataset,batch_size=batch_size,
                shuffle=False,drop_last=False,num_workers=x.num_workers,
                pin_memory=x.pin_memory,collate_fn=x.collate_fn) for x in loaders[direction])
            values = getdist_1652_val_and_get_recall(model,*pair,device,task_name=direction)
            results[direction] = dict(zip(('R@1','R@5','R@10','AP'),values))
        return results
