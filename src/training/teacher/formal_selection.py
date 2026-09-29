"""Four-GPU Teacher's rank-zero U1652 selection at batch size sixteen."""
import torch
from torch.utils.data import DataLoader
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
from src.evaluation.model_loader import EvaluationEncoder
from src.evaluation.precision_contract import VERSION
from src.evaluation.u1652_canonical import single_process_evaluation, getdist_1652_val_and_get_recall
from .formal_precision import formal_selection_signature

SELECTION_BATCH = 16


def selection_metadata(image_size, batch_size=SELECTION_BATCH):
    if image_size not in (224,256) or batch_size != SELECTION_BATCH:
        raise ValueError('Four-GPU Teacher selection protocol changed')
    return dict(selection_rank=0,selection_world_size=1,selection_mode='SINGLE_GPU_CANONICAL',
                image_size=int(image_size),eval_batch_size=batch_size,precision_contract=VERSION)


def best_selection_update(results, previous_score, previous_epoch, epoch):
    score=results['D2S']['R@1']+results['S2D']['R@1']
    update=previous_epoch is None or score>previous_score
    return dict(score=score,best_score=score if update else previous_score,
                best_epoch=epoch if update else previous_epoch,best_update=update)


def _batch_sixteen(loader):
    return DataLoader(loader.dataset,batch_size=SELECTION_BATCH,
                      shuffle=False,drop_last=False,num_workers=loader.num_workers,
                      pin_memory=loader.pin_memory,collate_fn=loader.collate_fn)


@torch.no_grad()
def certified_teacher_selection(model, *, image_size, device,
                                data_dir='data/U1652', num_workers=4,
                                batch_size=SELECTION_BATCH, loaders=None):
    selection_metadata(image_size,batch_size)
    teacher=model.module if hasattr(model,'module') else model
    formal_selection_signature(teacher,'teacher',image_size)
    training=teacher.training
    try:
        with single_process_evaluation():
            if loaders is None:
                loaders=build_1652_val_dataloaders(data_dir=data_dir,
                    img_size=[image_size,image_size],batch_size=batch_size,
                    num_workers=num_workers,distributed=False)
            encoder=EvaluationEncoder(teacher,4096).eval()
            results={}
            for direction in ('D2S','S2D'):
                pair=tuple(_batch_sixteen(loader) for loader in loaders[direction])
                values=getdist_1652_val_and_get_recall(encoder,*pair,device,task_name=direction,
                                                        precomputed_features=None)
                results[direction]=dict(zip(('R@1','R@5','R@10','AP'),values))
            return results
    finally:
        teacher.train(training)
