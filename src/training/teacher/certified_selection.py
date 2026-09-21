"""Live rank0 selection preserves Teacher precision and uses formal evaluation."""
import torch
from src.evaluation.model_loader import EvaluationEncoder
from src.evaluation.precision_contract import selection_signature, VERSION
from src.evaluation.u1652_canonical import canonical_loader, evaluate_u1652_single_gpu_canonical

def selection_metadata(image_size):
    return dict(selection_rank=0, selection_world_size=1, selection_mode='SINGLE_GPU_CANONICAL',
                image_size=int(image_size), eval_batch_size=8, precision_contract=VERSION)

def best_selection_update(results, previous_score, previous_epoch, epoch):
    score = results['D2S']['R@1'] + results['S2D']['R@1']
    update = previous_epoch is None or score > previous_score
    return dict(score=score, best_score=score if update else previous_score,
                best_epoch=epoch if update else previous_epoch, best_update=update)

@torch.no_grad()
def certified_teacher_selection(model, *, image_size, device, data_dir='data/U1652',
                                num_workers=4, loaders=None):
    teacher = model.module if hasattr(model, 'module') else model
    selection_signature(teacher, 'teacher', image_size)
    training = teacher.training
    try:
        return evaluate_u1652_single_gpu_canonical(EvaluationEncoder(teacher,4096).eval(),
            image_size=image_size, device=device, data_dir=data_dir,
            num_workers=num_workers, loaders=loaders)
    finally:
        teacher.train(training)
