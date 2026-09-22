"""Rank0-only selection and best-state synchronization for the non-SAM core."""
import json
import torch
from src.evaluation.middle_canonical import evaluate_middle_u1652_canonical, CANONICAL_MIDDLE_EVAL_BATCH
from src.evaluation.model_loader import EvaluationEncoder
from src.evaluation.precision_contract import selection_signature, VERSION
from src.utils.selection_sync import run_rank0_selection

def selection_metadata(image_size=224):
    return dict(selection_mode='SINGLE_GPU_CANONICAL',selection_world_size=1,
        selection_rank=0,image_size=image_size,eval_batch_size=CANONICAL_MIDDLE_EVAL_BATCH,
        precision_contract=VERSION)

@torch.no_grad()
def select_and_save(engine, controller, config, epoch, step, device):
    def action():
        model=engine.module
        size=config['data']['input_size']
        selection_signature(model,'middle',size)
        print('MIDDLE_SELECTION='+json.dumps(dict(selection_metadata(size),training_world_size=config['data']['world_size'])),flush=True)
        results=evaluate_middle_u1652_canonical(EvaluationEncoder(model,768).eval(),
            image_size=size,device=device,data_dir=config['data'].get('val_dir','data/U1652'),
            num_workers=config['data']['num_workers'])
        metrics={d+'_'+k:v for d,values in results.items() for k,v in
            (('R1',values['R@1']),('R5',values['R@5']),('AP',values['AP']))}
        metrics['R1_sum']=metrics['D2S_R1']+metrics['S2D_R1']
        improved=controller.save_best_if_improved(engine,epoch,step,metrics)
        return dict(metrics=metrics,improved=improved,best_score=controller.best_score,
            best_epoch=controller.best_epoch,best_metrics=controller.best_metrics)
    try:
        result=run_rank0_selection(action)
        controller.best_score=result['best_score'];controller.best_epoch=result['best_epoch']
        controller.best_metrics=result['best_metrics']
        return result['metrics'],result['improved']
    finally:
        engine.train()
