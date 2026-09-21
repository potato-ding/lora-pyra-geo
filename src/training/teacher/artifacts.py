"""Self-contained formal Teacher checkpoints; historical sidecars stay legacy."""
from pathlib import Path
import torch
from src.evaluation.precision_contract import selection_signature, flat_selection_metrics, VERSION
from src.training.teacher.certified_selection import selection_metadata

SCHEMA = 'TEACHER_BEST_MODEL_V2'
FORMAL_EXPERIMENTS = {'T0-INFONCE-R224', 'T0-INFONCE-R384', 'T0-INFONCE-R448'}

def is_formal_teacher(args):
    return getattr(args, 'experiment_id', None) in FORMAL_EXPERIMENTS

def validate_training_artifacts(directory, *, require_best=True):
    names = {p.name for p in Path(directory).iterdir()}
    unexpected = names - {'best_model.pth', 'train.log'}
    if unexpected:
        raise RuntimeError('Teacher training artifact contract: unexpected ' + repr(sorted(unexpected)))
    if require_best and not {'best_model.pth', 'train.log'} <= names:
        raise RuntimeError('Teacher training artifact contract: missing best_model.pth or train.log')

def save_best_checkpoint(model, args, metrics, directory, training_world_size):
    """Called only after the unchanged strict-greater-than best decision."""
    signature = selection_signature(model, 'teacher', args.img_size)
    refs = flat_selection_metrics(metrics)
    score = refs['D2S']['R@1'] + refs['S2D']['R@1']
    refs['R1_sum'] = score
    protocol = selection_metadata(args.img_size)
    metadata = dict(protocol, experiment_id=args.experiment_id, best_epoch=metrics['epoch'],
                    best_score=score, training_world_size=int(training_world_size),
                    selection_metrics=refs)
    payload = dict(artifact_schema=SCHEMA, model={n:t.detach().cpu() for n,t in model.state_dict().items()},
                   precision_signature=signature, selection_metrics=refs,
                   selection_protocol=protocol, metadata=metadata,
                   hyperparameters=dict(vars(args)))
    torch.save(payload, Path(directory)/'best_model.pth')
    return metadata

def checkpoint_metadata(payload):
    if not isinstance(payload, dict) or payload.get('artifact_schema') != SCHEMA:
        return None
    required = {'experiment_id','image_size','best_epoch','best_score','selection_mode',
                'selection_world_size','selection_rank','training_world_size','eval_batch_size',
                'precision_contract','selection_metrics'}
    metadata = payload.get('metadata', {})
    if not required <= metadata.keys() or not isinstance(payload.get('hyperparameters'),dict):
        raise ValueError('Incomplete formal Teacher checkpoint metadata')
    expected = selection_metadata(metadata['image_size'])
    if any(metadata[k] != v for k,v in expected.items()):
        raise ValueError('Invalid formal Teacher selection protocol')
    refs = metadata['selection_metrics']
    flat = flat_selection_metrics(refs)
    score = flat['D2S']['R@1'] + flat['S2D']['R@1']
    if refs.get('R1_sum') != score or metadata['best_score'] != score:
        raise ValueError('Invalid formal Teacher best score')
    if payload.get('selection_metrics') != refs or payload.get('selection_protocol') != expected:
        raise ValueError('Conflicting Teacher checkpoint metadata')
    if payload.get('precision_signature',{}).get('image_size') != metadata['image_size']:
        raise ValueError('Teacher checkpoint image size mismatch')
    return metadata
