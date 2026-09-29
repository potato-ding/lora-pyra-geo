"""Teacher V3 checkpoints for rank-zero U1652 selection at batch sixteen."""
from pathlib import Path
import hashlib
import json
import torch
from src.evaluation.precision_contract import flat_selection_metrics
from src.training.teacher.formal_precision import formal_selection_signature
from src.training.teacher.formal_selection import selection_metadata

SCHEMA = 'TEACHER_BEST_MODEL_V3'
FORMAL_EXPERIMENTS = {'T0-INFONCE-R224', 'T0-INFONCE-R256'}

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
    if Path(directory).resolve() != Path(args.output_dir).resolve():
        raise ValueError('Teacher output_dir/checkpoint destination mismatch')
    signature = formal_selection_signature(model, 'teacher', args.img_size)
    refs = flat_selection_metrics(metrics)
    score = refs['D2S']['R@1'] + refs['S2D']['R@1']
    refs['R1_sum'] = score
    protocol = selection_metadata(args.img_size,args.val_batch_size)
    metadata = dict(protocol, experiment_id=args.experiment_id, best_epoch=metrics['epoch'],
                    best_score=score, training_world_size=int(training_world_size),
                    selection_metrics=refs)
    formal_config = json.loads(Path(args.config).read_text()) if getattr(args, 'config', None) else None
    if formal_config is not None:
        encoded=json.dumps(formal_config,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        metadata['config_sha256']=hashlib.sha256(encoded).hexdigest()
    payload = dict(artifact_schema=SCHEMA, model={n:t.detach().cpu() for n,t in model.state_dict().items()},
                   precision_signature=signature, selection_metrics=refs,
                   selection_protocol=protocol, metadata=metadata,
                   hyperparameters=dict(vars(args)))
    if formal_config is not None:payload['config']=formal_config
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
    expected = selection_metadata(metadata['image_size'],16)
    if any(metadata[k] != v for k,v in expected.items()):
        raise ValueError('Invalid formal Teacher selection protocol')
    refs = metadata['selection_metrics']
    flat = flat_selection_metrics(refs)
    score = flat['D2S']['R@1'] + flat['S2D']['R@1']
    if refs.get('R1_sum') != score or metadata['best_score'] != score:
        raise ValueError('Invalid formal Teacher best score')
    if payload.get('selection_metrics') != refs or payload.get('selection_protocol') != expected:
        raise ValueError('Conflicting Teacher checkpoint metadata')
    signature=payload.get('precision_signature',{})
    if signature.get('image_size') != metadata['image_size'] or signature.get('selection_batch_size') != 16:
        raise ValueError('Teacher checkpoint geometry/selection batch mismatch')
    config=payload.get('config')
    if not isinstance(config,dict) or (config.get('val_batch_size') != 16
            or config.get('img_size') != metadata['image_size']
            or config.get('output_dir') != payload['hyperparameters'].get('output_dir')
            or metadata.get('training_world_size') != 4
            or not 6 <= metadata.get('best_epoch',0) <= config.get('epochs',0)):
        raise ValueError('Teacher run/config/selection identity mismatch')
    if 'config_sha256' in metadata:
        config=payload.get('config')
        if not isinstance(config,dict):raise ValueError('Teacher config identity missing')
        encoded=json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        if hashlib.sha256(encoded).hexdigest()!=metadata['config_sha256']:
            raise ValueError('Teacher config SHA mismatch')
    return metadata
