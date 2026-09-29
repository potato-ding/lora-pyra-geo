"""Self-contained Middle best checkpoint; no Teacher tensors or sidecar files."""
import math
import hashlib
import json
import torch
from .checkpoint import CheckpointController, portable_state, raw_model
from .distributed import rank
from .selection import selection_metadata
from src.evaluation.middle_canonical import FORMAL_MIDDLE_SELECTION_BATCH
from src.evaluation.precision_contract import selection_signature, flat_selection_metrics

SCHEMA='MIDDLE_BEST_MODEL_V2'

def checkpoint_metadata(payload):
    if payload.get('artifact_schema') != SCHEMA:
        return None
    m=payload['metadata'];config=payload['config']
    if 'public_config' in payload:
        public=payload['public_config']
        if not isinstance(public,dict):
            raise ValueError('Invalid public Middle config')
        encoded=json.dumps(public,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        if hashlib.sha256(encoded).hexdigest()!=m.get('public_config_sha256'):
            raise ValueError('Middle public config SHA mismatch')
        if public.get('experiment_id')!=config['experiment']['name'] or public.get('img_size')!=config['data']['input_size']:
            raise ValueError('Middle public/runtime config identity mismatch')
    if 'canonical_runtime_config_sha256' in m:
        from .config_identity import runtime_fingerprint
        if m['canonical_runtime_config_sha256']!=runtime_fingerprint(config):
            raise ValueError('Middle canonical runtime config SHA mismatch')
    from .core_config import validate_image_size, validate_teacher_identity
    size=validate_image_size(config['data']['input_size'])
    expected=selection_metadata(
        size,FORMAL_MIDDLE_SELECTION_BATCH if 'public_config' in payload else 32)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Invalid Middle selection protocol')
    if m['experiment_id']!=config['experiment']['name'] or m['training_world_size']!=2:
        raise ValueError('Invalid Middle experiment metadata')
    first_epoch=6 if 'public_config' in payload else 1
    if not isinstance(m['best_epoch'],int) or not first_epoch<=m['best_epoch']<=10:
        raise ValueError('Invalid best epoch')
    refs=flat_selection_metrics(m['selection_metrics'])
    if not all(math.isfinite(v) for values in refs.values() for v in values.values()):raise ValueError('Nonfinite Middle metrics')
    score=refs['D2S']['R@1']+refs['S2D']['R@1']
    if m['best_score']!=score or m['selection_metrics'].get('R1_sum')!=score:raise ValueError('Invalid Middle best score')
    if payload['selection_metrics']!=m['selection_metrics'] or payload['selection_protocol']!=expected:
        raise ValueError('Conflicting Middle selection reference')
    if 'public_config' in payload and config['checkpoint'].get('selection_eval_batch_size')!=FORMAL_MIDDLE_SELECTION_BATCH:
        raise ValueError('Middle runtime selection batch mismatch')
    if (payload['precision_signature']['image_size']!=size
            or payload['precision_signature'].get('selection_batch_size')!=expected['eval_batch_size']):
        raise ValueError('Middle resolution or selection batch mismatch')
    formal_e3=str(config['experiment']['name']).startswith(
        ('M2-SAM-E3-KD-R', 'M3-SAM-HRD-SEM-R'))
    if 'teacher' in m:
        validate_teacher_identity(m['teacher']['checkpoint_metadata'],size)
    elif formal_e3 or (size!=224 and len(config['distillation'])>1):
        raise ValueError('Missing Middle Teacher checkpoint identity')
    if formal_e3 and (m.get('sam') is not True or m.get('sharpness')!=config['sam']):
        raise ValueError('Missing or inconsistent E3 SAM metadata')
    return m

class MiddleCheckpointController(CheckpointController):
    def __init__(self,output_dir,config,public_config=None):
        components=set(config['distillation'])
        objective={
            frozenset({'base_loss'}): 'PairInfoNCE',
            frozenset({'base_loss','margin'}): 'PairInfoNCE_HRD',
            frozenset({'base_loss','margin','adaptive_bridge_v2'}): 'PairInfoNCE_HRD_SEMANTIC',
        }.get(frozenset(components))
        if objective is None:raise ValueError('Unsupported Middle objective components')
        super().__init__(output_dir,objective=objective)
        self.config=config
        self.public_config=public_config
    def _save_portable(self,model_or_engine,filename,epoch,global_step,metrics=None):
        if filename!='best_model.pth':raise ValueError('Formal Middle saves best_model.pth only')
        if rank()!=0:return
        refs=flat_selection_metrics(metrics);refs['R1_sum']=self.best_score
        selection_batch=self.config['checkpoint'].get('selection_eval_batch_size',32)
        protocol=selection_metadata(self.config['data']['input_size'],selection_batch)
        metadata=dict(protocol,experiment_id=self.config['experiment']['name'],best_epoch=self.best_epoch,
            best_score=self.best_score,training_world_size=self.config['data']['world_size'],selection_metrics=refs,
            distillation=self.config['distillation'],sam=self.config['sam']['enabled'])
        if self.public_config is not None:
            encoded=json.dumps(self.public_config,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
            metadata['public_config_sha256']=hashlib.sha256(encoded).hexdigest()
        teacher_identity=getattr(raw_model(model_or_engine),'distillation_teacher_identity',None)
        if teacher_identity is not None:metadata['teacher']=teacher_identity
        if self.config['sam'].get('framework')=='M2_DISTILL_SAM_V1':
            from .config_identity import runtime_fingerprint
            metadata['canonical_runtime_config_sha256']=runtime_fingerprint(self.config)
            metadata.update({k:self.config['sam'][k] for k in (
                'sharpness_mode','search_direction','perturb_scope','rho')})
            metadata.update({k:self.config['sam'][k] for k in ('balanced_task_weight','balanced_kd_weight') if k in self.config['sam']})
            if self.config['sam'].get('sharpness_mode') == 'asam':
                metadata['asam_eta'] = self.config['sam']['asam_eta']
            metadata.update(sharpness=self.config['sam'],
                best_epoch_gradient_diagnostics=raw_model(model_or_engine).sam_epoch_diagnostics)
        payload=dict(artifact_schema=SCHEMA,model=portable_state(model_or_engine),config=self.config,
            metadata=metadata,selection_protocol=protocol,selection_metrics=refs,
            precision_signature=selection_signature(raw_model(model_or_engine),'middle',protocol['image_size'],
                selection_batch_size=selection_batch))
        if self.public_config is not None:payload['public_config']=self.public_config
        checkpoint_metadata(payload)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        temporary=self.output_dir/'best_model.pth.tmp'
        torch.save(payload,temporary);temporary.replace(self.output_dir/filename)
