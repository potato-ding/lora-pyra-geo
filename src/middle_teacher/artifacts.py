"""Self-contained Middle best checkpoint; no Teacher tensors or sidecar files."""
import math
import torch
from .checkpoint import CheckpointController, portable_state, raw_model
from .distributed import rank
from .selection import selection_metadata
from src.evaluation.precision_contract import selection_signature, flat_selection_metrics

SCHEMA='MIDDLE_BEST_MODEL_V2'

def checkpoint_metadata(payload):
    if payload.get('artifact_schema') != SCHEMA:
        return None
    m=payload['metadata'];config=payload['config']
    expected=selection_metadata(224)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Invalid Middle selection protocol')
    if m['experiment_id']!=config['experiment']['name'] or m['training_world_size']!=2:
        raise ValueError('Invalid Middle experiment metadata')
    if not isinstance(m['best_epoch'],int) or not 1<=m['best_epoch']<=10:raise ValueError('Invalid best epoch')
    refs=flat_selection_metrics(m['selection_metrics'])
    if not all(math.isfinite(v) for values in refs.values() for v in values.values()):raise ValueError('Nonfinite Middle metrics')
    score=refs['D2S']['R@1']+refs['S2D']['R@1']
    if m['best_score']!=score or m['selection_metrics'].get('R1_sum')!=score:raise ValueError('Invalid Middle best score')
    if payload['selection_metrics']!=m['selection_metrics'] or payload['selection_protocol']!=expected:
        raise ValueError('Conflicting Middle selection reference')
    if payload['precision_signature']['image_size']!=224:raise ValueError('Middle resolution mismatch')
    return m

class MiddleCheckpointController(CheckpointController):
    def __init__(self,output_dir,config):
        super().__init__(output_dir,objective='PairInfoNCE_HRD_SEMANTIC' if len(config['distillation'])>1 else 'PairInfoNCE')
        self.config=config
    def _save_portable(self,model_or_engine,filename,epoch,global_step,metrics=None):
        if filename!='best_model.pth':raise ValueError('Formal Middle saves best_model.pth only')
        if rank()!=0:return
        refs=flat_selection_metrics(metrics);refs['R1_sum']=self.best_score
        protocol=selection_metadata(self.config['data']['input_size'])
        metadata=dict(protocol,experiment_id=self.config['experiment']['name'],best_epoch=self.best_epoch,
            best_score=self.best_score,training_world_size=self.config['data']['world_size'],selection_metrics=refs,
            distillation=self.config['distillation'],sam=self.config['sam']['enabled'])
        payload=dict(artifact_schema=SCHEMA,model=portable_state(model_or_engine),config=self.config,
            metadata=metadata,selection_protocol=protocol,selection_metrics=refs,
            precision_signature=selection_signature(raw_model(model_or_engine),'middle',protocol['image_size']))
        checkpoint_metadata(payload)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        temporary=self.output_dir/'best_model.pth.tmp'
        torch.save(payload,temporary);temporary.replace(self.output_dir/filename)
