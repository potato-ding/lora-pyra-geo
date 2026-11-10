"""Portable M3+ADSD checkpoint; training heads are separate from deployment state."""
import hashlib,json
import torch
from src.evaluation.protocols import RETRIEVAL_PROTOCOL
from src.evaluation.precision_contract import selection_signature,flat_selection_metrics
from .checkpoint import CheckpointController,portable_state,raw_model
from .artifacts import SCHEMA,checkpoint_metadata
from .distributed import rank
from .selection import selection_metadata
from src.student.subspace_utils import tensor_sha256


def validate_adsd_checkpoint(payload):
    metadata=checkpoint_metadata(payload)
    if metadata is None or payload['public_config']['experiment_id']!='M-SAM-ADSD-R256':
        raise ValueError('M3 ADSD checkpoint identity')
    if payload['config']['distillation'] != {'base_loss': 'pair_infonce'} or any(k in payload['public_config'] for k in ('hrd','semantic')):
        raise ValueError('ADSD-only checkpoint contains another KD method')
    if metadata.get('teacher') is None or metadata.get('sharpness') != payload['config']['sam']:
        raise ValueError('ADSD-only provider/SAM metadata missing')
    auxiliary=payload['training_auxiliary'];heads=auxiliary['adsd'];gate=auxiliary['gate']
    if any(n.startswith('adsd_supervision.') for n in payload['model']):raise ValueError('Training heads in deployment')
    expected={
        'teacher_mean':(4096,),'top128_basis':(4096,128),'random32_basis':(4096,32),
        'projector_top.linear.weight':(128,768),'projector_top.linear.bias':(128,),
        'projector_top.fc1.weight':(920,768),'projector_top.fc1.bias':(920,),
        'projector_top.fc2.weight':(128,920),'projector_top.fc2.bias':(128,),
        'projector_top.gate.alpha':(),
        'projector_random.linear.weight':(32,768),'projector_random.linear.bias':(32,)}
    if set(heads)!=set(expected):raise ValueError('ADSD auxiliary inventory')
    for n,shape in expected.items():
        t=heads[n]
        if tuple(t.shape)!=shape or t.dtype!=torch.float32 or not torch.isfinite(t).all():raise ValueError('ADSD tensor contract '+n)
    if set(gate)!={'d'} or gate['d'].shape!=() or gate['d'].dtype!=torch.float32 or not torch.isfinite(gate['d']):
        raise ValueError('ADSD gate contract')
    for n,digest in metadata['adsd_assets']['tensor_sha256'].items():
        if tensor_sha256(heads[n])!=digest:raise ValueError('ADSD asset identity')
    if not metadata.get('source_identity'):raise ValueError('Source record missing')
    return metadata


class ADSDCheckpointController(CheckpointController):
    def __init__(self,output_dir,config,public_config=None,*,supervision,gate,assets,source):
        self.supervision,self.gate,self.assets,self.source=supervision,gate,assets,source
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
        payload=dict(selection_dataset_protocol=RETRIEVAL_PROTOCOL, artifact_schema=SCHEMA,model=portable_state(model_or_engine),config=self.config,
            metadata=metadata,selection_protocol=protocol,selection_metrics=refs,
            precision_signature=selection_signature(raw_model(model_or_engine),'middle',protocol['image_size'],
                selection_batch_size=selection_batch))
        if self.public_config is not None:payload['public_config']=self.public_config
        payload['training_auxiliary']={'adsd':{k:v.detach().cpu().clone() for k,v in self.supervision.state_dict().items()},
                                       'gate':{k:v.detach().cpu().clone() for k,v in self.gate.state_dict().items()}}
        metadata['adsd_assets']=self.assets
        metadata['source_identity']=self.source
        validate_adsd_checkpoint(payload)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        temporary=self.output_dir/'best_model.pth.tmp'
        torch.save(payload,temporary);temporary.replace(self.output_dir/filename)
