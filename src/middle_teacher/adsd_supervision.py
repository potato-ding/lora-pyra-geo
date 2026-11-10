"""768D Middle train-only asymmetric heads over frozen 4096D T0 assets."""
import json
import random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.student.part2 import FP32Module, ScalarGate
from src.student.formal_supervision import SubspaceSupervision
from src.student.direct_teacher_assets import teacher_identity, load_top_source, generate_random32
from src.student.subspace_utils import tensor_sha256, orthogonality_error
from src.middle_teacher.checkpoint import sha256
from src.utils.gather_features_and_labels_and_views import concat_all_gather


def load_assets(public):
    a=public['adsd'];identity=teacher_identity(public['teacher_checkpoint'])
    if json.loads(Path(a['teacher_asset_identity']).read_text())!=identity:
        raise ValueError('T0 asset identity mismatch')
    top=load_top_source(a['supervision_asset'],identity['checkpoint_sha256'])
    if top['metadata']['teacher_config_sha256']!=identity['config_sha256']:
        raise ValueError('T0 Top config identity mismatch')
    basis=torch.load(a['random_basis_asset'],map_location='cpu',weights_only=True)
    ref=a['random_reference']
    if basis.shape!=(4096,32) or basis.dtype!=torch.float32 or not torch.isfinite(basis).all():
        raise ValueError('T0 Random32 shape/precision')
    if tensor_sha256(basis)!=ref['t0_basis_sha256'] or not torch.equal(basis,generate_random32(ref['random_basis_seed'])):
        raise ValueError('T0 Random32 SHA/protocol mismatch')
    tensors={k:top[k] for k in ('teacher_mean','top128_basis')};tensors['random32_basis']=basis
    audit={'teacher_checkpoint_sha256':identity['checkpoint_sha256'],
           'teacher_config_sha256':identity['config_sha256'],
           'manifest_sha256':sha256(a['supervision_asset']),
           'random_asset_sha256':sha256(a['random_basis_asset']),
           'random_basis_seed':ref['random_basis_seed'],
           'tensor_sha256':{k:tensor_sha256(v) for k,v in tensors.items()}}
    return tensors,audit


class LinearHead(FP32Module):
    def __init__(self,out_dim):
        super().__init__();self.linear=nn.Linear(768,out_dim)
    def forward(self,x):
        raw=F.linear(x.float(),self.linear.weight,self.linear.bias)
        return F.normalize(raw,dim=-1),raw


class ResidualHead(LinearHead):
    def __init__(self):
        super().__init__(128)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.fc1=nn.Linear(768,920);self.fc2=nn.Linear(920,128)
        self.gate=ScalarGate()
    def residual(self,x):return self.fc2(F.gelu(self.fc1(x.float())))
    @torch.no_grad()
    def calibrate(self,x):
        base=self.linear(x.float());res=self.residual(x)
        scale=base.square().mean().sqrt()/res.square().mean().sqrt()
        if not torch.isfinite(scale) or scale<=0:raise ValueError('Invalid head calibration')
        self.fc2.weight.mul_(scale);self.fc2.bias.mul_(scale)
        return dict(rows=len(x),scale=float(scale),alpha=float(self.gate.alpha),
                    residual_base_ratio=float((self.gate.alpha*self.residual(x)).norm()/base.norm()))
    def forward(self,x):
        raw=self.linear(x.float())+self.gate.alpha*self.residual(x)
        return F.normalize(raw,dim=-1),raw


class MiddleSubspaceSupervision(FP32Module):
    teacher_targets=SubspaceSupervision.teacher_targets
    _branch_loss=staticmethod(SubspaceSupervision._branch_loss)
    forward=SubspaceSupervision.forward
    def __init__(self,tensors,calibration,seed):
        super().__init__()
        for name,value in tensors.items():self.register_buffer(name,value.clone())
        self.random_structure_metadata={'source':'T0 shared asset'}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.projector_top=ResidualHead();self.projector_random=LinearHead(32)
        self.initialization_audit=self.projector_top.calibrate(calibration.cpu().float())


@contextmanager
def deployment_only(model):
    heads=model._modules.pop('adsd_supervision')
    try:yield model
    finally:model.add_module('adsd_supervision',heads)


@torch.no_grad()
def calibration_descriptors(model,config,device):
    from src.data.middle_teacher import create_middle_teacher_train_dataset_and_loader
    py_state=random.getstate();np_state=np.random.get_state();was_training=model.training
    try:
        with torch.random.fork_rng(devices=[device.index]):
            seed=config['adsd']['calibration_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            _,loader=create_middle_teacher_train_dataset_and_loader(config)
            loader.batch_sampler.set_epoch(0)
            model.eval();rows=[]
            for i,batch in enumerate(loader):
                images=torch.cat(batch[:2]).to(device=device,dtype=torch.bfloat16)
                rows.append(concat_all_gather(model(images)).cpu())
                if sum(len(x) for x in rows)>=config['adsd']['calibration_rows']:break
            result=torch.cat(rows)[:config['adsd']['calibration_rows']].float()
            if result.shape!=(768,768) or not torch.isfinite(result).all():raise ValueError('Middle calibration contract')
            return result
    finally:
        random.setstate(py_state);np.random.set_state(np_state);model.train(was_training)


def add_head_optimizer_groups(model,optimizer,config):
    a=config['adsd']
    for decay in (True,False):
        ps=[p for n,p in model.adsd_supervision.named_parameters() if (p.ndim>1 and not n.endswith('.bias'))==decay]
        optimizer.add_param_group(dict(params=ps,lr=a['head_lr'],weight_decay=a['head_weight_decay'] if decay else 0.,
                                      group_name='adsd_'+('decay' if decay else 'no_decay')))
    # Dtype groups must be finalized before scheduler construction.
    groups=[]
    for g in optimizer.param_groups:
        for dtype in {p.dtype for p in g['params']}:
            groups.append(dict(g,params=[p for p in g['params'] if p.dtype==dtype]))
    optimizer.param_groups[:]=groups
