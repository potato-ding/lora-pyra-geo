"""Frozen pre-pruning numeric reference; no historical implementation imports."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from src.student import formal_supervision as module
from src.student.random_structure import configure_basis
from src.student.part2 import install_residual_top
from src.student.allocation_gbw import AllocationGate,objective_from_descriptors
from src.student.objective import PairInfoNCE
from src.student.subspace_utils import tensor_sha256

def test_s3_frozen_supervision_and_gbw():
    reference=json.loads(Path('tests/fixtures/s3_math_golden.json').read_text())
    previous=torch.get_num_threads()
    try:
        torch.set_num_threads(2);torch.manual_seed(431)
        asset=dict(teacher_mean=torch.randn(768),top128_basis=torch.linalg.qr(torch.randn(768,128)).Q,metadata={'image_size':224})
        with patch.object(module,'load_top_source',return_value=asset),patch.object(module,'file_sha256',return_value='test'):
            torch.manual_seed(107);sup=module.FormalSupervision('test','sha')
        cfg=dict(random_basis_mode='gaussian_qr_per_run',random_basis_seed=345,random_projector_type='linear',seed=0,img_size=224,experiment_name='S3-ADUAL-LEARNABLE-R224',batch_size=32,stst_weight=.2,stst_warmup_epochs=5)
        configure_basis(sup,cfg);install_residual_top(sup,'rmlp',torch.randn(768,512))
        z=F.normalize(torch.randn(64,512),dim=1);y=F.normalize(torch.randn(64,768),dim=1)
        assert {k:tensor_sha256(v) for k,v in sup.state_dict().items()}==reference['state_sha256']
        gate=AllocationGate('bounded',0.)
        total,gl,metrics=objective_from_descriptors(SimpleNamespace(logit_scale=torch.tensor(0.)),sup,z,y,PairInfoNCE(label_smoothing=.1),cfg,1,gate)
        gl.backward()
        assert float(total)==reference['total'] and float(gl)==reference['gate_loss']
        assert float(gate.d.grad)==reference['gate_gradient']
        assert {k:float(v) for k,v in metrics.items() if v is not None}==reference['metrics']
    finally:torch.set_num_threads(previous)
