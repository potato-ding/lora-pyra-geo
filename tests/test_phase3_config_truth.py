import copy
import json
import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.student.allocation_gbw import AllocationGate, gate_objective, objective_from_descriptors
from src.student.core_config import validate_config
from src.student.objective import PairInfoNCE
from src.student.formal_engine import deepspeed_config
from src.training.teacher.args import parse_args


class ToySupervision(nn.Module):
    def forward(self, z, y, pairs):
        top = (z - y).square().mean(1)
        random = (z + y).square().mean(1)
        audit = {}
        for name, values in (('top', top), ('random', random)):
            audit[name + '_drone_loss'] = values[:pairs].mean()
            audit[name + '_satellite_loss'] = values[pairs:].mean()
            audit[name + '_loss'] = .5 * (audit[name + '_drone_loss'] + audit[name + '_satellite_loss'])
        return audit['top_loss'] + audit['random_loss'], audit


@pytest.mark.parametrize('size', (224, 256))
def test_formal_config_is_runtime_source(size):
    cfg = json.loads(Path(f'configs/student/r{size}/s3-adual-learnable-r{size}.json').read_text())
    assert validate_config(cfg) == cfg
    assert 'lambda_top' not in cfg and 'lambda_random' not in cfg
    for key, other in (('stst_weight', .3), ('stst_warmup_epochs', 4),
                       ('batch_size', 16), ('gate_initial_d', .2)):
        invalid = dict(cfg, **{key: other})
        with pytest.raises(ValueError):
            validate_config(invalid)
    for key in ('lambda_top', 'lambda_random'):
        with pytest.raises(ValueError):
            validate_config(dict(cfg, **{key: 1.0}))
    ds = deepspeed_config(cfg)
    assert (ds['train_batch_size'], ds['train_micro_batch_size_per_gpu'],
            ds['gradient_accumulation_steps']) == (32, 32, 1)
    assert ds['bf16']['enabled'] and not ds['fp16']['enabled']


def test_formal_student_equation_and_config_flow():
    cfg = json.loads(Path('configs/student/r224/s3-adual-learnable-r224.json').read_text())
    student = nn.Module()
    student.logit_scale = nn.Parameter(torch.tensor(2.0))
    torch.manual_seed(91)
    z = torch.randn(64, 4, requires_grad=True)
    y = torch.randn(64, 4)
    gate = AllocationGate(cfg['gate_parameterization'], cfg['gate_initial_d'])
    total, gate_loss, metrics = objective_from_descriptors(
        student, ToySupervision(), z, y, PairInfoNCE(), cfg, 3, gate)
    assert float(metrics['w_top']) == pytest.approx(1.0)
    assert float(metrics['w_rand']) == pytest.approx(1.0)
    assert float(metrics['w_top'] + metrics['w_rand']) == pytest.approx(2.0)
    kd = metrics['w_top'] * metrics['L_top'] + metrics['w_rand'] * metrics['L_random']
    expected = metrics['InfoNCE'] + .2 * (3 / 5) * kd
    assert torch.allclose(total.detach(), expected)
    assert torch.allclose(gate_loss.detach(), (3 / 5) *
        (torch.log(metrics['G_top'] + 1e-8) - torch.log(metrics['G_rand'] + 1e-8)).square())
    changed = dict(cfg, stst_weight=.3, stst_warmup_epochs=2)
    other, _, other_metrics = objective_from_descriptors(
        student, ToySupervision(), z, y, PairInfoNCE(), changed, 3, gate)
    assert torch.allclose(other.detach(), other_metrics['InfoNCE'] + .3 * kd)
    assert not torch.allclose(other.detach(), total.detach())




def test_teacher_formal_default_epoch_count():
    assert parse_args(['--config','configs/teacher/t0_certified_224.json']).epochs == 10
    with pytest.raises(ValueError):
        parse_args(['--experiment_id', 'T0-INFONCE-R224'])


@pytest.mark.parametrize('size', (224, 256))
def test_teacher_formal_json_reaches_cli_runtime(tmp_path, size):
    path = Path(f'configs/teacher/t0_certified_{size}.json')
    cfg = json.loads(path.read_text())
    args = parse_args(['--config', str(path)])
    retired={'triplet_weight','same_domain_triplet_weight','weak_paired_cross_view_weight'}
    assert all(getattr(args, key) == value for key, value in cfg.items() if key not in retired)
    assert all(cfg[key]==0 for key in retired)
    assert args.epochs == 10
    modified = dict(cfg, epochs=11)
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(modified))
    with pytest.raises(ValueError):
        parse_args(['--config', str(bad)])
    bad.write_text(json.dumps(dict(cfg, lr=.01)))
    with pytest.raises(ValueError):
        parse_args(['--config', str(bad)])
    with pytest.raises(ValueError):
        parse_args(['--config', str(path), '--lr', '0.01'])
