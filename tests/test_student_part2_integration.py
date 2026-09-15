"""Formal P2 dispatch guards and unchanged optimizer/launcher mathematics."""
import ast,copy,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.student.train import load_config,deepspeed_config
from src.student.part1 import BandProjector
from src.student.part2 import ResidualTopProjector
from src.student.part2_integration import validate_config,prepare_precision_groups
from src.student.optimizer import build_student_optimizer
from p2_source_contract import before_p2

ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('name',['rmlp','rkan'])
def test_configs_strictly_matched(name):
    cfg=load_config(ROOT/f'configs/student/certified_r224/p2_top_{name}_s0.json')
    assert validate_config(cfg)
    for changes in [dict(seed=1),dict(lambda_top=1.247),dict(batch_size=16),
                    dict(top_dim=64),dict(random_layout='single64'),dict(epochs=2),
                    dict(p2_grid_size=6),dict(experiment_name='OTHER'),dict(top_interface='residual_other')]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**changes))

def test_linear_path_no_change():
    cfg=load_config(ROOT/'configs/student/certified_r224/p1_t128_r32_s0.json')
    assert not validate_config(cfg)
    model=nn.Linear(3,2)
    optimizer=build_student_optimizer(model)
    groups=optimizer.param_groups
    prepare_precision_groups(model,optimizer,cfg)
    assert optimizer.param_groups is groups
    assert next(model.parameters()).dtype==torch.float32

def test_adamw_dtype_split_preserves_updates_and_scheduler():
    torch.manual_seed(0)
    model=nn.ModuleDict({'student':nn.Linear(512,3),'top':ResidualTopProjector(BandProjector(128),'rmlp')}).bfloat16()
    other=copy.deepcopy(model)
    a=build_student_optimizer(model);b=build_student_optimizer(other)
    original={id(p):g['weight_decay'] for g in b.param_groups for p in g['params']}
    prepare_precision_groups(other,b,dict(top_interface='residual_mlp'))
    assert all(original[id(p)]==g['weight_decay'] for g in b.param_groups for p in g['params'])
    assert next(g for g in b.param_groups if any(p is other['top'].alpha for p in g['params']))['weight_decay']==0.
    from src.student.scheduler import build_student_scheduler
    cfg=SimpleNamespace(epochs=30,warmup_epochs=.1,min_lr_ratio=.01)
    sa=build_student_scheduler(a,cfg,1182);sb=build_student_scheduler(b,cfg,1182)
    for _ in range(2):
        for p,q in zip(model.parameters(),other.parameters()):
            p.grad=torch.randn_like(p);q.grad=p.grad.clone()
        a.step();b.step();sa.step();sb.step()
        assert all(torch.equal(p,q) for p,q in zip(model.parameters(),other.parameters()))
        assert set(sa.get_last_lr())==set(sb.get_last_lr())

def test_only_explicit_trainer_insertions_and_frozen_component():
    old=subprocess.check_output(['git','show','baf1f3250b10d1b2ee9317c9b78e6eff50b879d6:src/student/train.py'],cwd=ROOT,text=True)
    now=(ROOT/'src/student/train.py').read_text()
    assert before_p2('src/student/train.py',now)==old
    for name in ['src/student/part2.py','src/student/part1.py','src/student/optimizer.py',
                 'src/student/scheduler.py','src/student/canonical_selection.py','src/student/evaluate_best.py',
                 'scripts/train_student_certified.sh']:
        assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show','baf1f3250b10d1b2ee9317c9b78e6eff50b879d6:'+name],cwd=ROOT)
    ref=subprocess.check_output(['git','show','58921e3683b37f478f7eefbca6ec367005edb74a:src/student/train.py'],cwd=ROOT,text=True)
    def ds(t):return ast.dump(next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name=='deepspeed_config'))
    assert ds(ref)==ds(now)
