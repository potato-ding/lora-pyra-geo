"""Seed/name safety migration; the certified training mathematics stays unchanged."""
import ast
import importlib.util
import json
import random
import subprocess
import types
from pathlib import Path
import numpy as np
import pytest
import torch
from src.student.part1 import validate_part1_config, PartISupervision
from src.student.train import load_config
from src.student.part1_smoke import seed_smoke_runtime
from src.student.runtime import _seed_all
from test_student_part1 import banks

OLD = '58921e3683b37f478f7eefbca6ec367005edb74a'
ROOT = Path(__file__).resolve().parents[1]

def old_text(name):
    return subprocess.check_output(['git','show',f'{OLD}:{name}'],cwd=ROOT,text=True)

def config(top,seed):
    cfg=json.loads((ROOT/f'configs/student/certified_r224/p1_t{top}_r32_s0.json').read_text())
    name=f'P1-T{top}-R32-S{seed}'
    return dict(cfg,seed=seed,experiment_name=name,output_dir=str(Path(cfg['output_dir']).parent/name))

@pytest.mark.parametrize('top',[64,128])
@pytest.mark.parametrize('seed',[0,1,2])
def test_legal_seed_name_directory(top,seed):
    validate_part1_config(config(top,seed))

@pytest.mark.parametrize('seed',[3,-1,True,1.0,'1',None])
def test_unknown_seed_rejected(seed):
    with pytest.raises(ValueError):validate_part1_config(config(64,seed))

@pytest.mark.parametrize('top',[64,128])
@pytest.mark.parametrize('changes',[
    {'seed':1,'experiment_name':'P1-T64-R32-S0'},
    {'seed':2,'output_dir':'/tmp/P1-T64-R32-S1'},
    {'experiment_name':'UNKNOWN'},
    {'output_dir':'/tmp/UNKNOWN'},
    {'experiment_name':None},
])
def test_mismatch_rejected(top,changes):
    cfg=config(top,1);cfg.update(changes)
    with pytest.raises(ValueError):validate_part1_config(cfg)

@pytest.mark.parametrize('top',[64,128])
def test_legacy_s0_config_unchanged_and_new_configs_matched(top):
    path=f'configs/student/certified_r224/p1_t{top}_r32_s0.json'
    from gbw_source_contract import before_gbw
    current=before_gbw(path,(ROOT/path).read_text())
    if path=='src/student/train.py':
        # Only nullable disabled-branch reporting may differ from historical trainer.
        current=current.replace("'random_loss':None if kd_audit['random_loss'] is None else kd_audit['random_loss'].detach(),",
                                "'random_loss':kd_audit['random_loss'].detach(),")
        current=current.replace("{k:('DISABLED' if v is None else float(v)) for k,v in components.items()}",
                                "{k:float(v) for k,v in components.items()}")
    assert current==old_text(path)
    base=load_config(ROOT/path)
    for seed in [1,2]:
        new=load_config(ROOT/f'configs/student/certified_r224/p1_t{top}_r32_s{seed}.json')
        changes={k for k in set(base)|set(new) if base.get(k)!=new.get(k)}
        assert changes=={'seed','experiment_name','output_dir'}

@pytest.mark.parametrize('seed',[0,1,2])
def test_smoke_uses_config_seed_and_all_rngs(seed):
    assert seed_smoke_runtime({'seed':seed})==seed
    actual=(random.random(),float(np.random.rand()),torch.rand(5))
    _seed_all(seed)
    expected=(random.random(),float(np.random.rand()),torch.rand(5))
    assert actual[:2]==expected[:2] and torch.equal(actual[2],expected[2])
    assert torch.initial_seed()==seed
    source=(ROOT/'src/student/part1_smoke.py').read_text()
    assert 'runtime_seed=seed_smoke_runtime(cfg)' in source
    assert '_seed_all(0)' not in source
    assert "/SMOKES'/Path(cfg['output_dir']).name" in source

@pytest.mark.parametrize('path',[
    'src/student/train.py','src/student/runtime.py','src/student/data.py',
    'src/student/dual_stst.py','src/student/objective.py','src/student/optimizer.py',
    'src/student/scheduler.py','src/student/model.py','src/student/artifacts.py',
    'src/student/canonical_selection.py','src/student/canonical_u1652_worker.py',
    'src/student/evaluate_best.py','src/student/launch.py',
    'src/dataset/teacher/datasets.py','src/dataset/transforms.py',
])
def test_training_selector_deployment_and_seed_propagation_source_unchanged(path):
    from gbw_source_contract import before_gbw
    current=before_gbw(path,(ROOT/path).read_text())
    if path=='src/student/train.py':
        # Only nullable disabled-branch reporting may differ from historical trainer.
        current=current.replace("'random_loss':None if kd_audit['random_loss'] is None else kd_audit['random_loss'].detach(),",
                                "'random_loss':kd_audit['random_loss'].detach(),")
        current=current.replace("{k:('DISABLED' if v is None else float(v)) for k,v in components.items()}",
                                "{k:float(v) for k,v in components.items()}")
    assert current==old_text(path)

def test_part1_existing_branch_functions_unchanged():
    def functions(text):
        tree=ast.parse(text)
        return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old=functions(old_text('src/student/part1.py'))
    new=functions((ROOT/'src/student/part1.py').read_text())
    for name in ['build_extended_tensors','check_extended_tensors','load_extended_asset','BandProjector']:
        assert ast.dump(old[name])==ast.dump(new[name])
    previous={n.name:n for n in old['PartISupervision'].body if isinstance(n,ast.FunctionDef)}
    current={n.name:n for n in new['PartISupervision'].body if isinstance(n,ast.FunctionDef)}
    assert ast.dump(previous['_apply'])==ast.dump(current['_apply'])
    forward=current['forward']
    assert ast.unparse(forward.body[0].test)=="self.random_layout == 'disabled'"
    forward.body=forward.body[1:]
    assert ast.dump(previous['forward'])==ast.dump(forward)
    # Actual R32/R64 target, projection, loss, RNG and group equivalence is tested
    # separately against the historical implementation with fixed input.

@pytest.mark.parametrize('top',[64,128])
def test_s0_old_new_math_exact(banks,top):
    old=types.ModuleType('src.student._old_seed_guard_test')
    old.__package__='src.student'
    exec(compile(old_text('src/student/part1.py'),'old_part1.py','exec'),old.__dict__)
    cfg=load_config(ROOT/f'configs/student/certified_r224/p1_t{top}_r32_s0.json')
    old.validate_part1_config(cfg)
    torch.manual_seed(0)
    a=old.PartISupervision(banks[1],banks[0],banks[2],top,'single32').bfloat16()
    state=torch.get_rng_state()
    torch.manual_seed(0)
    b=PartISupervision(banks[1],banks[0],banks[2],top,'single32').bfloat16()
    assert torch.equal(state,torch.get_rng_state())
    x=torch.randn(64,512);y=torch.randn(64,768)
    av,aa=a(x,y,32);bv,ba=b(x,y,32)
    assert torch.equal(av,bv)
    for k in ['top_loss','random_loss']:assert torch.equal(aa[k],ba[k])
    for i in [0,1]:assert torch.equal(a.teacher_targets(y)[i][0],b.teacher_targets(y)[i][0])
    for name in ['projector_top','projector_random']:assert torch.equal(getattr(a,name)(x)[0],getattr(b,name)(x)[0])
