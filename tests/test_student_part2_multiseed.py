"""Matched shared Top-RMLP seeds, historical source and exact numerical regression."""
import copy, json, subprocess, types
import pytest
import torch
import torch.nn.functional as F
from src.student.artifacts import ROOT
from src.student.part1 import PartISupervision
from src.student.part2 import trainable_count
from src.student.part2_integration import validate_config,prepare_top
from src.student.train import load_config
from test_student_part1 import banks

S0_COMMIT='7bbd78cc8fb5794f782e07e68c8e163c217db632'
IDENTITY={'seed','experiment_name','output_dir','sealed_provenance_file'}


@pytest.mark.parametrize('seed',[0,1,2])
def test_only_matched_shared_seeds(seed):
    cfg=load_config(ROOT/f'configs/student/certified_r224/p2_top_rmlp_s{seed}.json')
    ref=json.loads((ROOT/'configs/student/certified_r224/p2_top_rmlp_s0.json').read_text())
    assert validate_config(cfg)
    assert {k:v for k,v in cfg.items() if k not in IDENTITY}=={k:v for k,v in ref.items() if k not in IDENTITY}
    for changes in [dict(seed=3),dict(seed=True),dict(seed=(seed+1)%3),dict(experiment_name='P2-TOP-RMLP-S3'),
                    dict(lambda_top=1.247),dict(batch_size=16),dict(epochs=2),dict(top_dim=64),
                    dict(top_interface='factorial_dual'),dict(p2_alpha_init=.01),dict(p2_mlp_hidden_dim=1082),
                    dict(p2_calibration_sha256='0'*64),dict(random_layout='single64')]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**changes))


@pytest.mark.parametrize('seed',[0,1,2])
def test_shared_path_against_real_s0_source(banks,seed):
    old=types.ModuleType('src.student._historical_p2_integration')
    old.__package__='src.student'
    source=subprocess.check_output(['git','show',S0_COMMIT+':src/student/part2_integration.py'],cwd=ROOT,text=True)
    exec(compile(source,'historical_S0_part2_integration.py','exec'),old.__dict__)
    torch.manual_seed(seed)
    current=PartISupervision(banks[1],banks[0],banks[2],128,'single32')
    reference=copy.deepcopy(current)
    cfg=load_config(ROOT/f'configs/student/certified_r224/p2_top_rmlp_s{seed}.json')
    s0=json.loads((ROOT/'configs/student/certified_r224/p2_top_rmlp_s0.json').read_text())
    rng=torch.get_rng_state().clone();prepare_top(current,cfg)
    assert torch.equal(rng,torch.get_rng_state())
    old.prepare_top(reference,s0)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,reference.state_dict()[k]) for k,v in current.state_dict().items())
    assert not hasattr(current,'factorial_interface') and not hasattr(current.projector_top,'active_view')
    assert not hasattr(current.projector_random,'residual')
    assert trainable_count(current.projector_top.linear)==65664
    assert trainable_count(current.projector_top.residual)==589848
    assert trainable_count(current.projector_random)==16416
    assert current.projector_top.alpha.item()==pytest.approx(.001)
    torch.manual_seed(20260916)
    z=F.normalize(torch.randn(16,512),dim=1)
    components=[]
    for sup in [current,reference]:
        head=sup.projector_top
        base=F.linear(z,head.linear.weight.float(),head.linear.bias.float())
        residual=head.residual(z)
        gated=head.alpha*residual
        normalized,raw=head(z)
        random_normalized,random_raw=sup.projector_random(z)
        assert torch.equal(raw,base+gated)
        assert torch.equal(normalized,F.normalize(base+gated,dim=-1))
        assert torch.count_nonzero(gated[:8])>0 and torch.count_nonzero(gated[8:])>0
        components.append([base,residual,gated,normalized,raw,random_normalized,random_raw])
    assert all(torch.equal(a,b) for a,b in zip(*components))
