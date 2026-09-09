import pytest
import torch
from src.evaluation.evaluate import parse_args
from src.evaluation.model_loader import normalize_state
from src.student.objective import PairInfoNCE
from src.student.train import load_config


def test_student_formal_configs():
    for name in ('baseline','dual_stst'):
        assert load_config(f'configs/student/{name}.json')['epochs']==30


def test_pair_infonce_is_bidirectional():
    torch.manual_seed(0)
    a=torch.randn(4,512);b=torch.randn(4,512)
    criterion=PairInfoNCE()
    assert torch.equal(criterion(a,b,torch.tensor(3.)),criterion(b,a,torch.tensor(3.)))


def test_strict_prefix_normalization():
    tensor=torch.tensor([1.])
    assert torch.equal(normalize_state({'state_dict':{'module.x':tensor}})['x'],tensor)
    with pytest.raises(ValueError):normalize_state({'state_dict':{'x':tensor,'module.x':tensor}})


@pytest.mark.parametrize('option,value',[('--gta-query-mode','both'),('--gta-query-mode','S2D'),('--gta-split','same-area')])
def test_gta_nonformal_cli_rejected(option,value):
    with pytest.raises(SystemExit):
        parse_args(['--model-type','middle','--checkpoint','unused','--dataset','gta','--output-dir','unused',option,value])
