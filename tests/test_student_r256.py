import json
from pathlib import Path
import torch
import pytest
from src.student.core_config import validate_config
from src.student.model import StudentModel
from src.student.formal_supervision import FormalSupervision
from src.student.formal_top import prepare_top
from src.student.random_structure import configure_basis,capture_training_auxiliary,restore_training_auxiliary
from src.student.allocation_gbw import AllocationGate

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/student/r256'
ASSET=ROOT/'src/checkpoint/student/R256/TOP128_CANONICAL'


def test_r256_repvit_geometry_and_descriptor():
    model=StudentModel(ckpt_path=None).eval()
    with torch.no_grad():
        out=model(torch.zeros(2,3,256,256),return_audit_features=True)
    assert out['f4'].shape==(2,512,8,8)
    assert out['f4_gap'].shape==out['bn_input'].shape==(2,512)
    assert out['final_descriptor'].shape==(2,512)
    assert out['final_descriptor'].dtype==torch.float32

