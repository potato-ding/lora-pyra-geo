"""Public source identity and pre-removal mathematical fingerprints."""
import hashlib,json
from pathlib import Path
import torch

def test_frozen_model_and_loss_sources():
    baseline=json.loads(Path('tests/fixtures/formal_source_fingerprints.json').read_text())
    for name,value in baseline.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==value,name

def test_top128_pca_golden():
    from src.student.build_top_only import fit_top128
    from src.student.subspace_utils import tensor_sha256
    previous=torch.get_num_threads()
    try:
        torch.set_num_threads(2);torch.manual_seed(519)
        rows=[torch.nn.functional.normalize(torch.randn(701,768),dim=1) for _ in range(2)]
        torch.set_num_threads(8);actual=fit_top128(rows)
        expected=json.loads(Path('tests/fixtures/top128_math_golden.json').read_text())
        assert {k:tensor_sha256(v) for k,v in actual.items()}==expected
    finally:torch.set_num_threads(previous)

def test_all_formal_source_identity_groups_exist():
    from src.source_contract import source_identity
    for stage in ('teacher','t2m','m2s','evaluation','asset_build','repro'):
        identity=source_identity(stage)
        assert identity and all(Path(p).is_file() for p in identity)
        assert not any('/legacy_' in p for p in identity)

def test_only_final_size_and_sam_configs():
    from src.middle_teacher.distill_sam import validate_sharpness
    for size in (224,256):
        cfg=json.loads(Path(f'configs/middle_teacher/m2-sam-e3-kd-r{size}-s0.json').read_text())
        assert validate_sharpness(cfg)
