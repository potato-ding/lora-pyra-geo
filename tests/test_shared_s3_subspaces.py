"""S3 predictor controls reuse S3 targets and retain the formal SAM protocol."""
import copy
import json
from pathlib import Path
import pytest
from src.student.formal_config import load_formal_config
ROOT = Path(__file__).resolve().parents[1]




@pytest.mark.parametrize("source_method,source_provider", [("S2-ADSD-R256", "M3"), ("S3-SAM-ADSD-R256", "M4")])
def test_s3_reference_rejects_other_method_or_provider(monkeypatch, source_method, source_provider):
    from src.student.shared_subspaces import load_reference
    _, cfg = load_formal_config(ROOT / "configs/student/s3_sam_adsd_shared_m3_256.json")
    reference = cfg["subspace_reference"]
    monkeypatch.setattr("src.student.shared_subspaces.file_sha256", lambda p: reference["checkpoint_sha256"])
    monkeypatch.setattr("src.student.formal_checkpoint.validate_checkpoint", lambda p: None)
    monkeypatch.setattr("torch.load", lambda *a, **kw: {"public_config": {"experiment_id": source_method, "middle_provider": source_provider}})
    with pytest.raises(ValueError, match="method/provider"):
        load_reference(reference, cfg, {})




def test_independent_asymmetric_s3_changes_only_random_source_and_output():
    _, baseline = load_formal_config(ROOT / "configs/student/s3_sam_adsd_shared_m3_256.json")
    _, cfg = load_formal_config(ROOT / "configs/student/s3_sam_adsd_independent1_m3_256.json")
    assert cfg.pop("random_basis_replicate") == 1
    assert "subspace_reference" not in cfg and cfg["distillation"]["random_head"] == "linear"
    baseline.pop("subspace_reference")
    baseline["distillation"]["random_basis"] = "gaussian_qr_per_run"
    cfg["output_dir"] = baseline["output_dir"]
    assert cfg == baseline


def test_shared_s3_keeps_formal_asymmetric_sam_protocol():
    _, cfg = load_formal_config(ROOT / "configs/student/s3_sam_adsd_shared_m3_256.json")
    baseline = json.loads((ROOT / "configs/student/s3_sam_adsd_256.json").read_text())
    assert "/S3-256-M3/" in cfg["subspace_reference"]["checkpoint"]
    assert cfg["distillation"]["top_head"] == "residual_mlp"
    assert cfg["distillation"]["random_head"] == "linear"
    assert cfg["sam"] == baseline["sam"]
    for key in ("epochs", "seed", "lr", "weight_decay", "task_loss", "local_pair_batch", "selection_start_epoch"):
        assert cfg[key] == baseline[key]


def test_m3_hrd_control_only_adds_hrd_and_changes_output():
    from src.student.middle_teacher_hrd import validate_config
    from src.student.formal_train import collect_student_source_identity
    _,base=load_formal_config(ROOT/'configs/student/s3_sam_adsd_shared_m3_256.json')
    _,cfg=load_formal_config(ROOT/'configs/student/s3_sam_adsd_hrd_shared_m3_256.json')
    assert {k for k in cfg if cfg[k]!=base.get(k)}=={'hrd','output_dir'}
    source=collect_student_source_identity(cfg)
    for n in ('src/student/middle_teacher_hrd.py','src/student/direct_teacher_hrd.py','src/middle_teacher/losses/hard_rank_distillation.py'):
        assert n in source
    for field,value in [('middle_provider','M4'),('img_size',224),('teacher_provider','T0')]:
        bad=copy.deepcopy(cfg);bad[field]=value
        with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(cfg);bad['hrd']['weight']=.2
    with pytest.raises(ValueError):validate_config(bad)
