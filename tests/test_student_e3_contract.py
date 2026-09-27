"""Formal E3 Middle provenance gates for both Student resolutions."""
import copy
import json
from pathlib import Path

import pytest

from src.middle_teacher.fchain_runtime import validate_fchain
from src.student.core_config import validate_config
from src.student.middle_source import validate_e3_middle

ROOT = Path(__file__).resolve().parents[1]
E3_224 = ROOT / "src/checkpoint/middle_teacher/R224/M2-SAM-E3-KD-R224-S0/best_model.pth"
CONFIG_224 = ROOT / "configs/middle_teacher/m2-sam-e3-kd-r224-s0.json"


def test_e3_resolution_templates_and_student_bindings():
    configs = [json.loads((ROOT / f"configs/middle_teacher/m2-sam-e3-kd-r{size}-s0.json").read_text())
               for size in (224, 256)]
    normalized = copy.deepcopy(configs)
    normalized[1]["experiment"]["name"] = normalized[0]["experiment"]["name"]
    normalized[1]["checkpoint"]["output_dir"] = normalized[0]["checkpoint"]["output_dir"]
    normalized[1]["data"]["input_size"] = 224
    from src.middle_teacher.config_identity import canonical_runtime_config
    assert canonical_runtime_config(normalized[0]) == canonical_runtime_config(normalized[1])
    for size, middle in zip((224, 256), configs):
        validate_fchain(middle, None)
        student = json.loads((ROOT / f"configs/student/r{size}/s3-adual-learnable-r{size}.json").read_text())
        validate_config(student)
        assert student["middle_config"].endswith(f"m2-sam-e3-kd-r{size}-s0.json")
        assert student["middle_checkpoint"].endswith(f"M2-SAM-E3-KD-R{size}-S0/best_model.pth")


@pytest.mark.skipif(not E3_224.is_file(), reason="E3 R224 checkpoint not installed")
def test_r224_real_e3_checkpoint_assets_and_fail_closed(monkeypatch):
    from src.student import middle_source
    student = json.loads((ROOT / "configs/student/r224/s3-adual-learnable-r224.json").read_text())
    validate_config(student, check_assets=True)
    config, actual = validate_e3_middle(E3_224, CONFIG_224, 224, student["middle_checkpoint_sha256"])
    assert actual == student["middle_checkpoint_sha256"]
    assert config["sam"]["search_direction"] == "kd"
    with pytest.raises(ValueError):
        validate_e3_middle(E3_224, CONFIG_224, 256)
    with pytest.raises(ValueError, match="SHA"):
        validate_e3_middle(E3_224, CONFIG_224, 224, "0" * 64)
    payload = middle_source.safe_load(E3_224)
    invalid = copy.copy(payload)
    invalid["metadata"] = dict(payload["metadata"], sam=False)
    monkeypatch.setattr(middle_source, "safe_load", lambda _: invalid)
    with pytest.raises(ValueError, match="SAM metadata"):
        validate_e3_middle(E3_224, CONFIG_224, 224)
    invalid["metadata"] = dict(payload["metadata"], teacher=None)
    with pytest.raises(ValueError, match="Teacher provenance"):
        validate_e3_middle(E3_224, CONFIG_224, 224)
    invalid["metadata"] = payload["metadata"]
    invalid["precision_signature"] = dict(payload["precision_signature"], image_size=256)
    with pytest.raises(ValueError):
        validate_e3_middle(E3_224, CONFIG_224, 224)



def test_e3_legacy_and_canonical_runtime_config_identity():
    from src.middle_teacher.config_identity import canonical_runtime_config, runtime_fingerprint
    legacy=json.loads(CONFIG_224.read_text())
    canonical=canonical_runtime_config(legacy)
    assert 'balanced_task_weight' not in canonical['sam']
    assert 'balanced_kd_weight' not in canonical['sam']
    assert runtime_fingerprint(legacy)==runtime_fingerprint(canonical)
    future=json.loads((ROOT/'configs/middle_teacher/m2-sam-e3-kd-r256-s0.json').read_text())
    assert 'balanced_task_weight' not in future['sam']
    assert 'balanced_kd_weight' not in future['sam']
    validate_fchain(future,None)
    altered=copy.deepcopy(legacy);altered['sam']['rho']=.11
    assert runtime_fingerprint(altered)!=runtime_fingerprint(legacy)
    unknown=copy.deepcopy(legacy);unknown['sam']['mystery']=1
    with pytest.raises(ValueError,match='Unknown E3 SAM'):
        runtime_fingerprint(unknown)
    invalid=copy.deepcopy(legacy);invalid['sam']['balanced_task_weight']=.6
    with pytest.raises(ValueError,match='nonruntime'):
        runtime_fingerprint(invalid)


def test_future_e3_v2_checkpoint_preserves_sam_and_source(tmp_path, monkeypatch):
    import torch
    from src.middle_teacher.artifacts import MiddleCheckpointController, checkpoint_metadata
    from src.middle_teacher.config_identity import runtime_fingerprint
    from src.evaluation.model_loader import load_encoder
    import src.middle_teacher.model as model_module
    config=json.loads((ROOT/'configs/middle_teacher/m2-sam-e3-kd-r256-s0.json').read_text())
    model=torch.nn.Linear(4,768,bias=False).bfloat16()
    model.distillation_teacher_identity=dict(
        checkpoint='/teacher/R256/T0-INFONCE-R256/best_model.pth',sha256='f'*64,
        checkpoint_metadata=dict(experiment_id='T0-INFONCE-R256',image_size=256,
                                 selection_mode='SINGLE_GPU_CANONICAL',selection_world_size=1,selection_rank=0))
    model.sam_epoch_diagnostics=dict(steps=1,mean_task_grad_norm=1.,mean_kd_grad_norm=2.)
    controller=MiddleCheckpointController(tmp_path,config)
    metrics={d+'_'+k:v for d in ('D2S','S2D') for k,v in [('R1',80.),('R5',90.),('AP',75.)]}
    metrics['R1_sum']=160.
    assert controller.save_best_if_improved(model,1,1,metrics)
    path=tmp_path/'best_model.pth'
    payload=torch.load(path,weights_only=True)
    metadata=checkpoint_metadata(payload)
    assert metadata['canonical_runtime_config_sha256']==runtime_fingerprint(config)
    assert metadata['sharpness']==config['sam']
    assert metadata['teacher']==model.distillation_teacher_identity
    assert 'balanced_task_weight' not in metadata
    monkeypatch.setattr(model_module,'build_middle_teacher',
                        lambda *a,**kw:torch.nn.Linear(4,768,bias=False))
    loaded,audit=load_encoder('middle',path,device='cpu',image_size=256)
    assert audit['artifact_classification']=='FORMAL_MIDDLE_CHECKPOINT'
    assert torch.equal(loaded.model.weight,model.weight)
    for replacement in (
            dict(metadata=dict(metadata,canonical_runtime_config_sha256='0'*64)),
            dict(metadata={k:v for k,v in metadata.items() if k!='teacher'}),
            dict(precision_signature=dict(payload['precision_signature'],image_size=224)),
            dict(selection_metrics=dict(payload['selection_metrics'],D2S=dict(payload['selection_metrics']['D2S'],**{'R@1':1.})))):
        invalid=dict(payload,**replacement)
        with pytest.raises(ValueError):checkpoint_metadata(invalid)
