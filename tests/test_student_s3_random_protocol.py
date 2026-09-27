"""Shared formal S3 Random32 lifecycle; no training or evaluator run."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from src.student.allocation_gbw import AllocationGate
from src.student.artifacts import deployment_state_dict
from src.student.subspace_utils import tensor_sha256
from src.student.core_config import validate_config
from src.student.model import StudentModel
from src.student.formal_top import prepare_top
from src.student.random_structure import (
    ALGORITHM, capture_training_auxiliary, configure_basis,
    restore_training_auxiliary, restore_training_checkpoint, validate_basis,
)
from src.student.formal_supervision import make_supervision

ROOT = Path(__file__).resolve().parents[1]


def formal_config(size):
    path = ROOT / f"configs/student/r{size}/s3-adual-learnable-r{size}.json"
    cfg = json.loads(path.read_text())
    validate_config(cfg)
    assert cfg["random_basis_mode"] == "gaussian_qr_per_run"
    assert cfg["random_projector_type"] == "linear"
    assert cfg["random_dim"] == 32
    assert "original_stst_asset" not in cfg
    return cfg


def source_for_test(cfg, size, tmp_path):
    if size == 224:
        validate_config(cfg, check_assets=True)
        return cfg
    # R256 E3 assets are future outputs. Use the installed Top128 tensors only
    # as a synthetic loader fixture; Random32 always comes from the run seed.
    from src.student.formal_supervision import load_top_source
    old = ROOT / "src/checkpoint/student/R256/TOP128_CANONICAL"
    old_meta = json.loads((old / "manifest.json").read_text())
    fixture = tmp_path / "top"
    fixture.mkdir()
    for name in ("teacher_mean.pt", "top128_basis.pt"):
        (fixture / name).write_bytes((old / name).read_bytes())
    old_meta["teacher_sha256"] = "f" * 64
    (fixture / "manifest.json").write_text(json.dumps(old_meta))
    cfg = dict(cfg, stst_asset=str(fixture / "manifest.json"),
               middle_checkpoint_sha256="f" * 64,
               p2_calibration_path=str(old / "diagnostic_inputs.pt"))
    load_top_source(cfg["stst_asset"], cfg["middle_checkpoint_sha256"])
    return cfg


@pytest.mark.parametrize("size", [224, 256])
def test_formal_random_lifecycle(size, tmp_path, monkeypatch):
    cfg = source_for_test(formal_config(size), size, tmp_path)
    supervision = make_supervision(cfg, cfg["middle_checkpoint_sha256"])
    before = torch.get_rng_state().clone()
    configure_basis(supervision, cfg)
    assert torch.equal(before, torch.get_rng_state())
    assert supervision.random_basis_generation_count == 1
    basis = supervision.random32_basis.clone()
    assert basis.shape == (768, 32) and basis.dtype == torch.float32
    assert validate_basis(basis) <= 1e-5
    assert torch.allclose(basis.T @ basis, torch.eye(32), atol=1e-5)
    assert supervision.projector_random.linear.in_features == 512
    assert supervision.projector_random.linear.out_features == 32
    teacher = torch.randn(4, 768)
    (top, _), (random, _) = supervision.teacher_targets(teacher)
    centered = teacher.float() - supervision.teacher_mean
    torch.testing.assert_close(top, F.normalize(centered @ supervision.top32_basis, dim=-1))
    torch.testing.assert_close(random, F.normalize(centered @ basis, dim=-1))
    assert tensor_sha256(supervision.random32_basis) == tensor_sha256(basis)

    prepare_top(supervision, cfg)
    gate = AllocationGate("bounded", 0.)
    auxiliary = capture_training_auxiliary(supervision, gate)
    identity = auxiliary["basis_identity"]
    assert identity["random_basis_generation_count"] == 1
    assert identity["generation_algorithm"] == ALGORITHM
    assert identity["random_basis_seed"] == cfg["random_basis_seed"]
    assert identity["random_basis_sha256"] == tensor_sha256(basis)
    assert identity["random_basis_shape"] == [768, 32]
    assert identity["random_basis_dtype"] == "torch.float32"
    assert torch.equal(auxiliary["supervision"]["random32_basis"], basis)

    with patch("src.student.random_structure.generate_random_basis",
               side_effect=AssertionError("reload must use stored basis")):
        restored = make_supervision(cfg, cfg["middle_checkpoint_sha256"])
        configure_basis(restored, cfg, stored=auxiliary)
        prepare_top(restored, cfg)
        restored_gate = AllocationGate("bounded", 0.)
        restore_training_auxiliary(restored, restored_gate, auxiliary)
    assert restored.random_basis_generation_count == 0
    assert torch.equal(restored.random32_basis, basis)
    assert tensor_sha256(restored.random32_basis) == identity["random_basis_sha256"]
    bad = dict(auxiliary, supervision=dict(auxiliary["supervision"]))
    bad["supervision"]["random32_basis"] = basis.clone()
    bad["supervision"]["random32_basis"][0, 0] += .01
    with pytest.raises(ValueError, match="SHA"):
        configure_basis(make_supervision(cfg, cfg["middle_checkpoint_sha256"]), cfg, stored=bad)
    changed_cfg = dict(cfg, img_size=size + 1)
    with pytest.raises(ValueError, match="config identity"):
        configure_basis(make_supervision(cfg, cfg["middle_checkpoint_sha256"]),
                        changed_cfg, stored=auxiliary)
    with pytest.raises(ValueError):
        validate_basis(torch.ones(768, 32))
    assert torch.equal(restored.projector_random.linear.weight,
                       supervision.projector_random.linear.weight)
    assert torch.equal(restored_gate.d, gate.d)

    from src.evaluation import student_canonical
    from src.student.canonical_selection import select_epoch
    monkeypatch.setattr(student_canonical, "evaluate_student_u1652_canonical",
                        lambda *a, **k: {d: {"R@1": 1., "R@5": 2., "R@10": 3., "AP": 1.}
                                         for d in ("D2S", "S2D")})
    student = StudentModel(ckpt_path=None).bfloat16()
    deployment = deployment_state_dict(student)
    assert all("random32_basis" not in key for key in deployment)
    run = tmp_path / "run"
    run.mkdir()
    metadata = dict(cfg, **supervision.random_structure_metadata, formal_config=dict(cfg))
    select_epoch(student, run, 1, float("-inf"), "unused",
                 image_size=size, run_metadata=metadata,
                 allocation={"mode": "learnable", "lambda_top": 1., "lambda_random": 1.},
                 training_auxiliary=auxiliary)
    saved = torch.load(run / "best_model.pth", map_location="cpu", weights_only=True)
    assert saved["metadata"]["random_basis_generation_count"] == 1
    assert saved["metadata"]["random_basis_sha256"] == identity["random_basis_sha256"]
    assert torch.equal(saved["training_auxiliary"]["supervision"]["random32_basis"], basis)
    if size == 224:
        from src.student.checkpoint_contract import verify_student_best
        from src.evaluation.model_loader import load_encoder
        assert verify_student_best(saved, verify_assets=True)['experiment_id'] == cfg['experiment_name']
        encoder, audit = load_encoder('student', run / 'best_model.pth', device='cpu')
        assert encoder.descriptor_dim == 512
        assert audit['artifact_classification'] == 'FORMAL_STUDENT_CHECKPOINT'
        import copy
        for field, replacement in (
                ('artifact_schema', 'UNKNOWN'), ('protocol_id', 'STU-1G-B32-R256-v1')):
            corrupt = dict(saved, **{field: replacement})
            with pytest.raises(ValueError):verify_student_best(corrupt)
        corrupt = dict(saved, metadata=dict(saved['metadata'], student_architecture='wrong'))
        with pytest.raises(ValueError, match='model/artifact'):verify_student_best(corrupt)
        corrupt = dict(saved, metadata=dict(saved['metadata'], middle_checkpoint_sha256='0'*64))
        with pytest.raises(ValueError, match='source asset SHA'):verify_student_best(corrupt, verify_assets=True)
        corrupt = dict(saved, precision_signature=dict(saved['precision_signature'], image_size=256))
        with pytest.raises(ValueError, match='precision'):verify_student_best(corrupt)
        corrupt = dict(saved, selection_metrics={d:dict(v) for d,v in saved['selection_metrics'].items()})
        corrupt['selection_metrics']['D2S']['R@1'] += 1
        with pytest.raises(ValueError, match='selection'):verify_student_best(corrupt)
        from src.student.artifacts import validate_training_complete
        complete_config, complete_best = validate_training_complete(run)
        assert complete_best['best_epoch'] == 1 and complete_config['img_size'] == 224
        assert not any((run/name).exists() for name in ('run_config.json','epoch_metrics.json','last_model.pth','best_metrics.json'))
        from src.student.evaluate_best import main as evaluate_best
        from src.evaluation import evaluate as unified
        def fake_unified(argv):
            output=Path(argv[argv.index('--output-dir')+1])
            checkpoint=argv[argv.index('--checkpoint')+1]
            payload=dict(model_type='student',checkpoint=checkpoint,u1652_eval_batch_size=32,
                         protocol={'BEST_MODEL_RELOAD_CONSISTENCY':'PASS'})
            (output/'test_1652.json').write_text(json.dumps(payload))
        monkeypatch.setattr(unified,'main',fake_unified)
        evaluate_best(['--run-dir',str(run),'--dataset','u1652','--device','cpu'])
        assert (run/'test_1652_best.json').is_file()
        (run / 'train.log').write_text('fixture')
        from src.student.artifacts import RESULT_FILES, file_sha256, package_results, write_json
        for name in RESULT_FILES[1:]:
            write_json(run/name,dict(checkpoint_sha256=file_sha256(run/'best_model.pth')))
        packaged=package_results(run)
        import tarfile
        with tarfile.open(packaged['PACKAGE']) as archive:
            names=set(archive.getnames())
        assert names==set(('train.log',)+RESULT_FILES+('RESULT_MANIFEST.txt',))
        assert 'LAST_MODEL' not in packaged
    with patch("src.student.random_structure.generate_random_basis",
               side_effect=AssertionError("checkpoint reload must use stored basis")):
        _, checkpoint_supervision, checkpoint_gate = restore_training_checkpoint(
            run / "best_model.pth", cfg)
    assert checkpoint_supervision.random_basis_generation_count == 0
    assert torch.equal(checkpoint_supervision.random32_basis, basis)
    assert torch.equal(checkpoint_gate.d, gate.d)
    if size == 224:
        source = torch.load(
            ROOT / "src/checkpoint/student/R224/ASSETS_E3_KDSAM/top128_random32.pt",
            map_location="cpu", weights_only=True)
        assert torch.equal(supervision.teacher_mean, source["teacher_mean"])
        assert torch.equal(supervision.top32_basis, source["top128_basis"])


def test_formal_seeds_are_distinct_and_generator_shared():
    a, b = (formal_config(size) for size in (224, 256))
    assert a["random_basis_seed"] != b["random_basis_seed"]
    from src.student.random_structure import generate_random_basis
    x = generate_random_basis(a["random_basis_seed"])
    y = generate_random_basis(b["random_basis_seed"])
    assert not torch.equal(x, y)
    assert tensor_sha256(x) != tensor_sha256(y)



@pytest.mark.parametrize("size", [224, 256])
def test_formal_config_rejects_legacy_random_ownership(size):
    cfg = formal_config(size)
    with pytest.raises(ValueError):
        validate_config(dict(cfg, random_basis_mode="asset"))
    with pytest.raises(ValueError):
        validate_config(dict(cfg, random_A_seed=20260808))
    with pytest.raises(ValueError):
        validate_config(dict(cfg, original_stst_asset="/historical/random.pt"))
