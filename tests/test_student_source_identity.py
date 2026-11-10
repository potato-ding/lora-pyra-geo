"""Student-specific source records cover SAM without changing frozen stages."""
import hashlib
import json
from pathlib import Path

import pytest

from src.source_contract import source_identity
from src.student import formal_train

ROOT = Path(__file__).resolve().parents[1]
SAM_SOURCE = "src/student/distill_sam.py"
S3_CONFIGS = sorted((ROOT / "configs/student").glob("s3_*.json"))


@pytest.mark.parametrize("path", S3_CONFIGS, ids=lambda p: p.stem)
def test_every_s3_config_records_actual_sam_source(path):
    config = json.loads(path.read_text())
    actual = formal_train.collect_student_source_identity(config)
    assert actual[SAM_SOURCE] == hashlib.sha256((ROOT / SAM_SOURCE).read_bytes()).hexdigest()
    if "subspace_reference" in config or "random_basis_replicate" in config:
        assert "src/student/shared_subspaces.py" in actual
    if config.get("teacher_provider") == "T0":
        for name in ("direct_teacher_assets.py", "direct_teacher_distillation.py", "shared_subspaces.py"):
            assert "src/student/" + name in actual


@pytest.mark.parametrize("method", ["S0-INFONCE", "S1-TSD", "S2-ADSD", "S4-ADSD-SAM-MIDDLE"])
def test_non_sam_student_source_record_is_unchanged(method):
    config = {"experiment_id": method + "-R256"}
    expected = source_identity("m2s", gbw=method in ("S2-ADSD", "S4-ADSD-SAM-MIDDLE"))
    assert formal_train.collect_student_source_identity(config) == expected


def test_sam_source_content_change_is_recorded_and_missing_file_fails(tmp_path, monkeypatch):
    config = {"experiment_id": "S3-SAM-ADSD-R256"}
    frozen = {stage: source_identity(stage) for stage in ("teacher", "t2m")}
    sam_file = tmp_path / SAM_SOURCE
    sam_file.parent.mkdir(parents=True)
    sam_file.write_bytes(b"sam implementation version one")
    monkeypatch.setattr(formal_train, "ROOT", tmp_path)
    first = formal_train.collect_student_source_identity(config)
    sam_file.write_bytes(b"sam implementation version two")
    second = formal_train.collect_student_source_identity(config)
    assert first[SAM_SOURCE] != second[SAM_SOURCE]
    assert {k for k in first if first[k] != second[k]} == {SAM_SOURCE}
    assert {stage: source_identity(stage) for stage in frozen} == frozen
    sam_file.unlink()
    with pytest.raises(FileNotFoundError):
        formal_train.collect_student_source_identity(config)
