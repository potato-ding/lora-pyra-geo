"""Release helper preflights run without model inference or filesystem output."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_efficiency_default_jobs_use_explicit_current_student_providers():
    suite = load_script("run_efficiency_suite")
    jobs = suite.select_entries()
    assert len(jobs) == len({job["label"] for job in jobs}) == 14
    students = [job for job in jobs if job["model"] == "student"]
    assert {Path(job["checkpoint"]).parent.name for job in students} == {"S3-224-M3", "S3-256-M3"}
    assert all(job["model"] != "dmnil" for job in jobs)
    assert [job["model"] for job in suite.select_entries(models=["dmnil"])] == ["dmnil"]


def test_efficiency_validates_all_inputs_before_writing_or_launching(tmp_path):
    suite = load_script("run_efficiency_suite")
    out = tmp_path / "new_output"
    weight = tmp_path / "model.pth"
    weight.write_bytes(b"preflight only")
    job = {"label": "Student", "model": "student", "checkpoint": str(weight)}
    suite.validate_jobs([job], out)
    assert not out.exists()
    weight.unlink()
    with pytest.raises(FileNotFoundError):
        suite.validate_jobs([job], out)
    assert not out.exists()
    weight.write_bytes(b"preflight only")
    out.mkdir()
    (out / "Student.json").write_text("existing result")
    with pytest.raises(FileExistsError):
        suite.validate_jobs([job], out)
    assert (out / "Student.json").read_text() == "existing result"


def test_efficiency_rejects_ambiguous_job_selection():
    suite = load_script("run_efficiency_suite")
    for kwargs in ({"own_only": True, "models": ["student"]}, {"models": ["student", "student"]}, {"models": ["unknown"]}):
        with pytest.raises(ValueError):
            suite.select_entries(**kwargs)


@pytest.mark.parametrize("size", [224, 256])
def test_diagnostic_preflight_resolves_current_student_without_gpu(tmp_path, monkeypatch, size):
    diagnostic = load_script("diagnose_student_mean")
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    weight = tmp_path / f"src/checkpoint/student/R{size}/S3-{size}-M3/best_model.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"preflight only")
    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not load a model")
    monkeypatch.setattr(diagnostic, "load_encoder", forbidden)
    out = tmp_path / "diagnostic_output"
    diagnostic.main(["student", str(out), "--image-size", str(size), "--check"])
    assert not out.exists()
    out.mkdir()
    result = out / "student.json"
    result.write_text("existing result")
    with pytest.raises(FileExistsError):
        diagnostic.parse_args(["student", str(out), "--image-size", str(size), "--check"])
    assert result.read_text() == "existing result"


def test_explicit_missing_dmnil_fails_preflight(tmp_path, monkeypatch):
    suite = load_script("run_efficiency_suite")
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="DMNIL"):
        suite.validate_jobs(suite.select_entries(models=["dmnil"]), tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_dependency_manifests_agree_on_metric_and_efficiency_requirements():
    requirements = (ROOT / "requirements.txt").read_text()
    environment = (ROOT / "environment.yml").read_text()
    for requirement in ("scikit-learn==1.7.2", "fvcore==0.1.5.post20221221"):
        assert requirement in requirements and requirement in environment


@pytest.mark.parametrize("kind,folder,bundle,weight", [
    ("muse", "MuSe-Net", "weights/three_view_test", "net_209.pth"),
    ("ccr", "CCR", "weights/extracted/model/CCR_Model_University", "net_194.pth"),
])
def test_comparison_preflight_matches_nested_adapter_layout(tmp_path, monkeypatch, kind, folder, bundle, weight):
    suite = load_script("run_efficiency_suite")
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    base = tmp_path / "src/checkpoint/others" / folder
    (base / "source").mkdir(parents=True)
    checkpoint = base / bundle / weight
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"preflight only")
    if kind == "muse":
        (checkpoint.parent / "opts.yaml").write_text("{}")
    else:
        (base / "source/compatibility_adapter.py").write_text("# adapter")
    assert suite.comparison_input_paths(kind)[0][0] == checkpoint
    suite.validate_jobs(suite.select_entries(models=[kind]), tmp_path / "output")
    if kind == "muse":
        (base / "weights/three_view_ambiguous").mkdir()
        with pytest.raises(FileNotFoundError, match="exactly one"):
            suite.validate_jobs(suite.select_entries(models=[kind]), tmp_path / "output")
