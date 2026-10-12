"""M4 provider variants keep Student objectives unchanged and bind assets strictly."""
import json
from pathlib import Path
import pytest
from src.student.formal_config import load_formal_config

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("s1_tsd", "s2_adsd", "s3_sam_adsd")

@pytest.mark.parametrize("method", METHODS)
def test_m4_variant_changes_only_provider_and_output(method):
    _, base = load_formal_config(ROOT / f"configs/student/{method}_256.json")
    _, cfg = load_formal_config(ROOT / f"configs/student/{method}_m4_256.json")
    bindings = {"middle_provider", "middle_checkpoint", "middle_config",
                "supervision_asset", "top_calibration", "output_dir"}
    assert {k:v for k,v in cfg.items() if k not in bindings} == {
        k:v for k,v in base.items() if k not in bindings}
    assert cfg["middle_provider"] == "M4"
    assert "M4-256" in cfg["middle_checkpoint"]

@pytest.mark.parametrize("provider", [None, "M2", "M3", "M5"])
def test_m4_variant_rejects_wrong_provider(monkeypatch, provider):
    path = ROOT / "configs/student/s2_adsd_m4_256.json"
    cfg = json.loads(path.read_text())
    cfg["middle_provider"] = provider
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: json.dumps(cfg))
    with pytest.raises(ValueError, match="middle_provider=M4"):
        load_formal_config(path)

def test_legacy_config_cannot_silently_override_provider(monkeypatch):
    path = ROOT / "configs/student/s2_adsd_256.json"
    cfg = json.loads(path.read_text())
    cfg["middle_provider"] = "M4"
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: json.dumps(cfg))
    with pytest.raises(ValueError, match="keys mismatch"):
        load_formal_config(path)
