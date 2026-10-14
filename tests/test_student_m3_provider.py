"""Explicit M3 variants preserve Student mathematics and reject wrong providers."""
import json
from pathlib import Path
import pytest
from src.student.formal_config import load_formal_config
ROOT = Path(__file__).resolve().parents[1]
@pytest.mark.parametrize("method", ["s1_tsd", "s2_adsd"])
def test_m3_variant_preserves_math(method):
    _, cfg = load_formal_config(ROOT / f"configs/student/{method}_m3_256.json")
    _, base = load_formal_config(ROOT / f"configs/student/{method}_m4_256.json")
    bindings = {"middle_provider", "middle_checkpoint", "middle_config", "supervision_asset", "top_calibration", "output_dir"}
    assert {k:v for k,v in cfg.items() if k not in bindings} == {k:v for k,v in base.items() if k not in bindings}
    assert cfg["middle_provider"] == "M3"
@pytest.mark.parametrize("provider", [None, "M2", "M4"])
def test_m3_rejects_wrong_provider(monkeypatch, provider):
    path = ROOT / "configs/student/s2_adsd_m3_256.json"
    cfg = json.loads(path.read_text()); cfg["middle_provider"] = provider
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: json.dumps(cfg))
    with pytest.raises(ValueError, match="middle_provider=M3"):
        load_formal_config(path)
