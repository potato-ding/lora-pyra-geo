import copy
import pytest
from src.student.formal_config import ROOT, load_formal_config
from src.student.ablation_config import validate_config
from src.student.formal_train import sam_enabled

def test_d2_no_sam_only_declared_difference():
    _, original = load_formal_config(ROOT / "configs/student/d2_top256_random64_sam_m3_256.json")
    _, control = load_formal_config(ROOT / "configs/student/d2_top256_random64_no_sam_m3_256.json")
    differences = {k for k in original.keys() | control.keys() if original.get(k) != control.get(k)}
    assert differences == {"sam", "student_sam_control", "experiment_id", "output_dir"}
    assert sam_enabled(original) and not sam_enabled(control)
    for field, value in [("seed", 1), ("student_sam_control", "other"), ("sam", {"enabled": True})]:
        bad = copy.deepcopy(control)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_config(bad)
