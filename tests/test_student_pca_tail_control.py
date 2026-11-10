import copy
import pytest
import torch
from src.student.formal_config import ROOT, load_formal_config
from src.student.ablation_config import validate_config
from src.student.dimension_random import knowledge_tensors, basis_seed


def test_same_pca_tail_and_fixed_initial_heads():
    from src.student.formal_train import validate_assets
    from src.student.component_ablation import prepare

    _, original = load_formal_config(ROOT / "configs/student/d2_top256_random64_sam_m3_256.json")
    _, control = load_formal_config(ROOT / "configs/student/d2_top256_random64_sam_m3_256_pca257_320.json")
    assert {k for k in original.keys() | control.keys() if original.get(k) != control.get(k)} == {
        "secondary_subspace", "experiment_id", "output_dir"
    }
    source, _ = knowledge_tensors(original)
    target, _ = knowledge_tensors(control)
    assert torch.equal(target["top_basis"], source["top_basis"])
    assert torch.equal(target["random_basis"], source["top_basis"][:,256:320])
    assert (target["top_basis"][:,:256].T @ target["random_basis"]).abs().max() < 1e-5
    assert basis_seed(control) is None
    cal = torch.load(original["top_calibration"],map_location="cpu",weights_only=True)
    rng = torch.get_rng_state()
    sup_a, gate_a, _ = prepare(original,validate_assets(original),cal)
    torch.set_rng_state(rng)
    sup_b, gate_b, _ = prepare(control,validate_assets(control),cal)
    for name, tensor in sup_a.state_dict().items():
        if name != "random_basis":
            assert torch.equal(tensor,sup_b.state_dict()[name]), name
    assert torch.equal(gate_a.d,gate_b.d)
    z = torch.randn(8,512,requires_grad=True)
    loss, _ = sup_b(z,torch.randn(8,768),4)
    loss.backward()
    assert torch.isfinite(loss) and z.grad.isfinite().all() and z.grad.abs().sum() > 0


def test_pca_tail_control_fail_closed():
    _, cfg = load_formal_config(ROOT / "configs/student/d2_top256_random64_sam_m3_256_pca257_320.json")
    for name,value in [("seed",1),("secondary_subspace","other"),("student_sam_control","disabled")]:
        bad=copy.deepcopy(cfg)
        bad[name]=value
        with pytest.raises(ValueError):
            validate_config(bad)
