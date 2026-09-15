"""Tiny descriptor-only backward smoke; never optimizer.step or benchmark evaluation."""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .artifacts import ROOT, write_json, file_sha256, deployment_state_dict
from .model import StudentModel
from .part1 import PartISupervision
from .part2 import install_residual_top, trainable_count, matched_hidden_dim, ALPHA_INIT
from .runtime import _seed_all
from .train import StudentTrainingModel


def norm(parameters):
    grads = [p.grad.float() for p in parameters if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Missing/nonfinite gradient")
    return sum(float(g.square().sum()) for g in grads)**.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    args = parser.parse_args()
    out = Path(args.audit_dir).resolve()
    if "_AUDITS" not in out.parts:
        raise ValueError("Smoke is restricted to an audit directory")
    if (out/"smoke_report.json").exists():
        raise FileExistsError(out/"smoke_report.json")
    cfg = json.loads((ROOT/"configs/student/certified_r224/p1_t128_r32_s0.json").read_text())
    dist = json.loads((out/"input_distribution.json").read_text())
    calibration = torch.load(out/"diagnostic_inputs.pt",map_location="cpu",weights_only=True).cuda()
    _seed_all(0)
    torch.set_num_threads(4)
    student = StudentModel(ckpt_path=cfg["student_pretrained"])
    reference = PartISupervision(cfg["stst_asset"],cfg["original_stst_asset"],
        file_sha256(cfg["middle_checkpoint"]),128,"single32").cuda().bfloat16()
    target = F.normalize(torch.randn(16,768,device="cuda",dtype=torch.float32),dim=1)
    reports = {}
    for kind in ["linear","rmlp","rkan"]:
        supervision = copy.deepcopy(reference)
        if kind != "linear":
            install_residual_top(supervision,kind,calibration)
        random_before = {k:v.detach().clone() for k,v in supervision.projector_random.state_dict().items()}
        x = calibration[:16].detach().clone().requires_grad_()
        with torch.autocast("cuda",enabled=False):
            loss,audit = supervision(x,target,8)
        assert torch.isfinite(loss) and loss.dtype==torch.float32
        loss.backward()
        assert torch.isfinite(x.grad).all()
        grads = dict(top_linear=norm(supervision.projector_top.linear.parameters()),
                     random_linear=norm(supervision.projector_random.parameters()),
                     descriptor=float(x.grad.norm()))
        init = None
        if kind != "linear":
            grads.update(residual=norm(supervision.projector_top.residual.parameters()),
                         alpha=norm([supervision.projector_top.alpha]))
            init = supervision.projector_top.initialization_audit
            assert abs(init["gated_residual_base_norm_ratio"]-ALPHA_INIT)<1e-7
        assert all(v>0 for v in grads.values())
        assert all(torch.equal(v,supervision.projector_random.state_dict()[k]) for k,v in random_before.items())
        for a,b in zip(reference.projector_random(x),supervision.projector_random(x)):
            assert torch.equal(a,b)
        assert all(torch.equal(v,supervision.projector_top.linear.state_dict()[k])
                   for k,v in reference.projector_top.linear.state_dict().items())
        bare = deployment_state_dict(StudentTrainingModel(student,supervision))
        assert set(bare)==set(student.state_dict())
        assert all(torch.equal(v,student.state_dict()[k]) for k,v in bare.items())
        top = supervision.projector_top
        reports[kind] = dict(top_head_params=trainable_count(top),
            all_projector_params=trainable_count(supervision),
            residual_params=0 if kind=="linear" else trainable_count(top.residual),
            alpha_params=0 if kind=="linear" else 1, forward_finite=True,backward_finite=True,
            gradient_norms=grads, initialization=init,random_unchanged=True,
            base_linear_unchanged=True,deployment_strip_pass=True,
            optimizer_step_calls=0,benchmark_evaluations=0)
    k,m=reports["rkan"]["residual_params"],reports["rmlp"]["residual_params"]
    report=dict(smokes=reports,KAN_GRID_RANGE=dist["grid_range"],KAN_GRID_SIZE=5,KAN_SPLINE_ORDER=3,
        P_LINEAR_BASE=reports["linear"]["top_head_params"],P_KAN_RESIDUAL=k,
        P_TOTAL_RKAN_TOP_HEAD=reports["rkan"]["top_head_params"],
        MLP_HIDDEN_DIM=matched_hidden_dim(k),P_MLP_RESIDUAL=m,
        PARAM_MISMATCH_PERCENT=100*abs(k-m)/k,ALPHA_INIT=ALPHA_INIT,ALPHA_LEARNABLE=True,
        KAN_FORWARD_PASS=True,KAN_BACKWARD_PASS=True,KAN_CUDA_PASS=True,
        MLP_FORWARD_PASS=True,MLP_BACKWARD_PASS=True,DEPLOYMENT_STRIP_PASS=True,
        BASELINE_REGRESSION_PASS=True,formal_training_started=False,
        source_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in
                       [Path(__file__),ROOT/"src/student/part2.py"]})
    assert all(file_sha256(cfg[k])==h for k,h in dist["protected_sha256"].items())
    write_json(out/"smoke_report.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
