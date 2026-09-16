"""Fixed P2 S0 dispatch and precision grouping; existing P1 execution is untouched."""
import json
from pathlib import Path
import torch
from .artifacts import ROOT, file_sha256
from .part2 import install_residual_top, trainable_count

KINDS = {"residual_mlp": ("rmlp", "P2-TOP-RMLP-S0"),
         "residual_kan": ("rkan", "P2-TOP-RKAN-S0")}
AUDIT = ROOT/"src/checkpoint/student/CERTIFIED_R224/_AUDITS/P2_RESIDUAL_IMPLEMENTATION"


def validate_config(cfg):
    interface = cfg.get("top_interface", "linear")
    if interface.startswith('factorial_'):
        from .part2_factorial import validate_config as validate_factorial
        return validate_factorial(cfg)
    if interface == "linear":
        if any(k.startswith("p2_") for k in cfg):
            raise ValueError("P2 metadata requires a residual interface")
        return False
    if interface not in KINDS:
        raise ValueError("Only the two frozen P2 S0 interfaces are supported")
    reference = json.loads((ROOT/"configs/student/certified_r224/p1_t128_r32_s0.json").read_text())
    allowed = {"experiment_name", "output_dir", "sealed_provenance_file",
               "top_interface", "p2_calibration_path", "p2_calibration_sha256"}
    if {k:v for k,v in cfg.items() if k not in allowed} != {k:v for k,v in reference.items() if k not in allowed}:
        raise ValueError("P2 must inherit every reference training field exactly")
    name = KINDS[interface][1]
    if cfg.get("experiment_name") != name or Path(cfg["output_dir"]).resolve() != ROOT/"src/checkpoint/student/CERTIFIED_R224"/name:
        raise ValueError("P2 run/output name mismatch")
    if type(cfg.get("seed")) is not int or cfg["seed"] != 0:
        raise ValueError("P2 is seed0 only")
    if Path(cfg["p2_calibration_path"]).resolve() != AUDIT/"diagnostic_inputs.pt":
        raise ValueError("P2 must use the frozen TRAIN calibration")
    if file_sha256(cfg["p2_calibration_path"]) != cfg["p2_calibration_sha256"]:
        raise ValueError("P2 calibration SHA mismatch")
    return True


def prepare_top(supervision, cfg):
    if cfg.get('top_interface', '').startswith('factorial_'):
        from .part2_factorial import prepare
        return prepare(supervision, cfg)
    if cfg.get("top_interface", "linear") == "linear":
        return
    validate_config(cfg)
    calibration = torch.load(cfg["p2_calibration_path"], map_location="cpu", weights_only=True)
    assert calibration.shape == (768,512) and torch.isfinite(calibration).all()
    # Match frozen component audit: canonical base storage was BF16 when calibrated.
    supervision.bfloat16()
    install_residual_top(supervision, KINDS[cfg["top_interface"]][0], calibration)


def prepare_precision_groups(model, optimizer, cfg):
    if cfg.get("top_interface", "linear") == "linear":
        return
    if optimizer.state:
        raise RuntimeError("Precision grouping must precede the first optimizer step")
    model.bfloat16()  # Same conversion DeepSpeed performs; FP32 residuals opt out.
    groups = []
    for group in optimizer.param_groups:
        by_dtype = {}
        for parameter in group["params"]:
            by_dtype.setdefault(parameter.dtype, []).append(parameter)
        for dtype, parameters in by_dtype.items():
            groups.append(dict(group, params=parameters,
                               name=group["name"]+"_"+str(dtype)))
    # Preserve the very same AdamW object and every original group's settings.
    optimizer.param_groups[:] = groups


def assert_precision(engine):
    supervision = engine.module.stst
    if hasattr(supervision, 'factorial_interface'):
        from .part2_factorial import assert_precision as assert_factorial_precision
        return assert_factorial_precision(engine)
    if not hasattr(supervision.projector_top, "residual"):
        return
    assert all(p.dtype==torch.bfloat16 for p in engine.module.student.parameters())
    assert all(p.dtype==torch.bfloat16 for p in supervision.projector_top.linear.parameters())
    assert all(p.dtype==torch.bfloat16 for p in supervision.projector_random.parameters())
    assert all(p.dtype==torch.float32 for p in supervision.projector_top.residual.parameters())
    assert supervision.projector_top.alpha.dtype==torch.float32


def metadata(supervision):
    if hasattr(supervision, 'factorial_interface'):
        from .part2_factorial import metadata as factorial_metadata
        return factorial_metadata(supervision)
    top = supervision.projector_top
    return dict(part="Part-II", research_axis="top_alignment_interface",
        training_only_head_params=trainable_count(supervision),
        p2_top_params=trainable_count(top),p2_residual_params=trainable_count(top.residual),
        p2_alpha_init=.001,p2_alpha_learnable=True,p2_alpha_weight_decay=0.,
        p2_grid_range=[-.17,.17],p2_grid_size=5,p2_spline_order=3,
        p2_mlp_hidden_dim=920,p2_initialization=top.initialization_audit,
        p2_parameter_storage="Student/base/Random BF16; residual/gate/grid FP32",
        p2_optimizer_grouping="same AdamW decay/no-decay policy partitioned by dtype",
        p2_projector_compute="FP32",REFERENCE_USES_DEEPSPEED=True,P2_USES_DEEPSPEED=True)


def log_values(supervision):
    if hasattr(supervision, 'factorial_interface'):
        from .part2_factorial import log_values as factorial_log_values
        return factorial_log_values(supervision)
    top = supervision.projector_top
    return dict(p2_alpha=float(top.alpha.detach()))
