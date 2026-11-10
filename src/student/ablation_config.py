"""Six explicit R256 component controls against the immutable final M3 S3 run."""
import json
from pathlib import Path
from .formal_config import ROOT, BASE_KEYS
from .shared_subspaces import validate_reference_config

VARIANTS = {
    "b2_full768": {"target": "full768", "full_head": "linear", "branch_weight": 1.0},
    "b3_top128": {"target": "top128", "top_head": "residual_mlp", "branch_weight": 1.0},
    "b4_random32": {"target": "random32", "random_head": "linear", "branch_weight": 1.0},
    "b5_dual_linear": {"target": "dual", "top_head": "linear", "random_head": "linear", "allocation": "learnable"},
    "b6_dual_rmlp": {"target": "dual", "top_head": "residual_mlp", "random_head": "residual_mlp", "allocation": "learnable"},
    "b7_fixed_equal": {"target": "dual", "top_head": "residual_mlp", "random_head": "linear", "allocation": "fixed", "w_top": 1.0, "w_random": 1.0},
}
EXTRA_VARIANTS = {
    "b1_m0_provider": {"target":"dual", "top_head":"residual_mlp", "random_head":"linear", "allocation":"learnable"},
    "b8_no_sam": {"target":"dual", "top_head":"residual_mlp", "random_head":"linear", "allocation":"learnable"},
}
FULL_VARIANTS = {
    "c1_full768_rmlp": {"target":"full768", "full_head":"residual_mlp", "residual_hidden_dim":512,"branch_weight":1.0},
    "c2_full768_rmlp": {"target":"full768", "full_head":"residual_mlp", "residual_hidden_dim":512,"branch_weight":1.0},
}
from .dimension_config import DIM_VARIANTS
ALL_VARIANTS = {**VARIANTS, **EXTRA_VARIANTS, **FULL_VARIANTS, **DIM_VARIANTS}
REFERENCE = "S3-256-M3"
BASE_CONFIG = ROOT / "configs/student/s3_sam_adsd_shared_m3_256.json"
REPEAT_VARIANTS = {"c2_full768_rmlp", "d1_top256_random32", "d2_top256_random64"}


def validate_config(cfg):
    if "provider_control" in cfg:
        from .provider_dimensions import validate_config as validate_provider
        return validate_provider(cfg)
    key = cfg.get("ablation", {}).get("variant")
    if key not in ALL_VARIANTS:
        raise ValueError("Unknown Student ablation")
    base = json.loads(BASE_CONFIG.read_text())
    expected = dict(base)
    expected.update(experiment_id=key.upper().replace("_", "-") + "-SAM-R256",
                    output_dir=cfg.get("output_dir"),
                    ablation={"variant": key, **ALL_VARIANTS[key]})
    if key in REPEAT_VARIANTS:
        if type(cfg.get("seed")) is not int or cfg["seed"] not in (0, 1, 2):
            raise ValueError("Declared repeat seed must be 0, 1 or 2")
        expected["seed"] = cfg["seed"]
    if key in DIM_VARIANTS:
        expected["dimension_asset"] = str(ROOT/"src/checkpoint/student/R256/middle_assets/M3-256/dimension_extension/manifest.json")
    if "random_basis_control" in cfg:
        from .dimension_random import validate_control
        validate_control(cfg)
        expected["random_basis_control"] = cfg["random_basis_control"]
    expected.pop("distillation")
    expected["distillation"] = {"type": "component_ablation", "weight": .2, "warmup_epochs": 5}
    if key == "c2_full768_rmlp":
        expected["distillation"]["weight"] = .4
    if key == "b1_m0_provider":
        expected.update(middle_provider="M0",
            middle_checkpoint=str(ROOT/"src/checkpoint/middle_teacher/R256/M0-256/best_model.pth"),
            middle_config=str(ROOT/"configs/middle_teacher/m0_infonce_256.json"),
            supervision_asset=str(ROOT/"src/checkpoint/student/R256/middle_assets/M0-256/top128/manifest.json"),
            top_calibration=str(ROOT/"src/checkpoint/student/R256/middle_assets/M0-256/top_calibration.pt"))
        expected["random_reference"] = expected.pop("subspace_reference")
    elif key == "b8_no_sam":
        expected["experiment_id"] = "B8-NO-SAM-R256"
        expected["sam"] = {"enabled":False}
    if "student_sam_control" in cfg:
        if (key != "d2_top256_random64" or cfg["student_sam_control"] != "disabled"
                or cfg.get("seed") != 0 or "random_basis_control" in cfg):
            raise ValueError("No-SAM dimension control requires original D2 seed0 assets")
        expected["student_sam_control"] = "disabled"
        expected["sam"] = {"enabled": False}
        expected["experiment_id"] = "D2-TOP256-RANDOM64-NO-SAM-R256"
    if "secondary_subspace" in cfg:
        if (key != "d2_top256_random64" or cfg.get("seed") != 0
                or cfg["secondary_subspace"] != "same_pca_columns_257_320"
                or "random_basis_control" in cfg or "student_sam_control" in cfg):
            raise ValueError("PCA tail control requires seed0 D2 with SAM")
        expected["secondary_subspace"] = "same_pca_columns_257_320"
        expected["experiment_id"] = "D2-TOP256-PCA257-320-SAM-R256"
    if cfg != expected:
        raise ValueError("Ablation must differ from final S3 only in the declared component and destination")
    if not isinstance(cfg["output_dir"], str) or not cfg["output_dir"]:
        raise ValueError("Ablation output directory missing")
    ref=cfg.get("subspace_reference",cfg.get("random_reference"))
    validate_reference_config(ref)
    if Path(ref["checkpoint"]).parent.name != REFERENCE:
        raise ValueError("Ablations must reuse final S3-256-M3, never another random replicate")
    return cfg


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    validate_config(cfg)
    if "provider_control" in cfg:
        if path.parent != BASE_CONFIG.parent or path.name != "d2_" + cfg["provider_control"] + "_256.json":
            raise ValueError("Provider filename mismatch")
        return path, cfg
    suffix = "_sam_m0_256.json" if cfg["ablation"]["variant"]=="b1_m0_provider" else ("_m3_256.json" if cfg["ablation"]["variant"]=="b8_no_sam" else "_sam_m3_256.json")
    if "student_sam_control" in cfg:
        suffix = "_no_sam_m3_256.json"
    repeat_suffix = "_seed" + str(cfg["seed"]) if cfg["seed"] in (1, 2) and cfg["ablation"]["variant"] in REPEAT_VARIANTS else ""
    if "random_basis_control" in cfg:
        repeat_suffix = "_randombasis" + str(cfg["random_basis_control"]["replicate"])
    if "secondary_subspace" in cfg:
        repeat_suffix = "_pca257_320"
    expected_name = cfg["ablation"]["variant"] + suffix.removesuffix(".json") + repeat_suffix + ".json"
    if path.parent != BASE_CONFIG.parent or path.name != expected_name:
        raise ValueError("Ablation filename mismatch")
    return path, cfg


def reference_config(cfg):
    validate_config(cfg)
    base = json.loads(BASE_CONFIG.read_text())
    base["output_dir"] = cfg["output_dir"]
    return base
