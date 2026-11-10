"""VkD feature-distillation adaptation: fixed M3 retrieval protocol, no subspace assets."""
import json
from pathlib import Path
from .formal_config import ROOT

WEIGHTS = {"w05": .5, "w10": 1., "w20": 2.}

def validate_config(cfg):
    base = json.loads((ROOT / "configs/student/s3_sam_adsd_shared_m3_256.json").read_text())
    for key in ("subspace_reference", "supervision_asset", "top_calibration"):
        base.pop(key)
    d = cfg.get("distillation", {})
    weight = d.get("weight")
    seed = cfg.get("seed")
    if type(weight) not in (int, float) or weight not in WEIGHTS.values():
        raise ValueError("VkD weight must be predeclared .5/1/2")
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("VkD training seed must be 0/1/2")
    base.update(experiment_id="VKD-SAM-R256", seed=seed, output_dir=cfg.get("output_dir"),
        distillation={"type":"VkD", "weight":weight, "warmup_epochs":5,
            "student_feature":"shared_bn_before_l2", "teacher_feature":"frozen_retrieval_descriptor",
            "teacher_normalization":"layer_norm", "layer_norm_eps":1e-5,
            "loss":"smooth_l1", "smooth_l1_beta":1., "reduction":"mean",
            "projector":"orthogonal_linear", "bias":False,
            "orthogonal_map":"householder", "use_trivialization":True,
            "projector_precision":"float32", "projector_initialization":"training_seed"})
    if cfg != base:
        raise ValueError("VkD config changed outside declared method, seed, weight or destination")
    if cfg["output_dir"] is not None and (not isinstance(cfg["output_dir"],str) or not cfg["output_dir"]):
        raise ValueError("VkD output_dir must be nonempty or null")
    return cfg

def load_config(path):
    path=Path(path).resolve()
    cfg=json.loads(path.read_text());validate_config(cfg)
    tag=next(k for k,v in WEIGHTS.items() if v==cfg["distillation"]["weight"])
    expected=f"vkd_sam_m3_{tag}_256_seed{cfg['seed']}.json"
    if path.parent != ROOT / "configs/student" or path.name != expected:
        raise ValueError("VkD config filename mismatch")
    return path,cfg
