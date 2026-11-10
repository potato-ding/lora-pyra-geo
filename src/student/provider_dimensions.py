"""Provider controls for the final Top256/Random64 Student."""
import json
import copy
from pathlib import Path
import torch
from .formal_config import ROOT
from .artifacts import file_sha256
from .subspace_utils import tensor_sha256

CONTROLS = {
    "a_t0": ("T0", False),
    "b_m0": ("M0", False),
    "c_t0_hnd": ("T0", True),
}


def original_config():
    return json.loads((ROOT / "configs/student/d2_top256_random64_sam_m3_256.json").read_text())


def validate_config(cfg):
    control = cfg.get("provider_control")
    if control not in CONTROLS:
        raise ValueError("Unknown provider control")
    expected = original_config()
    provider, _ = CONTROLS[control]
    checkpoint = ROOT / ("src/checkpoint/teacher/T0-256/best_model.pth" if provider == "T0"
                         else "src/checkpoint/middle_teacher/R256/M0-256/best_model.pth")
    parent = "teacher_assets/T0-256" if provider == "T0" else "middle_assets/M0-256"
    expected.update(supervision_provider_checkpoint=str(checkpoint),
                    provider_supervision_asset=str(ROOT / "src/checkpoint/student/R256" / parent / "dimension256_64/manifest.json"))
    expected.update(provider_control=control, output_dir=cfg.get("output_dir"),
                    experiment_id="D2-" + control.upper().replace("_", "-") + "-SAM-R256")
    if cfg != expected or not cfg["output_dir"]:
        raise ValueError("Provider control changed unrelated D2 settings")
    return cfg


def load_knowledge(cfg):
    validate_config(cfg)
    provider, _ = CONTROLS[cfg["provider_control"]]
    parent = "teacher_assets/T0-256" if provider == "T0" else "middle_assets/M0-256"
    path = ROOT / "src/checkpoint/student/R256" / parent / "dimension256_64"
    metadata = json.loads((path / "manifest.json").read_text())
    if (metadata["schema"], metadata["provider"], metadata["resolution"], metadata["split"],
        metadata["rows"], metadata["identities_per_view"]) != (
            "PROVIDER256_64_V1", provider, 256, "train", 1402, 701):
        raise ValueError("Provider asset protocol mismatch")
    checkpoint = ROOT / ("src/checkpoint/teacher/T0-256/best_model.pth" if provider == "T0"
                         else "src/checkpoint/middle_teacher/R256/M0-256/best_model.pth")
    if metadata["checkpoint"] != str(checkpoint) or file_sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Provider checkpoint SHA mismatch")
    if file_sha256(path.parent / "top128/manifest.json") != metadata["old_manifest_sha256"]:
        raise ValueError("Provider source manifest changed")
    dim = 4096 if provider == "T0" else 768
    tensors = {}
    for name, shape in (("teacher_mean", (dim,)), ("top_basis", (dim,256)), ("random_basis", (dim,64))):
        t = torch.load(path / (name + ".pt"), map_location="cpu", weights_only=True)
        if t.shape != shape or t.dtype != torch.float32 or not torch.isfinite(t).all():
            raise ValueError("Provider tensor shape/precision mismatch")
        if tensor_sha256(t) != metadata["tensor_sha256"][name]:
            raise ValueError("Provider tensor SHA mismatch")
        if name != "teacher_mean" and (t.T @ t - torch.eye(t.shape[1])).abs().max() > 1e-5:
            raise ValueError("Provider basis not orthonormal")
        tensors[name] = t
    return tensors, metadata, path


def validate_assets(cfg):
    from .formal_train import validate_assets as validate_original
    assets = validate_original(original_config())
    _, metadata, path = load_knowledge(cfg)
    assets["supervision_provider_checkpoint"] = metadata["checkpoint_sha256"]
    assets["provider_dimension_asset"] = file_sha256(path / "manifest.json")
    return assets


def load_provider(cfg, device):
    from src.evaluation.model_loader import load_encoder
    provider, _ = CONTROLS[cfg["provider_control"]]
    _, metadata, _ = load_knowledge(cfg)
    if provider == "T0":
        from src.training.teacher.formal_reload import load_teacher_v3
        from .direct_teacher_distillation import FrozenTeacherChunks
        encoder, _ = load_teacher_v3(metadata["checkpoint"],device=device,image_size=256)
        return FrozenTeacherChunks(encoder,4)
    encoder, _ = load_encoder("middle",metadata["checkpoint"],device=device,image_size=256)
    return encoder


def prepare(cfg):
    from .formal_train import validate_assets as original_assets
    from .dimension_supervision import prepare as original_prepare
    base = original_config()
    calibration = torch.load(base["top_calibration"],map_location="cpu",weights_only=True)
    sup, gate, seed = original_prepare(base,original_assets(base),calibration)
    tensors, _, _ = load_knowledge(cfg)
    for name, tensor in tensors.items():
        setattr(sup, name, tensor.clone())
    return sup, gate, seed


def validate_auxiliary(payload):
    from .dimension_checkpoint import validate_auxiliary as original_validate
    cfg = payload["public_config"]
    tensors, _, path = load_knowledge(cfg)
    aux = payload["training_auxiliary"]
    if payload["metadata"]["asset_sha256"]["supervision_provider_checkpoint"] != file_sha256(cfg["supervision_provider_checkpoint"]):
        raise ValueError("Supervision provider identity mismatch")
    if payload["metadata"]["asset_sha256"]["provider_dimension_asset"] != file_sha256(path / "manifest.json"):
        raise ValueError("Provider manifest identity mismatch")
    for name,t in tensors.items():
        if not torch.equal(aux["heads"][name],t):
            raise ValueError("Provider target checkpoint mismatch")
    # Validate the unchanged Student heads and GBW using the D2 contract.
    reference = original_config()
    original_tensors, _ = __import__("src.student.dimension_supervision",fromlist=["load_asset"]).load_asset(reference)
    checked = copy.deepcopy(payload)
    checked["public_config"] = reference
    checked["training_auxiliary"]["method"] = reference["experiment_id"].rsplit("-R",1)[0]
    for name,t in original_tensors.items():
        checked["training_auxiliary"]["heads"][name] = t[:,:256] if name=="top_basis" else t
    original_validate(checked)
