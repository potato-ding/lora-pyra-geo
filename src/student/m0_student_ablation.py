"""B1 M0 provider, own TRAIN PCA, and exact final-S3 Random32 reuse."""
import json
from pathlib import Path
import torch
from .ablation_config import ROOT, BASE_CONFIG, validate_config
from .artifacts import file_sha256
from .formal_assets import validate_middle_provider
from .formal_supervision import load_top_source
from .shared_subspaces import load_reference, initialization_hashes
from .subspace_utils import tensor_sha256
from .random_structure import validate_basis


def validate_assets(cfg):
    validate_config(cfg)
    if cfg["ablation"]["variant"] != "b1_m0_provider":
        raise ValueError("M0 is only allowed for B1")
    _,sha=validate_middle_provider(cfg["middle_checkpoint"],cfg["middle_config"],256,
                                   expected_method="M0",allow_m0_ablation=True)
    pretrained=ROOT/cfg["pretrained_checkpoint"]
    values={"student_pretrained":file_sha256(pretrained),"middle_checkpoint":sha,
            "middle_config":file_sha256(cfg["middle_config"])}
    path=Path(cfg["supervision_asset"])
    if not path.is_file():
        raise FileNotFoundError("B1_ASSET_BUILD_PENDING: "+str(path))
    top=load_top_source(path,sha)
    if top["metadata"]["image_size"]!=256 or top["metadata"].get("teacher_config_sha256")!=values["middle_config"]:
        raise ValueError("B1 M0 Top provenance/resolution mismatch")
    calpath=Path(cfg["top_calibration"])
    calibration=torch.load(calpath,map_location="cpu",weights_only=True)
    if calibration.shape!=(768,512) or calibration.dtype!=torch.float32 or not torch.isfinite(calibration).all():
        raise ValueError("B1 calibration must be finite FP32 [768,512]")
    metapath=Path(str(calpath)+".json")
    meta=json.loads(metapath.read_text())
    required={"split":"train","seed":cfg["seed"],"img_size":256,
              "middle_checkpoint_sha256":sha,"middle_config_sha256":values["middle_config"],
              "student_pretrained_sha256":values["student_pretrained"],"calibration_sha256":file_sha256(calpath)}
    if any(meta.get(k)!=v for k,v in required.items()):
        raise ValueError("B1 calibration identity mismatch")
    values.update(supervision_asset=file_sha256(path),top_calibration=file_sha256(calpath),
                  top_calibration_metadata=file_sha256(metapath),
                  random_reference_checkpoint=cfg["random_reference"]["checkpoint_sha256"])
    reference_tensors(cfg)
    return values


def reference_tensors(cfg):
    from .formal_train import validate_assets as validate_formal_assets
    base=json.loads(BASE_CONFIG.read_text())
    if cfg["random_reference"]!=base["subspace_reference"]:
        raise ValueError("B1 must bind the immutable final S3 reference")
    assets=validate_formal_assets(base)
    return load_reference(cfg["random_reference"],base,assets)


def supervision_tensors(cfg,assets):
    reference=reference_tensors(cfg)
    top=load_top_source(cfg["supervision_asset"],assets["middle_checkpoint"])
    return {"teacher_mean":top["teacher_mean"],"top128_basis":top["top128_basis"],
            "random32_basis":reference["random32_basis"]}


def validate_knowledge(heads,cfg,assets):
    path=Path(cfg["supervision_asset"])
    if file_sha256(path)!=assets["supervision_asset"]:
        raise ValueError("B1 Top manifest SHA mismatch")
    top=load_top_source(path,assets["middle_checkpoint"])
    if top["metadata"]["image_size"]!=256 or top["metadata"].get("teacher_config_sha256")!=assets["middle_config"]:
        raise ValueError("B1 Top provider metadata mismatch")
    for key in ("teacher_mean","top128_basis"):
        if not torch.is_tensor(heads.get(key)) or not torch.equal(heads[key],top[key]):
            raise ValueError("B1 knowledge must match M0 Top: "+key)
    basis=heads.get("random32_basis")
    validate_basis(basis)
    if tensor_sha256(basis)!=cfg["random_reference"]["tensor_sha256"]["random32_basis"]:
        raise ValueError("B1 Random32 differs from final S3")
    if assets["random_reference_checkpoint"]!=cfg["random_reference"]["checkpoint_sha256"]:
        raise ValueError("B1 Random reference checkpoint identity mismatch")


def write_identity(output,cfg,supervision,student):
    identity={"source":cfg["random_reference"],"provider":"M0",
              "top_source":cfg["supervision_asset"],"random_generation_count":supervision.random_basis_generation_count,
              "tensor_sha256":{k:tensor_sha256(getattr(supervision,k)) for k in ("teacher_mean","top128_basis","random32_basis")}}
    if identity["random_generation_count"]!=0:
        raise ValueError("B1 unexpectedly generated a new random basis")
    (output/"subspace_identity.json").write_text(json.dumps(identity,indent=2)+"\n")
    (output/"initialization_identity.json").write_text(json.dumps({"student":initialization_hashes(student),
        "base_heads":initialization_hashes(supervision,base_heads_only=True)},indent=2)+"\n")
    print("B1_M0_SUBSPACE_IDENTITY="+json.dumps(identity),flush=True)
