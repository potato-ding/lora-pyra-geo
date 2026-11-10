"""T0 provider and 4096D assets; reuse the formal S3 objectives and SAM engine."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from .artifacts import file_sha256
from .subspace_utils import tensor_sha256, orthogonality_error
from .direct_teacher_assets import load_top_source, teacher_identity
from .formal_supervision import SubspaceSupervision, SubspaceProjector, BandProjector
from .random_structure import ALGORITHM

CONFIG_KEYS = {"teacher_provider", "teacher_checkpoint", "teacher_asset_identity",
               "supervision_asset", "top_calibration", "random_basis_asset",
               "random_reference", "teacher_forward_batch_size", "distillation",
               "head_initialization_rng_asset", "head_initialization_rng_sha256"}


def validate_config(cfg):
    if "hrd" in cfg:
        from .direct_teacher_hrd import validate_hrd_config
        validate_hrd_config(cfg)
    from .formal_config import validate_adsd
    if cfg["teacher_provider"] != "T0" or cfg["img_size"] != 256 or cfg["experiment_id"] != "S3-SAM-ADSD-R256":
        raise ValueError("Direct S3 requires T0 R256")
    if type(cfg["teacher_forward_batch_size"]) is not int or cfg["teacher_forward_batch_size"] != 4:
        raise ValueError("Direct frozen Teacher forward batch must be 4")
    component = copy.deepcopy(cfg["distillation"])
    if component["random_basis"] != "teacher_gaussian_qr_same_seed":
        raise ValueError("Direct Teacher Random32 protocol changed")
    component["random_basis"] = "gaussian_qr_per_run"
    validate_adsd(component)
    rng_sha = cfg["head_initialization_rng_sha256"]
    if not isinstance(rng_sha,str) or len(rng_sha)!=64 or any(c not in "0123456789abcdef" for c in rng_sha):
        raise ValueError("Head initialization RNG SHA invalid")
    ref = cfg["random_reference"]
    if set(ref) != {"checkpoint", "checkpoint_sha256", "random_basis_seed", "source_basis_sha256", "t0_basis_sha256", "generation_algorithm"}:
        raise ValueError("Direct Random32 reference inventory")
    if not isinstance(ref["checkpoint"], str) or not ref["checkpoint"]:
        raise ValueError("Direct Random32 source checkpoint missing")
    for key in ("checkpoint_sha256", "source_basis_sha256", "t0_basis_sha256"):
        v=ref[key]
        if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("Direct Random32 SHA invalid")
    if type(ref["random_basis_seed"]) is not int or not 0 <= ref["random_basis_seed"] < 2**63 or ref["generation_algorithm"] != ALGORITHM:
        raise ValueError("Direct Random32 seed/algorithm mismatch")


def validate_random_tensor(basis, cfg):
    if basis.shape != (4096,32) or basis.dtype != torch.float32 or not torch.isfinite(basis).all():
        raise ValueError("Direct Random32 must be finite FP32 [4096,32]")
    if tensor_sha256(basis) != cfg["random_reference"]["t0_basis_sha256"] or orthogonality_error(basis)>1e-5:
        raise ValueError("Direct Random32 tensor identity mismatch")


def validate_assets(cfg, root):
    from .formal_checkpoint import validate_checkpoint
    validate_config(cfg)
    required = ("teacher_checkpoint", "teacher_asset_identity", "supervision_asset", "top_calibration", "random_basis_asset", "head_initialization_rng_asset")
    for key in required:
        if not Path(cfg[key]).is_file(): raise FileNotFoundError(cfg[key])
    if file_sha256(cfg["head_initialization_rng_asset"]) != cfg["head_initialization_rng_sha256"]:
        raise ValueError("Head initialization RNG file SHA mismatch")
    identity = teacher_identity(cfg["teacher_checkpoint"])
    recorded = json.loads(Path(cfg["teacher_asset_identity"]).read_text())
    if identity != recorded:
        raise ValueError("T0 asset Teacher identity no longer matches selected checkpoint")
    top = load_top_source(cfg["supervision_asset"], identity["checkpoint_sha256"])
    if top["metadata"]["teacher_config_sha256"] != identity["config_sha256"]:
        raise ValueError("T0 Top128 config identity mismatch")
    cal_path = Path(cfg["top_calibration"])
    cal = torch.load(cal_path, map_location="cpu", weights_only=True)
    meta_path = Path(str(cal_path)+".json")
    meta = json.loads(meta_path.read_text())
    pretrained_sha = file_sha256(root/cfg["pretrained_checkpoint"])
    expected = {"split":"train", "seed":0, "img_size":256,
                "teacher_checkpoint_sha256":identity["checkpoint_sha256"],
                "teacher_config_sha256":identity["config_sha256"],
                "student_pretrained_sha256":pretrained_sha,
                "calibration_sha256":file_sha256(cal_path)}
    if (cal.shape != (768,512) or cal.dtype != torch.float32 or not torch.isfinite(cal).all()
            or any(meta.get(k)!=v for k,v in expected.items())):
        raise ValueError("Direct Student calibration provenance/shape mismatch")
    ref = cfg["random_reference"]
    if file_sha256(ref["checkpoint"]) != ref["checkpoint_sha256"]:
        raise ValueError("Original S3 reference checkpoint SHA mismatch")
    source=torch.load(ref["checkpoint"],map_location="cpu",weights_only=True)
    validate_checkpoint(source)
    if source["public_config"]["experiment_id"] != "S3-SAM-ADSD-R256" or source["public_config"].get("teacher_provider") is not None:
        raise ValueError("Random reference must be original M3-supervised S3 R256")
    if source["public_config"].get("middle_provider", "M3") != "M3":
        raise ValueError("Random reference requires M3")
    auxiliary=source["training_auxiliary"]
    if auxiliary["random_basis_seed"]!=ref["random_basis_seed"] or tensor_sha256(auxiliary["heads"]["random32_basis"])!=ref["source_basis_sha256"]:
        raise ValueError("Original S3 random source seed/tensor mismatch")
    basis=torch.load(cfg["random_basis_asset"],map_location="cpu",weights_only=True)
    validate_random_tensor(basis,cfg)
    # Regenerate only during preflight to verify protocol, never in the training constructor.
    from .direct_teacher_assets import generate_random32
    if not torch.equal(basis,generate_random32(ref["random_basis_seed"])):
        raise ValueError("T0 random basis does not follow the approved shared-seed protocol")
    return dict(student_pretrained=pretrained_sha, teacher_checkpoint=identity["checkpoint_sha256"],
                teacher_asset_identity=file_sha256(cfg["teacher_asset_identity"]),
                supervision_asset=file_sha256(cfg["supervision_asset"]),top_calibration=file_sha256(cal_path),
                top_calibration_metadata=file_sha256(meta_path),random_basis_asset=file_sha256(cfg["random_basis_asset"]),
                random_reference_checkpoint=ref["checkpoint_sha256"],
                head_initialization_rng=cfg["head_initialization_rng_sha256"])


class TeacherSubspaceSupervision(SubspaceSupervision):
    """Same Student heads, target projection and loss; T0 knowledge buffers are 4096D."""
    def __init__(self, cfg, teacher_sha, calibration):
        nn.Module.__init__(self)
        asset=load_top_source(cfg["supervision_asset"],teacher_sha)
        self.asset_path=cfg["supervision_asset"];self.asset_sha256=file_sha256(self.asset_path)
        self.metadata=asset["metadata"];self.student_dim=512
        self.top_dim=128;self.random_layout="single32";self.random_total_dim=32
        self.register_buffer("teacher_mean",asset["teacher_mean"].clone())
        self.register_buffer("top128_basis",asset["top128_basis"].clone())
        basis=torch.load(cfg["random_basis_asset"],map_location="cpu",weights_only=True)
        validate_random_tensor(basis,cfg)
        self.register_buffer("random32_basis",basis.clone())
        # Preserve exactly the canonical Student head initialization sequence.
        if file_sha256(cfg["head_initialization_rng_asset"]) != cfg["head_initialization_rng_sha256"]:
            raise ValueError("Head initialization RNG SHA mismatch")
        initial_rng=torch.load(cfg["head_initialization_rng_asset"],map_location="cpu",weights_only=True)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initial_rng)
            reference_top=SubspaceProjector();reference_random=SubspaceProjector()
            reference_random.load_state_dict(reference_top.state_dict(),strict=True)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20260914)
                top=BandProjector(128);BandProjector(32);random=BandProjector(32)
                with torch.no_grad():
                    top.linear.weight[:32].copy_(reference_top.linear.weight);top.linear.bias[:32].copy_(reference_top.linear.bias)
                    random.linear.weight.copy_(reference_random.linear.weight);random.linear.bias.copy_(reference_random.linear.bias)
            self.projector_top=top;self.projector_random=random
            from .part2 import install_residual_top
            install_residual_top(self,"rmlp",calibration)
        self.random_basis_generation_count=0
        self.random_structure_metadata=dict(cfg["random_reference"],shape=[4096,32],same_subspace_as_middle=False)


class FrozenTeacherChunks(nn.Module):
    """Bound frozen 7B inference memory without splitting the Student logical batch."""
    input_dtype=torch.float32
    def __init__(self, teacher, batch_size):
        super().__init__();self.teacher=teacher;self.batch_size=batch_size
        self.eval().requires_grad_(False)
    @torch.no_grad()
    def forward(self,images):
        if self.training or any(p.requires_grad for p in self.teacher.parameters()):
            raise RuntimeError("T0 provider must stay frozen")
        result=torch.cat([self.teacher(chunk.float()) for chunk in images.split(self.batch_size)])
        if result.shape!=(images.shape[0],4096) or result.dtype!=torch.float32 or not torch.isfinite(result).all():
            raise RuntimeError("Direct frozen T0 descriptor contract")
        return result


def load_provider(cfg,device):
    from src.training.teacher.formal_reload import load_teacher_v3
    teacher,_=load_teacher_v3(cfg["teacher_checkpoint"],device=device,image_size=256)
    return FrozenTeacherChunks(teacher,cfg["teacher_forward_batch_size"])


def validate_checkpoint_auxiliary(payload):
    cfg=payload["public_config"];validate_config(cfg)
    aux=payload["training_auxiliary"];heads=aux["heads"]
    if payload["metadata"]["asset_sha256"]["head_initialization_rng"] != cfg["head_initialization_rng_sha256"]:
        raise ValueError("Direct head initialization provenance mismatch")
    gate=aux["allocation_gate"]["d"]
    if gate.shape!=() or not torch.isfinite(gate):
        raise ValueError("Direct GBW gate must be a finite FP32 scalar")
    validate_random_tensor(heads["random32_basis"],cfg)
    if aux["random_basis_seed"]!=cfg["random_reference"]["random_basis_seed"]:
        raise ValueError("Direct checkpoint random seed mismatch")
    if payload["metadata"]["asset_sha256"]["random_reference_checkpoint"]!=cfg["random_reference"]["checkpoint_sha256"]:
        raise ValueError("Direct checkpoint random reference SHA mismatch")
    for branch,dim,kind in (("top",128,"residual_mlp"),("random",32,"linear")):
        prefix="projector_"+branch+"."
        expected={"linear.weight":(dim,512),"linear.bias":(dim,)}
        if kind=="residual_mlp":expected.update({"residual.fc1.weight":(920,512),"residual.fc1.bias":(920,),"residual.fc2.weight":(dim,920),"residual.fc2.bias":(dim,),"gate.alpha":()})
        if {k[len(prefix):] for k in heads if k.startswith(prefix)}!=set(expected):
            raise ValueError("Direct predictor tensor inventory mismatch")
        for name,shape in expected.items():
            t=heads[prefix+name]
            if t.shape!=shape or not torch.isfinite(t).all() or (name.startswith(("residual.","gate.")) and t.dtype!=torch.float32):
                raise ValueError("Direct predictor tensor shape/precision mismatch")
    for key in ("teacher_mean","top128_basis"):
        if not torch.isfinite(heads[key]).all():raise ValueError("Direct knowledge nonfinite")
    if orthogonality_error(heads["top128_basis"])>1e-5:raise ValueError("Direct Top128 not orthonormal")
