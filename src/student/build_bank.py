"""Build the fixed Final Middle TRAIN-only Dual-STST asset; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.dataset.transforms import get_test_transforms
from src.evaluation.model_loader import load_encoder
from .subspace import construct_train_subspace
from .dual_stst import file_sha256, load_stst_asset

ROOT = Path(__file__).resolve().parents[2]
TEACHER = ROOT / "src/checkpoint/middle_teacher/CERTIFIED_R224/SAM-MABV2-RHO010-S0"
ASSETS = ROOT / "src/checkpoint/student/CERTIFIED_R224/STST_ASSETS"
DEFAULT_BANK = ASSETS / "SAM-MABV2-RHO010-S0_train_shared_k32.pt"

class TrainImages(Dataset):
    def __init__(self, domain):
        self.root = ROOT / "data/U1652/train" / domain
        self.transform = get_test_transforms([224, 224])
        self.ids = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.rows = []
        for pid in self.ids:
            paths = sorted(p for p in (self.root/pid).iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if not paths or (domain == "satellite" and len(paths) != 1):
                raise ValueError("Invalid TRAIN identity " + pid)
            self.rows.extend((p, int(pid)) for p in paths)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        path, label = self.rows[index]
        with Image.open(path) as im:
            array = np.array(im.convert("RGB"))
        return self.transform(image=array)["image"], label

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args(argv)
    if DEFAULT_BANK.exists(): raise FileExistsError(DEFAULT_BANK)
    torch.set_num_threads(8)
    ASSETS.mkdir(parents=True, exist_ok=True)
    checkpoint = TEACHER/"best_model.pth"
    initial_sha = file_sha256(checkpoint)
    datasets = {d: TrainImages(d) for d in ("drone", "satellite")}
    if datasets["drone"].ids != datasets["satellite"].ids or len(datasets["drone"].ids) != 701:
        raise ValueError("Exactly 701 matching TRAIN identities required")
    model, audit = load_encoder("middle", checkpoint, TEACHER/"run_config.json", args.device)
    if audit["missing"] or audit["unexpected"] or audit["sha256"] != initial_sha:
        raise RuntimeError("Teacher strict load/identity failed")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Teacher must be eval/frozen")
    features, labels, smoke = {}, {}, None
    for domain, dataset in datasets.items():
        chunks, ids = [], []
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=False, pin_memory=True)
        with torch.inference_mode():
            for step, (images, target) in enumerate(loader):
                desc = model(images.to(args.device, non_blocking=True))
                if desc.dtype != torch.float32 or not torch.isfinite(desc).all():
                    raise RuntimeError("Invalid descriptor")
                if smoke is None:
                    norms = desc.norm(dim=1)
                    smoke = dict(strict_load=audit, teacher_frozen=True, teacher_eval=True,
                        parameter_dtypes=sorted({str(p.dtype) for p in model.model.parameters()}),
                        descriptor_dtype=str(desc.dtype), shape=list(desc.shape),
                        finite=True, norm_min=float(norms.min()), norm_max=float(norms.max()))
                    (ASSETS/"teacher_strict_load_smoke.json").write_text(json.dumps(smoke, indent=2))
                chunks.append(desc.cpu()); ids.append(target)
                if step % 100 == 0:
                    print(json.dumps(dict(domain=domain, done=min((step+1)*args.batch_size,len(dataset)),
                                          total=len(dataset))), flush=True)
        features[domain] = torch.cat(chunks); labels[domain] = torch.cat(ids)
    bank = construct_train_subspace(features["drone"], labels["drone"],
            features["satellite"], labels["satellite"], middle_sha256=initial_sha, split="train")
    bank["metadata"].update(teacher_checkpoint=str(checkpoint),
        teacher_config=str(TEACHER/"run_config.json"),
        teacher_config_sha256=file_sha256(TEACHER/"run_config.json"),
        teacher_architecture="dinov3_vitb16", teacher_descriptor_dim=768,
        source_code_sha256=file_sha256(Path(__file__)),
        canonical_subspace_sha256=file_sha256(Path(__file__).with_name("subspace.py")),
        train_root=str(ROOT/"data/U1652/train"), image_size=224,
        preprocessing="deterministic canonical test transform applied to TRAIN images only",
        input_dtype="float32", parameter_dtype="bfloat16", descriptor_dtype="float32",
        image_counts={d:len(ds) for d,ds in datasets.items()},
        train_path_identity_sha256={d:hashlib.sha256("\n".join(str(path.relative_to(ROOT/"data/U1652/train"))+":"+str(label)
            for path,label in ds.rows).encode()).hexdigest() for d,ds in datasets.items()})
    checks = dict(teacher_sha_match=file_sha256(checkpoint)==initial_sha,
        mean_shape=list(bank["teacher_mean"].shape),
        top_shape=list(bank["top32_basis"].shape), random_shape=list(bank["random32_basis"].shape),
        finite=all(torch.isfinite(bank[k]).all().item() for k in ("teacher_mean","top32_basis","random32_basis")),
        top_orthogonality_error=float((bank["top32_basis"].T@bank["top32_basis"]-torch.eye(32)).abs().max()),
        random_orthogonality_error=float((bank["random32_basis"].T@bank["random32_basis"]-torch.eye(32)).abs().max()))
    checks["pass"] = (checks["teacher_sha_match"] and checks["finite"] and
        checks["mean_shape"]==[768] and checks["top_shape"]==checks["random_shape"]==[768,32]
        and checks["top_orthogonality_error"]<=1e-4 and checks["random_orthogonality_error"]<=1e-4)
    if not checks["pass"]: raise RuntimeError(checks)
    with DEFAULT_BANK.open("xb") as handle: torch.save(bank, handle)
    load_stst_asset(DEFAULT_BANK, expected_teacher_sha256=initial_sha)
    checks.update(bank_path=str(DEFAULT_BANK),bank_sha256=file_sha256(DEFAULT_BANK),metadata=bank["metadata"])
    (ASSETS/"bank_validation.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)

if __name__ == "__main__": main()
