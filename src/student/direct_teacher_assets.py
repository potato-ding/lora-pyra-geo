"""TRAIN-only T0-256 Top128 assets, independent of Middle asset contracts."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from .artifacts import file_sha256
from .build_bank import TrainImages, ROOT
from .subspace import identity_centroids
from .subspace_utils import tensor_sha256, orthogonality_error

SCHEMA = "T0_TOP128_CANONICAL_V1"
DIM = 4096


def teacher_identity(checkpoint):
    from src.training.teacher.formal_checkpoint import checkpoint_metadata, SCHEMA as TEACHER_SCHEMA
    from src.evaluation.protocols import require_selection_protocol
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != "best_model.pth":
        raise ValueError("Direct Teacher requires selected best_model.pth")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("artifact_schema") != TEACHER_SCHEMA:
        raise ValueError("Direct Teacher requires Teacher V3")
    require_selection_protocol(payload.get("selection_dataset_protocol"))
    meta = checkpoint_metadata(payload)
    signature = payload["precision_signature"]
    if (meta["experiment_id"] != "T0-INFONCE-R256" or meta["image_size"] != 256
            or signature.get("parameter_dtype") != "bfloat16"
            or signature.get("descriptor_dtype") != "float32"
            or signature.get("selection_batch_size") != 16):
        raise ValueError("Direct Teacher resolution/precision identity mismatch")
    config = payload["config"]
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if digest != meta["config_sha256"]:
        raise ValueError("Teacher stored config SHA mismatch")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "config_sha256": digest, "config": config, "precision_signature": signature,
            "artifact_schema": TEACHER_SCHEMA, "descriptor_dim": DIM}


def fit_top128(centroids):
    x = centroids.double()
    if x.ndim != 2 or min(x.shape) < 128 or not torch.isfinite(x).all():
        raise ValueError("Top128 requires finite matrix with rank capacity >=128")
    mean64 = x.mean(0)
    mean = mean64.float().contiguous()
    _, _, vh = torch.linalg.svd(x - mean64, full_matrices=False)
    first32 = vh[:32].T.float().contiguous()
    _, _, vh = torch.linalg.svd(x - mean.double(), full_matrices=False)
    top = vh[:128].T.contiguous()
    top[:, :32] *= torch.where((top[:, :32] * first32.double()).sum(0) < 0, -1., 1.)
    top = top.float().contiguous()
    top[:, :32] = first32
    if orthogonality_error(top) > 1e-5:
        raise ValueError("Nonorthogonal T0 Top128")
    return {"teacher_mean": mean, "top128_basis": top}


def load_top_source(manifest_path, expected_teacher_sha):
    path = Path(manifest_path)
    meta = json.loads(path.read_text())
    required = {"schema": SCHEMA, "teacher_checkpoint_sha256": expected_teacher_sha,
                "image_size":256, "descriptor_dim":DIM, "split":"train",
                "train_ids":701, "bank_rows":1402, "top_dim":128}
    if any(meta.get(k) != v for k,v in required.items()):
        raise ValueError("Direct Teacher Top128 provenance/schema mismatch")
    values = {}
    for name, shape in [("teacher_mean",(DIM,)), ("top128_basis",(DIM,128))]:
        t = torch.load(path.parent/(name+".pt"),map_location="cpu",weights_only=True)
        if t.shape != shape or t.dtype != torch.float32 or not torch.isfinite(t).all() or tensor_sha256(t) != meta["tensor_sha256"][name]:
            raise ValueError("Direct Teacher tensor identity mismatch: "+name)
        values[name] = t
    if orthogonality_error(values["top128_basis"]) > 1e-5:
        raise ValueError("Direct Teacher Top128 not orthonormal")
    values["metadata"] = meta
    return values


def generate_random32(seed):
    # Same Gaussian / QR / sign convention as formal Middle ADSD, in T0 space.
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Random32 seed out of range")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.linalg.qr(torch.randn(DIM,32,generator=generator,dtype=torch.float64),mode="reduced").Q
    pivots = q.abs().argmax(dim=0)
    q *= torch.where(q[pivots,torch.arange(32)] < 0,-1.,1.)
    return q.float().contiguous()


def build(args):
    from src.training.teacher.formal_reload import load_teacher_v3
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    device = torch.device(args.device)
    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise ValueError("T0 asset builder requires one visible CUDA GPU")
    torch.set_num_threads(4)
    identity = teacher_identity(args.teacher_checkpoint)
    pretrained_sha = file_sha256(args.student_pretrained)
    datasets = {d:TrainImages(d,args.train_root,256) for d in ("drone","satellite")}
    if datasets["drone"].ids != datasets["satellite"].ids or len(datasets["drone"].ids) != 701:
        raise ValueError("Exactly 701 matching TRAIN identities required")
    teacher,audit = load_teacher_v3(args.teacher_checkpoint,device=device,image_size=256)
    teacher.eval().requires_grad_(False)
    if teacher.descriptor_dim != DIM or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Teacher must be frozen with 4096D output")
    rows=[];path_hashes={}
    for domain,ds in datasets.items():
        path_text="\n".join(str(p.relative_to(Path(args.train_root)))+":"+str(label) for p,label in ds.rows)
        path_hashes[domain]=hashlib.sha256(path_text.encode()).hexdigest()
        chunks=[];labels=[]
        loader=DataLoader(ds,batch_size=args.teacher_batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)
        with torch.inference_mode():
            for step,(images,ids) in enumerate(loader,1):
                z=teacher(images.to(device)).cpu()
                if z.dtype != torch.float32 or z.shape[1] != DIM or not torch.isfinite(z).all():
                    raise ValueError("Teacher descriptor contract failed")
                chunks.append(z);labels.append(ids)
                if step==1 or step%200==0:
                    print("T0_ASSET_PROGRESS="+json.dumps(dict(domain=domain,batch=step,total_batches=len(loader))),flush=True)
        row,ids=identity_centroids(torch.cat(chunks),torch.cat(labels))
        if row.shape!=(701,DIM):raise ValueError("Teacher centroid geometry")
        rows.append(row)
    centroids=torch.cat(rows)
    del teacher,chunks,loader
    gc.collect();torch.cuda.empty_cache()
    values=fit_top128(centroids)
    # The same initial Student protocol as existing Middle calibration assets.
    from .model import StudentModel
    from src.evaluation.model_loader import EvaluationEncoder
    from src.evaluation.precision_contract import apply_runtime_precision
    torch.manual_seed(0)
    student=StudentModel(ckpt_path=args.student_pretrained).to(device)
    apply_runtime_precision(student,"student")
    encoder=EvaluationEncoder(student,512).eval().requires_grad_(False)
    chunks=[];paths=[]
    with torch.inference_mode():
        for domain,ds in datasets.items():
            loader=DataLoader(Subset(ds,range(384)),batch_size=32,shuffle=False,num_workers=args.num_workers)
            chunks.extend(encoder(images.to(device)).cpu() for images,_ in loader)
            paths.extend(str(p) for p,_ in ds.rows[:384])
    calibration=torch.cat(chunks)
    if calibration.shape!=(768,512) or calibration.dtype!=torch.float32 or not torch.isfinite(calibration).all():
        raise ValueError("Student calibration contract failed")
    if file_sha256(args.teacher_checkpoint)!=identity["checkpoint_sha256"] or file_sha256(args.student_pretrained)!=pretrained_sha:
        raise RuntimeError("Source asset changed during build")
    top=out/"top128";top.mkdir(parents=True)
    for name,t in values.items():
        with (top/(name+".pt")).open("xb") as f:torch.save(t,f)
    with (out/"train_identity_centroids.pt").open("xb") as f:torch.save(centroids,f)
    meta=dict(schema=SCHEMA,teacher_checkpoint_sha256=identity["checkpoint_sha256"],
              teacher_config_sha256=identity["config_sha256"],teacher_checkpoint=identity["checkpoint"],
              image_size=256,descriptor_dim=DIM,split="train",train_ids=701,bank_rows=1402,top_dim=128,
              train_path_identity_sha256=path_hashes,teacher_precision_signature=identity["precision_signature"],
              teacher_extraction_batch_size=args.teacher_batch_size,
              fitting_protocol="identity_centroids_drone_then_satellite_FP64_SVD_legacy32_sign_alignment",
              tensor_sha256={k:tensor_sha256(v) for k,v in values.items()},
              centroid_sha256=tensor_sha256(centroids),builder_sha256=file_sha256(__file__))
    (top/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    calibration_path=out/"top_calibration.pt"
    with calibration_path.open("xb") as f:torch.save(calibration,f)
    (out/"top_calibration.pt.json").write_text(json.dumps(dict(split="train",seed=0,img_size=256,
        paths=paths,teacher_checkpoint_sha256=identity["checkpoint_sha256"],teacher_config_sha256=identity["config_sha256"],
        student_pretrained_sha256=pretrained_sha,calibration_sha256=file_sha256(calibration_path)),indent=2)+"\n")
    (out/"teacher_identity.json").write_text(json.dumps(identity,indent=2)+"\n")
    load_top_source(top/"manifest.json",identity["checkpoint_sha256"])
    (out/"BUILD_COMPLETE.json").write_text(json.dumps(dict(status="PASS",top_shape=[DIM,128],calibration_shape=[768,512],
        random_basis_saved=False,teacher_checkpoint_sha256=identity["checkpoint_sha256"]),indent=2)+"\n")
    print("T0_DIRECT_ASSETS=PASS",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-checkpoint",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--student-pretrained",required=True)
    p.add_argument("--train-root",default=str(ROOT/"data/U1652/train"))
    p.add_argument("--teacher-batch-size",type=int,default=4)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    if args.teacher_batch_size<=0:raise ValueError("Positive Teacher batch size required")
    build(args)


if __name__=="__main__":main()
