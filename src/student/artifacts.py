"""Student training metadata and weight-free result packaging."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RESULT_FILES = ("test_1652_best.json", "test_sues200_all_best.json", "test_gta_cross_area_d2s_best.json")
U1652_EVAL_BATCH_SIZE = 32


def require_u1652_eval_batch_size(batch_size):
    if batch_size != U1652_EVAL_BATCH_SIZE:
        raise ValueError("Canonical Student U1652 evaluation requires batch_size=32")


def require_valid_run(run):
    if (Path(run) / "INVALIDATED.json").exists():
        raise ValueError("INVALIDATED Student run cannot issue formal results: " + str(run))


SLIM_FILES = ("train.log", "run_config.json", "best_metrics.json", "epoch_metrics.json") + RESULT_FILES

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)

def deployment_state_dict(model):
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "student"): raw = raw.student
    state={k:v.detach().cpu() for k,v in raw.state_dict().items()}
    if any(any(token in k.lower() for token in ('stst','projector','teacher')) for k in state):
        raise ValueError('Training-only distillation state leaked into deployment')
    return state

def selection_metadata():
    """Project-approved selection contract for every Student method."""
    return dict(checkpoint_selection_dataset="University-1652",
        checkpoint_selection_split="test",
        checkpoint_selection_metric="D2S_R1 + S2D_R1",
        checkpoint_selection_rule="strict_greater_than",
        checkpoint_selection_frequency="every_epoch",
        SUES_USED_FOR_SELECTION=False, GTA_USED_FOR_SELECTION=False)

def source_identity():
    paths=set((ROOT/"src/student").glob("*.py"))
    paths.update(ROOT/p for p in (
        "src/models/repvit_backbone.py", "src/models/repvit_module.py",
        "src/dataset/transforms.py", "src/dataset/teacher/datasets.py",
        "src/dataset/teacher/transforms.py", "src/dataset/teacher/val_dataloaders.py",
        "src/utils/gather_features_and_labels_and_views.py", "src/utils/train_eval_utils.py",
        "src/evaluation/metrics.py", "src/evaluation/evaluate.py", "src/evaluation/model_loader.py",
        "scripts/train_student.sh", "scripts/train_student_certified.sh"))
    return {str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(paths)}

def resolved_config(cfg, steps_per_epoch=None):
    commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    expected=os.environ.get("STUDENT_SEALED_COMMIT")
    if expected and commit != expected:
        raise RuntimeError("HEAD changed after sealed launch")
    metadata=dict(cfg)
    metadata.update(selection_metadata())
    require_u1652_eval_batch_size(cfg.get("u1652_eval_batch_size", U1652_EVAL_BATCH_SIZE))
    metadata.update(u1652_eval_batch_size=U1652_EVAL_BATCH_SIZE, validation_buffer_source="rank0")
    return dict(metadata, experiment_name=Path(cfg["output_dir"]).name,
        method=cfg["mode"], git_commit=commit, sealed_commit=expected or commit,
        source_sha256=source_identity(),
        student_architecture="RepViT-M1.5", student_pretrained_path=str(Path(cfg["student_pretrained"]).resolve()),
        student_pretrained_sha256=file_sha256(cfg["student_pretrained"]),
        image_size=cfg["img_size"], local_pair_batch=cfg["batch_size"],
        global_pair_batch=cfg["batch_size"]*cfg["world_size"],
        optimizer="AdamW", betas=[.9,.999], warmup={"epochs":cfg["warmup_epochs"],"formula":"int(warmup_epochs * steps_per_epoch)",
            "steps_per_epoch":steps_per_epoch,"warmup_steps":None if steps_per_epoch is None else int(cfg["warmup_epochs"]*steps_per_epoch)},
        scheduler="cosine_per_step", temperature_init=cfg["temperature"],
        precision_contract={"parameters":"bfloat16","forward":"bfloat16","descriptor":"float32",
                            "L2":"float32","similarity":"float32","loss":"float32"},
        train_dataset={"name":"University-1652","split":"train","path":cfg["train_data_dir"]},
        validation_dataset={"name":"University-1652","split":"test","path":cfg["val_data_dir"],"frequency":"every_epoch"},
        checkpoint_selection={"metric":"D2S_R1 + S2D_R1","rule":"strict_greater_than","dataset":"University-1652","split":"test","frequency":"every_epoch"},
        early_stop=False)

def best_record(epoch, metrics):
    score = float(metrics["D2S"]["R@1"] + metrics["S2D"]["R@1"])
    row = dict(best_epoch=epoch,best_score=score,selection_metric="D2S_R1 + S2D_R1",
               selection_rule="strict_greater_than",selection_dataset="University-1652")
    row.update(selection_metadata())
    row["u1652_eval_batch_size"] = U1652_EVAL_BATCH_SIZE
    for direction in ("D2S","S2D"):
        for source,target in (("R@1","R1"),("R@5","R5"),("AP","AP")):
            row[direction+"_"+target]=float(metrics[direction][source])
    if not all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))):
        raise ValueError("Nonfinite validation result")
    return row

def validate_training_complete(run):
    require_valid_run(run)
    run=Path(run)
    cfg=json.loads((run/"run_config.json").read_text())
    history=json.loads((run/"epoch_metrics.json").read_text())
    if cfg["epochs"]!=30 or [row["epoch"] for row in history]!=list(range(1,31)):
        raise ValueError("Complete thirty-epoch history required")
    best=json.loads((run/"best_metrics.json").read_text())
    expected=max(history,key=lambda row:row["metrics"]["D2S"]["R@1"]+row["metrics"]["S2D"]["R@1"])
    if best != best_record(expected["epoch"],expected["metrics"]):
        raise ValueError("Best metadata violates first strict maximum selection")
    for name in ("best_model.pth","last_model.pth","train.log"):
        if not (run/name).is_file(): raise FileNotFoundError(run/name)
    return cfg,best

def package_results(run):
    run=Path(run).resolve()
    cfg,best=validate_training_complete(run)
    for name in SLIM_FILES:
        if not (run/name).is_file():raise FileNotFoundError(run/name)
    best_sha=file_sha256(run/"best_model.pth")
    for name in RESULT_FILES:
        result=json.loads((run/name).read_text())
        if result["checkpoint_sha256"]!=best_sha:
            raise ValueError("Result checkpoint identity mismatch")
    archive=run/(run.name+"_RESULTS.tar.gz")
    manifest_path=run/"RESULT_MANIFEST.txt"
    if archive.exists() or manifest_path.exists(): raise FileExistsError("Result package already exists")
    manifest=dict(EXPERIMENT_NAME=run.name,METHOD=cfg["method"],SEED=cfg["seed"],GIT_COMMIT=cfg["git_commit"],
        STUDENT_PRETRAINED_PATH=cfg["student_pretrained_path"],STUDENT_PRETRAINED_SHA256=cfg["student_pretrained_sha256"],
        BEST_MODEL=str(run/"best_model.pth"),BEST_MODEL_SHA256=best_sha,
        BEST_EPOCH=best["best_epoch"],BEST_SCORE=best["best_score"],
        LAST_MODEL=str(run/"last_model.pth"),LAST_MODEL_SHA256=file_sha256(run/"last_model.pth"),
        TRAIN_LOG=str(run/"train.log"),U1652_RESULT=str(run/RESULT_FILES[0]),
        SUES_RESULT=str(run/RESULT_FILES[1]),GTA_RESULT=str(run/RESULT_FILES[2]),
        PACKAGE=str(archive),PACKAGE_SHA256="RECORDED_IN_EXTERNAL_FINAL_MANIFEST",
        PACKAGE_SIZE="RECORDED_IN_EXTERNAL_FINAL_MANIFEST",
        package_manifest_policy="Embedded pre-package manifest; external final manifest adds actual archive SHA256 and size. Avoids circular self-hash.",
        files={name:dict(sha256=file_sha256(run/name),size=(run/name).stat().st_size) for name in SLIM_FILES})
    embedded=(json.dumps(manifest,indent=2)+"\n").encode()
    with archive.open("xb") as target:
        with tarfile.open(fileobj=target,mode="w:gz") as tar:
            for name in SLIM_FILES:tar.add(run/name,arcname=name,recursive=False)
            info=tarfile.TarInfo("RESULT_MANIFEST.txt");info.size=len(embedded)
            tar.addfile(info,io.BytesIO(embedded))
    with tarfile.open(archive) as tar:
        if set(tar.getnames()) != set(SLIM_FILES+("RESULT_MANIFEST.txt",)):
            raise RuntimeError("Slim archive allowlist violation")
    manifest.update(PACKAGE_SHA256=file_sha256(archive),PACKAGE_SIZE=archive.stat().st_size,
                    embedded_manifest_sha256=hashlib.sha256(embedded).hexdigest())
    with manifest_path.open("x") as f:json.dump(manifest,f,indent=2)
    return manifest
