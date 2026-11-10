"""Formal single-GPU S0, TSD, ADSD and SAM-ADSD training."""
from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from src.evaluation.model_loader import EvaluationEncoder, load_encoder
from src.evaluation.precision_contract import selection_signature
from src.evaluation.protocols import RETRIEVAL_PROTOCOL
from src.evaluation.student_canonical import evaluate_student_u1652_canonical
from src.middle_teacher.checkpoint import sha256
from .adsd import ADSDGate
from .data import create_student_train_dataset_and_loader
from .formal_checkpoint import SCHEMA, fingerprint, validate_checkpoint
from .formal_config import ROOT, load_formal_config, should_select_epoch
from .formal_engine import StudentTrainingModel, deepspeed_config
from .formal_methods import (
    prepare_tsd, prepare_adsd, task_objective, tsd_objective,
    adsd_objective, sam_adsd_backward,
)
from .formal_supervision import load_top_source
from .model import StudentModel
from .dimension_config import is_dimension_config
from .objective import PairInfoNCE
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler, assert_student_scheduler_groups


def method_name(config):
    from .s0_backbones import S0_BACKBONES

    method = config["experiment_id"].rsplit("-R", 1)[0]
    return "S0-INFONCE" if method in S0_BACKBONES.values() else method


def sam_enabled(config):
    """Use the validated config to select the SAM or ordinary backward path."""
    return config.get("sam", {}).get("enabled", False) is True


def validate_assets(config):
    if "provider_control" in config:
        from .provider_dimensions import validate_assets as validate_provider
        return validate_provider(config)
    if is_dimension_config(config):
        from .dimension_supervision import validate_assets as validate_dimension_assets
        return validate_dimension_assets(config)
    if config.get("distillation", {}).get("type") == "VkD":
        from .vkd_config import validate_config
        from .formal_assets import validate_middle_provider
        validate_config(config)
        _, provider_sha = validate_middle_provider(config["middle_checkpoint"],config["middle_config"],256,expected_method="M3")
        pretrained = (ROOT / config["pretrained_checkpoint"]).resolve()
        if not pretrained.is_file(): raise FileNotFoundError(pretrained)
        return {"student_pretrained":sha256(pretrained),"middle_checkpoint":provider_sha,
                "middle_config":sha256(config["middle_config"])}
    if "ablation" in config:
        if config["ablation"]["variant"] == "b1_m0_provider":
            from .m0_student_ablation import validate_assets as validate_m0_assets
            return validate_m0_assets(config)
        from .ablation_config import reference_config
        return validate_assets(reference_config(config))
    if config.get("teacher_provider") == "T0":
        from .direct_teacher_distillation import validate_assets as validate_direct_assets
        return validate_direct_assets(config, ROOT)
    size = config["img_size"]
    pretrained = (ROOT / config["pretrained_checkpoint"]).resolve()
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    hashes = {"student_pretrained": sha256(pretrained)}
    if method_name(config) == "S0-INFONCE":
        return hashes
    middle_path = Path(config["middle_checkpoint"]).resolve()
    middle_config_path = Path(config["middle_config"]).resolve()
    if method_name(config) == "S-HRD-SAM":
        from .formal_assets import validate_middle_provider
        _, middle_sha = validate_middle_provider(
            middle_path, middle_config_path, size, expected_method="M3")
        hashes.update(middle_checkpoint=middle_sha, middle_config=sha256(middle_config_path))
        return hashes
    asset_path = Path(config["supervision_asset"]).resolve()
    calibration_path = Path(config["top_calibration"]).resolve()
    for path in (middle_path, middle_config_path, asset_path, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if middle_path.name != "best_model.pth":
        raise ValueError("Student requires the selected Middle best_model.pth")
    from .formal_assets import validate_middle_provider
    expected_provider = config.get("middle_provider") or {
        "S1-TSD": "M2", "S2-ADSD": "M2",
        "S3-SAM-ADSD": "M3", "S4-ADSD-SAM-MIDDLE": "M3",
    }[method_name(config)]
    _, middle_sha = validate_middle_provider(
        middle_path, middle_config_path, size, expected_method=expected_provider,
    )
    top = load_top_source(asset_path, middle_sha)
    if top["metadata"]["image_size"] != size:
        raise ValueError("Top128 source resolution mismatch")
    if top["metadata"].get("teacher_config_sha256") != sha256(middle_config_path):
        raise ValueError("Top128 source Middle config mismatch")
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=True)
    if calibration.shape != (768, 512) or calibration.dtype != torch.float32 or not torch.isfinite(calibration).all():
        raise ValueError("Top128 calibration must be finite FP32 [768,512]")
    calibration_meta_path = Path(str(calibration_path) + ".json")
    if not calibration_meta_path.is_file():
        raise FileNotFoundError(calibration_meta_path)
    calibration_meta = json.loads(calibration_meta_path.read_text())
    expected_calibration = {
        "split": "train", "img_size": size,
        "middle_checkpoint_sha256": middle_sha,
        "middle_config_sha256": sha256(middle_config_path),
        "student_pretrained_sha256": hashes["student_pretrained"],
        "calibration_sha256": sha256(calibration_path),
    }
    if any(calibration_meta.get(key) != value for key, value in expected_calibration.items()):
        raise ValueError("Top128 calibration provenance mismatch")
    hashes.update(
        middle_checkpoint=middle_sha, middle_config=sha256(middle_config_path),
        supervision_asset=sha256(asset_path), top_calibration=sha256(calibration_path),
        top_calibration_metadata=sha256(calibration_meta_path),
    )
    if "subspace_reference" in config:
        from .shared_subspaces import load_reference
        tensors = load_reference(config["subspace_reference"], config, hashes)
        if not torch.equal(tensors["teacher_mean"],top["teacher_mean"]) or not torch.equal(tensors["top128_basis"],top["top128_basis"]):
            raise ValueError("Shared source Top128 differs from canonical asset")
        hashes["subspace_reference_checkpoint"] = config["subspace_reference"]["checkpoint_sha256"]
    return hashes


@torch.no_grad()
def select_student(engine, config, output, epoch, best_score, asset_hashes,
                   supervision, gate, random_seed, source_identity):
    student = engine.module.student
    student.eval()
    try:
        signature = selection_signature(
            student, "student", config["img_size"],
            selection_batch_size=config["selection_eval_batch_size"],
        )
        metrics = evaluate_student_u1652_canonical(
            EvaluationEncoder(student, getattr(student, "embedding_dim", 512)).eval(),
            image_size=config["img_size"], device=next(student.parameters()).device,
            data_dir=str(ROOT / config["data_dir"]),
            num_workers=config["num_workers"],
            batch_size=config["selection_eval_batch_size"],
        )
        score = float(metrics["D2S"]["R@1"] + metrics["S2D"]["R@1"])
        if not torch.isfinite(torch.tensor(score)):
            raise FloatingPointError("Nonfinite Student selection score")
        improved = score > best_score
        if improved:
            auxiliary = None
            if supervision is not None:
                auxiliary = {
                    "method": method_name(config),
                    "heads": {k: v.detach().cpu().clone()
                              for k, v in supervision.state_dict().items()},
                    "random_basis_seed": random_seed,
                    "allocation_gate": None if gate is None else {
                        k: v.detach().cpu().clone() for k, v in gate.state_dict().items()
                    },
                }
            state = {
                k: (v.detach().cpu().float().clone() if v.is_floating_point()
                    else v.detach().cpu().clone())
                for k, v in student.state_dict().items()
            }
            metadata = {
                "experiment_id": config["experiment_id"],
                "image_size": config["img_size"],
                "config_sha256": fingerprint(config),
                "best_epoch": epoch, "best_score": score,
                "selection_world_size": 1,
                "selection_batch_size": config["selection_eval_batch_size"],
                "selection_metric": "U1652_D2S_R1+U1652_S2D_R1",
                "asset_sha256": asset_hashes,
                "source_identity": source_identity,
            }
            payload = {
                "artifact_schema": SCHEMA, "model": state,
                "selection_dataset_protocol": RETRIEVAL_PROTOCOL,
                "public_config": dict(config), "metadata": metadata,
                "precision_signature": signature,
                "selection_metrics": metrics,
                "training_auxiliary": auxiliary,
            }
            validate_checkpoint(payload)
            temporary = output / "_best_model.tmp"
            try:
                torch.save(payload, temporary)
                os.replace(temporary, output / "best_model.pth")
            finally:
                temporary.unlink(missing_ok=True)
        print("STUDENT_SELECTION=" + json.dumps({
            "epoch": epoch, "metrics": metrics, "R1_sum": score,
            "u1652_protocol": RETRIEVAL_PROTOCOL,
            "best_update": improved, "best_score": max(score, best_score),
        }), flush=True)
        return max(score, best_score)
    finally:
        student.train()


def collect_student_source_identity(config):
    """Record method-specific source files without changing frozen stage manifests."""
    from src.source_contract import source_identity as calculate_source_identity
    kind = method_name(config)
    if kind == "VKD-SAM":
        source = calculate_source_identity("m2s",gbw=False,sam=True)
        for name in ("vkd.py","vkd_config.py","vkd_checkpoint.py","distill_sam.py","formal_train.py","formal_config.py","formal_checkpoint.py"):
            source["src/student/"+name] = sha256(ROOT / "src/student" / name)
        return source
    if "ablation" in config:
        from .ablation_config import reference_config
        source = collect_student_source_identity(reference_config(config))
        for name in ("src/student/ablation_config.py", "src/student/component_ablation.py", "src/student/ablation_checkpoint.py"):
            source[name] = sha256(ROOT / name)
        if "provider_control" in config:
            source["src/student/provider_dimensions.py"] = sha256(ROOT/"src/student/provider_dimensions.py")
        if is_dimension_config(config):
            for name in ("dimension_config.py","dimension_supervision.py","dimension_checkpoint.py","dimension_random.py"):
                source["src/student/"+name] = sha256(ROOT/"src/student"/name)
        if config["ablation"]["variant"].startswith(("c1_","c2_")):
            source["src/student/full_feature_residual.py"] = sha256(ROOT / "src/student/full_feature_residual.py")
        if config["ablation"]["variant"] == "b1_m0_provider":
            for name in ("src/student/m0_student_ablation.py", "src/student/formal_assets.py"):
                source[name] = sha256(ROOT / name)
        return source
    source = calculate_source_identity(
        "m2s", gbw=kind in ("S2-ADSD", "S3-SAM-ADSD", "S4-ADSD-SAM-MIDDLE"),
        sam=kind in ("S3-SAM-ADSD", "S-HRD-SAM"),
    )
    if kind == "S-HRD-SAM" or "subspace_reference" in config or "random_basis_replicate" in config:
        source["src/student/shared_subspaces.py"] = sha256(ROOT / "src/student/shared_subspaces.py")
    if config.get("teacher_provider") == "T0":
        source.update(calculate_source_identity("teacher"))
        for name in ("src/student/direct_teacher_assets.py", "src/student/direct_teacher_distillation.py", "src/student/shared_subspaces.py"):
            source[name] = sha256(ROOT / name)
    if kind in ("S3-SAM-ADSD", "S-HRD-SAM"):
        sam_source = "src/student/distill_sam.py"
        source[sam_source] = sha256(ROOT / sam_source)
    if "hrd" in config and config.get("teacher_provider") != "T0":
        name = "src/student/middle_teacher_hrd.py"
        source[name] = sha256(ROOT / name)
    if "hrd" in config:
        for name in ("src/student/direct_teacher_hrd.py", "src/middle_teacher/losses/hard_rank_distillation.py"):
            source[name] = sha256(ROOT / name)
    if kind == "S-HRD-SAM":
        source["src/student/hrd_only.py"] = sha256(ROOT / "src/student/hrd_only.py")
    return source


def train(config_path):
    import deepspeed
    path, cfg = load_formal_config(config_path)
    output = Path(cfg["output_dir"]).resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") != 0:
        raise RuntimeError("Formal Student requires one visible GPU")
    if int(os.environ.get("WORLD_SIZE", "0")) != 1:
        raise RuntimeError("Formal Student requires world_size=1")
    if not output.is_dir() or Path(os.environ.get("STUDENT_TRAIN_LOG", "")).resolve() != (output / "train.log").resolve():
        raise RuntimeError("Use scripts/train_student.sh to capture train.log")
    assets = validate_assets(cfg)
    kind = method_name(cfg)
    source = collect_student_source_identity(cfg)
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("Formal Student requires one GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    args = SimpleNamespace(
        img_size=cfg["img_size"], batch_size=cfg["local_pair_batch"],
        train_data_dir=str(ROOT / cfg["data_dir"] / "train"),
        num_workers=cfg["num_workers"], seed=cfg["seed"],
        epochs=cfg["epochs"], warmup_epochs=cfg["warmup_epochs"],
        min_lr_ratio=cfg["min_lr_ratio"],
    )
    loader = create_student_train_dataset_and_loader(args)
    if "ablation" in cfg or kind in ("S-HRD-SAM", "VKD-SAM") or "subspace_reference" in cfg or "random_basis_replicate" in cfg or cfg.get("teacher_provider") == "T0":
        from .shared_subspaces import configure_shared_data
        configure_shared_data(loader, cfg["seed"])
    student = StudentModel(
        ckpt_path=str((ROOT / cfg["pretrained_checkpoint"]).resolve()),
        temperature=cfg["task_loss"]["temperature"],
        model_name=cfg["model"],
    ).to(device)
    middle = supervision = gate = None
    random_seed = None
    if kind != "S0-INFONCE":
        if "provider_control" in cfg:
            from .provider_dimensions import load_provider
            middle = load_provider(cfg,device)
        elif cfg.get("teacher_provider") == "T0":
            from .direct_teacher_distillation import load_provider
            middle = load_provider(cfg, device)
        else:
            middle, _ = load_encoder(
                "middle", cfg["middle_checkpoint"], cfg["middle_config"],
                device, image_size=cfg["img_size"],
            )
        if middle.training or any(p.requires_grad for p in middle.parameters()):
            raise RuntimeError("Middle must be frozen and in eval mode")
        if kind == "VKD-SAM":
            from .vkd import VkDHead
            supervision = VkDHead().to(device)
        elif "ablation" in cfg:
            from .component_ablation import prepare
            calibration = torch.load(cfg["top_calibration"],map_location="cpu",weights_only=True)
            supervision, gate, random_seed = prepare(cfg, assets, calibration)
            supervision = supervision.to(device)
            if gate is not None:
                gate = gate.to(device)
        elif kind != "S-HRD-SAM":
            calibration = torch.load(
                cfg["top_calibration"], map_location="cpu", weights_only=True,
            )
            if cfg.get("teacher_provider") == "T0":
                from .direct_teacher_distillation import TeacherSubspaceSupervision
                random_seed = cfg["random_reference"]["random_basis_seed"]
                supervision = TeacherSubspaceSupervision(cfg, assets["teacher_checkpoint"], calibration).to(device)
                gate = ADSDGate(cfg["distillation"]["allocation"]["initial_d"]).to(device)
            elif kind == "S1-TSD":
                supervision = prepare_tsd(
                    cfg["supervision_asset"], assets["middle_checkpoint"], calibration,
                ).to(device)
            else:
                shared_tensors = None
                if "subspace_reference" in cfg:
                    from .shared_subspaces import load_reference
                    shared_tensors = load_reference(cfg["subspace_reference"], cfg, assets)
                random_seed = cfg["subspace_reference"]["random_basis_seed"] if shared_tensors is not None else secrets.randbits(63)
                supervision = prepare_adsd(
                    cfg["supervision_asset"], assets["middle_checkpoint"],
                    calibration, random_seed, cfg["seed"],
                    shared_tensors=shared_tensors,
                ).to(device)
                gate = ADSDGate(cfg["distillation"]["allocation"]["initial_d"]).to(device)
    if kind == "VKD-SAM":
        from .vkd import VkDTrainingModel
        model = VkDTrainingModel(student,supervision).to(device)
    else:
        model = StudentTrainingModel(student, supervision).to(device)
    optimizer = build_student_optimizer(model, cfg["lr"], cfg["weight_decay"])
    if kind == "VKD-SAM":
        from .vkd import protect_householder_optimizer
        protect_householder_optimizer(model,optimizer)
    # Finalize dtype groups before LambdaLR snapshots the optimizer groups.
    # The residual head stays FP32; every final group needs its own schedule.
    from .formal_top import prepare_student_precision_groups
    prepare_student_precision_groups(model, optimizer)
    scheduler = build_student_scheduler(optimizer, args, steps_per_epoch=len(loader))
    named = None
    if sam_enabled(cfg):
        from .distill_sam import collect_optimizer_trainable_params
        named = collect_optimizer_trainable_params(model, optimizer, middle.parameters())
    ds = dict(deepspeed_config({
        "batch_size": cfg["local_pair_batch"], "world_size": 1,
        "grad_accum_steps": cfg["grad_accum_steps"],
        "precision": cfg["backbone_precision"],
    }))
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler, config=ds,
    )
    assert_student_scheduler_groups(scheduler)
    selection_signature(
        engine.module.student, "student", cfg["img_size"],
        selection_batch_size=cfg["selection_eval_batch_size"],
    )
    if sam_enabled(cfg):
        from .distill_sam import assert_engine_ownership
        assert_engine_ownership(engine, named)
    gate_optimizer = gate_scheduler = None
    if gate is not None:
        gate_optimizer = torch.optim.AdamW([gate.d], lr=cfg["lr"], weight_decay=0.0)
        gate_scheduler = build_student_scheduler(
            gate_optimizer, args, steps_per_epoch=len(loader),
        )
        if any(gate.d is p for group in optimizer.param_groups for p in group["params"]):
            raise RuntimeError("GBW gate entered main Student optimizer")
    criterion = PairInfoNCE(cfg["task_loss"]["label_smoothing"])
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (output / "asset_identity.json").write_text(json.dumps(
        {"assets": assets, "source_identity": source}, indent=2) + "\n")
    print("STUDENT_METHOD=" + json.dumps({
        "method": kind, "resolution": cfg["img_size"],
        "local_pair_batch": cfg["local_pair_batch"], "world_size": 1,
        "selection_start_epoch": cfg["selection_start_epoch"],
        "selection_batch_size": cfg["selection_eval_batch_size"],
        "random_basis_seed": random_seed,
        "assets": assets,
    }), flush=True)
    if "subspace_reference" in cfg and not is_dimension_config(cfg):
        from .subspace_utils import tensor_sha256
        from .shared_subspaces import validate_tensors, NAMES
        validate_tensors(supervision.state_dict(), cfg["subspace_reference"])
        identity = {"source":cfg["subspace_reference"], "tensor_sha256":{name:tensor_sha256(getattr(supervision,name)) for name in NAMES}, "random_generation_count":supervision.random_basis_generation_count}
        (output / "subspace_identity.json").write_text(json.dumps(identity,indent=2)+"\n")
        print("SHARED_SUBSPACE_IDENTITY="+json.dumps(identity),flush=True)
        from .shared_subspaces import initialization_hashes
        initial = {"student":initialization_hashes(student),
                   "base_heads":initialization_hashes(supervision,base_heads_only=True),
                   "all_heads":initialization_hashes(supervision),
                   "training_seed":cfg["seed"],
                   "initialization_policy":"canonical head RNG rules retained; training/data seed from config",
                   "reference_use":"mean/Top/Random buffers only, never trained Student weights"}
        (output / "initialization_identity.json").write_text(json.dumps(initial,indent=2)+"\n")
    if "random_basis_replicate" in cfg:
        from .subspace_utils import tensor_sha256
        from .shared_subspaces import initialization_hashes
        identity = {"replicate": cfg["random_basis_replicate"], "random_basis_seed": random_seed,
                    "tensor_sha256": {name: tensor_sha256(getattr(supervision, name)) for name in ("teacher_mean", "top128_basis", "random32_basis")},
                    "random_generation_count": supervision.random_basis_generation_count}
        (output / "subspace_identity.json").write_text(json.dumps(identity, indent=2)+"\n")
        (output / "initialization_identity.json").write_text(json.dumps({"student": initialization_hashes(student), "base_heads": initialization_hashes(supervision, base_heads_only=True)}, indent=2)+"\n")
        print("INDEPENDENT_RANDOM32_IDENTITY="+json.dumps(identity), flush=True)
    if cfg.get("teacher_provider") == "T0":
        from .subspace_utils import tensor_sha256
        from .shared_subspaces import initialization_hashes
        identity = {"reference": cfg["random_reference"], "same_subspace_as_middle": False,
                    "tensor_sha256": {name: tensor_sha256(getattr(supervision, name)) for name in ("teacher_mean", "top128_basis", "random32_basis")},
                    "random_generation_count": supervision.random_basis_generation_count}
        (output / "subspace_identity.json").write_text(json.dumps(identity, indent=2)+"\n")
        (output / "initialization_identity.json").write_text(json.dumps({"student": initialization_hashes(student), "base_heads": initialization_hashes(supervision, base_heads_only=True)}, indent=2)+"\n")
        print("DIRECT_T0_SUPERVISION="+json.dumps(identity), flush=True)
    if kind == "S-HRD-SAM":
        from .shared_subspaces import initialization_hashes
        (output / "initialization_identity.json").write_text(json.dumps(
            {"student": initialization_hashes(student), "subspace_heads": None, "gate": None}, indent=2)+"\n")
    if cfg.get("ablation", {}).get("variant") == "b1_m0_provider":
        from .m0_student_ablation import write_identity
        write_identity(output,cfg,supervision,student)
    if kind == "VKD-SAM":
        from .shared_subspaces import initialization_hashes
        identity={"training_seed":cfg["seed"],"student":initialization_hashes(student),
            "all_heads":initialization_hashes(supervision),"feature":"shared BN before L2",
            "orthogonality_error":supervision.orthogonality_error(),"teacher_sha256":assets["middle_checkpoint"],
            "projector_weight_decay":0.,"warmup_epochs":cfg["distillation"]["warmup_epochs"]}
        (output / "initialization_identity.json").write_text(json.dumps(identity,indent=2)+"\n")
    if is_dimension_config(cfg):
        from .dimension_supervision import write_identity
        write_identity(output,cfg,supervision,student)
    best = float("-inf")
    for epoch in range(1, cfg["epochs"] + 1):
        engine.train()
        loader.batch_sampler.set_epoch(epoch - 1)
        for step, batch in enumerate(loader, start=1):
            images = torch.cat(batch[:2]).to(device, non_blocking=True)
            expected_shape = (2 * cfg["local_pair_batch"], 3, cfg["img_size"], cfg["img_size"])
            if tuple(images.shape) != expected_shape:
                raise ValueError("Student batch shape/resolution mismatch")
            if gate_optimizer is not None:
                gate_optimizer.zero_grad(set_to_none=True)
            if sam_enabled(cfg):
                if kind == "VKD-SAM":
                    from .vkd import sam_backward
                    loss,gate_loss,parts,audit = sam_backward(engine,middle,images,criterion,cfg,epoch,named)
                elif "ablation" in cfg:
                    from .component_ablation import sam_backward
                    loss, gate_loss, parts, audit = sam_backward(
                        engine,middle,images,criterion,cfg,epoch,gate,named,
                        identities=batch[2].to(device,non_blocking=True))
                elif kind == "S-HRD-SAM":
                    from .hrd_only import sam_hrd_backward
                    loss, gate_loss, parts, audit = sam_hrd_backward(
                        engine, middle, images, criterion, cfg, epoch, gate, named,
                        batch[2].to(device, non_blocking=True))
                elif "hrd" in cfg:
                    from .direct_teacher_hrd import sam_hrd_adsd_backward
                    loss, gate_loss, parts, audit = sam_hrd_adsd_backward(
                        engine, middle, images, criterion, cfg, epoch, gate, named,
                        batch[2].to(device, non_blocking=True),
                    )
                else:
                    loss, gate_loss, parts, audit = sam_adsd_backward(
                        engine, middle, images, criterion, cfg, epoch, gate, named,
                    )
                if not audit["PERTURB_RESTORE_EXACT"]:
                    raise RuntimeError("SAM restore failed")
            else:
                descriptors = engine(
                    images.to(next(engine.module.student.parameters()).dtype)
                )
                if kind == "S0-INFONCE":
                    loss = task_objective(
                        engine.module.student, descriptors, criterion,
                        cfg["local_pair_batch"],
                    )
                    parts = {"InfoNCE": loss.detach()}
                    gate_loss = None
                else:
                    with torch.no_grad():
                        targets = middle(images.to(torch.bfloat16)).detach().float()
                    if kind == "S1-TSD":
                        loss, parts = tsd_objective(
                            engine.module.student, supervision, descriptors, targets,
                            criterion, cfg, epoch,
                        )
                        gate_loss = None
                    elif "ablation" in cfg:
                        from .component_ablation import objective
                        loss, gate_loss, parts = objective(
                            engine.module.student, supervision, descriptors, targets,
                            criterion, cfg, epoch, gate)
                    else:
                        loss, gate_loss, parts = adsd_objective(
                            engine.module.student, supervision, descriptors, targets,
                            criterion, cfg, epoch, gate,
                        )
                engine.backward(loss)
            if gate is not None and gate.d.grad is not None:
                raise RuntimeError("GBW gate received main optimizer gradient")
            before_steps = engine.global_steps
            before_schedule = scheduler.last_epoch
            engine.step()
            assert_student_scheduler_groups(scheduler)
            if engine.global_steps != before_steps + 1 or scheduler.last_epoch != before_schedule + 1:
                raise RuntimeError("Student optimizer/scheduler step mismatch")
            if gate_loss is not None:
                gate_loss.backward()
                if gate.d.grad is None or not torch.isfinite(gate.d.grad):
                    raise RuntimeError("GBW gate gradient missing/nonfinite")
                gate_optimizer.step()
                gate_scheduler.step()
            if step <= 3 or step % 200 == 0:
                print("STUDENT_STEP=" + json.dumps({
                    "epoch": epoch, "step": step,
                    "learning_rates": {
                        group["name"]: float(group["lr"])
                        for group in scheduler.optimizer.param_groups
                    },
                    "loss": float(loss.detach()),
                    **{name: float(value) for name, value in parts.items()},
                }), flush=True)
        if should_select_epoch(epoch):
            engine.eval()
            best = select_student(
                engine, cfg, output, epoch, best, assets,
                supervision, gate, random_seed, source,
            )
        else:
            print(f"STUDENT_SELECTION_SKIPPED epoch={epoch}", flush=True)
    if not (output / "best_model.pth").is_file():
        raise RuntimeError("No best_model.pth after 30 epochs")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
