"""Train or evaluate diagnostic heads on a strictly frozen B0 backbone."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.distributed as dist

from src.diagnostics.runtime import (
    DATASET_CHOICES, build_formal_loaders, build_student,
    formal_pipeline_metrics, iter_loader_pairs, write_json,
)
from src.loss.blocks_infoNCE import Sample4GeoLoss
from src.diagnostics.probes import FrozenStudentProbe, parameter_audit
from src.training.student_train import (
    build_deepspeed_runtime_config, cast_images_to_model_dtype,
    gather_paired_views, unpack_sample4geo_batch,
)
from src.utils.initdist import try_init_dist
from src.utils.optimizer_and_scale import build_student_optimizer
from src.utils.run_logging import setup_rank0_run_log
from src.utils.scheduler import build_student_scheduler


PROBE_DIRS = {"P1": "P1-f3-linear", "P2": "P2-f4-linear", "P3": "P3-f4-mlp"}
TEST_FILENAMES = {
    "1652": "student_test_1652_best.json",
    "SUES-200": "student_test_sues200_best.json",
    "GTA-UAV": "student_test_gta_uav_best.json",
}


def tensor_collection_checksum(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def backbone_checksums(student):
    return {
        "parameters": tensor_collection_checksum(student.named_parameters()),
        "buffers": tensor_collection_checksum(student.named_buffers()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--probe", required=True, choices=("P1", "P2", "P3"))
    parser.add_argument("--student_checkpoint", default="src/checkpoint/student/B0-2GPU-3090/best_model.pth")
    parser.add_argument("--probe_checkpoint", default=None)
    parser.add_argument("--train_data_dir", default="data/U1652/train")
    parser.add_argument("--val_data_dir", default="data/U1652")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="1652")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--u1652_root", default=None)
    parser.add_argument("--sues_root", default=None)
    parser.add_argument("--gta_root", default=None)
    parser.add_argument("--output_root", default="src/checkpoint/student/diagnostic_probes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", default="configs/ds_student_baseline.json")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sues_height", choices=("150", "200", "250", "300", "all"), default="all")
    parser.add_argument("--sues_horizontal_flip", action="store_true")
    parser.add_argument("--gta_query_mode", choices=("D2S",), default="D2S")
    args = parser.parse_args(argv)
    args.temperature = 0.07
    args.prob_flip = 0.5
    args.amp = True
    args.grad_clip = 0.0
    return args


def save_probe(model, path, epoch, metrics=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "probe_type": model.probe_type,
        "probe": {k: v.detach().cpu() for k, v in model.probe.state_dict().items()},
        "metrics": metrics,
    }, path)


def load_probe(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.probe.load_state_dict(payload["probe"], strict=True)


@torch.no_grad()
def evaluate(model, args, device):
    loaders = build_formal_loaders(args)
    results = {}
    for height, direction, pair in iter_loader_pairs(args.dataset, loaders):
        key = f"{height or 'all'}/{direction}"
        results[key] = formal_pipeline_metrics(
            model, pair, device, args.dataset, f"Probe:{args.probe}:{args.dataset}:{key}",
            args.dataset == "SUES-200" and args.sues_horizontal_flip,
        )
    return results


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device, rank, _, world_size = try_init_dist() if args.deepspeed else (
        torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device), 0, 0, 1
    )
    output_dir = Path(args.output_root) / PROBE_DIRS[args.probe]
    if args.mode == "train":
        setup_rank0_run_log(str(output_dir), rank == 0)
    student = build_student(args.student_checkpoint, device)
    model = FrozenStudentProbe(student, args.probe).to(device)
    audit = parameter_audit(model)
    for name, value in audit.items():
        print(f"{name}={value}")
    if audit["backbone_trainable_params"] != 0:
        raise RuntimeError("frozen probe backbone has trainable parameters")

    if args.mode == "eval":
        checkpoint = args.probe_checkpoint or str(output_dir / "best_model.pth")
        load_probe(model, checkpoint)
        results = evaluate(model.eval(), args, device)
        if rank == 0:
            write_json(output_dir / TEST_FILENAMES[args.dataset], {
                "probe": args.probe, "checkpoint": checkpoint, "parameter_audit": audit, "results": results
            })
        return

    from src.dataset.datasets import create_student_train_dataset_and_loader

    if args.epochs != 30 or args.img_size != 224 or args.batch_size != 16 or args.grad_accum_steps != 1:
        print("[ProbeAdaptation] non-formal CLI override is active")
    print(f"[ProbeProtocol] local_pair_batch={args.batch_size} world_size={world_size} global_pair_batch={args.batch_size * world_size} grad_accum_steps={args.grad_accum_steps}")
    print("[ProbeAdaptation] B0 logit_scale is frozen and reused by symmetric InfoNCE")
    print("[ProbeAdaptation] optimizer=existing AdamW parameter grouping, probe parameters only")
    print("[ProbeAdaptation] scheduler=existing warmup+cosine protocol")
    print("[ProbeAdaptation] validation selection=D2S_R@1+S2D_R@1")
    train_loader = create_student_train_dataset_and_loader(args)
    optimizer = build_student_optimizer(model.probe, args.lr, args.weight_decay)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    probe_ids = {id(parameter) for parameter in model.probe.parameters() if parameter.requires_grad}
    if optimizer_ids != probe_ids:
        raise RuntimeError("optimizer must contain exactly the trainable probe parameters")
    scheduler = build_student_scheduler(optimizer, args, len(train_loader))
    engine = None
    if args.deepspeed:
        import deepspeed
        config = build_deepspeed_runtime_config(args.deepspeed_config, args, world_size)
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model, optimizer=optimizer, lr_scheduler=scheduler, config=config
        )
    freeze_before = backbone_checksums(model.student)
    criterion = Sample4GeoLoss(args.label_smoothing)
    best_r1_sum = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch - 1)
        epoch_loss = 0.0
        for batch in train_loader:
            images, meta = unpack_sample4geo_batch(batch, device)
            active = engine if engine is not None else model
            images = cast_images_to_model_dtype(active, images)
            descriptors = active(images)
            gathered, global_pairs = gather_paired_views(descriptors, meta["pair_batch_size"], with_grad=True)
            drone, satellite = gathered[:global_pairs], gathered[global_pairs:]
            loss = criterion(drone, satellite, model.student.logit_scale.exp())
            if engine is not None:
                engine.backward(loss); engine.step()
            else:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
            epoch_loss += float(loss.detach())
        validation_args = argparse.Namespace(**vars(args))
        validation_args.dataset = "1652"; validation_args.data_dir = args.val_data_dir
        validation_args.batch_size = args.val_batch_size
        metrics = evaluate(model.eval(), validation_args, device)
        r1_sum = metrics["all/D2S"]["R@1"] + metrics["all/S2D"]["R@1"]
        record = {"epoch": epoch, "loss": epoch_loss / max(1, len(train_loader)), "R1_sum": r1_sum, "results": metrics}
        history.append(record)
        if rank == 0:
            save_probe(model, output_dir / "last_model.pth", epoch, record)
            if r1_sum > best_r1_sum:
                best_r1_sum = r1_sum; save_probe(model, output_dir / "best_model.pth", epoch, record)
            write_json(output_dir / "best_metrics.json", {
                "selection_metric": "D2S R@1 + S2D R@1", "best_R1_sum": best_r1_sum,
                "parameter_audit": audit, "history": history,
            })
        print(f"[ProbeEpoch] epoch={epoch} loss={record['loss']:.6f} R1_sum={r1_sum:.6f}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    load_probe(model, output_dir / "best_model.pth")
    freeze_after = backbone_checksums(model.student)
    freeze_ok = freeze_before == freeze_after
    if not freeze_ok:
        raise RuntimeError("frozen backbone parameter or buffer checksum changed")
    if rank == 0:
        write_json(output_dir / "backbone_freeze_audit.json", {
            "before": freeze_before,
            "after": freeze_after,
            "unchanged": freeze_ok,
            "optimizer_contains_only_probe_parameters": optimizer_ids == probe_ids,
            "parameter_audit": audit,
        })
    for dataset in DATASET_CHOICES:
        eval_args = argparse.Namespace(**vars(args))
        eval_args.dataset = dataset
        eval_args.data_dir = {
            "1652": args.u1652_root,
            "SUES-200": args.sues_root,
            "GTA-UAV": args.gta_root,
        }[dataset]
        eval_args.batch_size = args.val_batch_size
        metrics = evaluate(model.eval(), eval_args, device)
        if rank == 0:
            write_json(output_dir / TEST_FILENAMES[dataset], {
                "probe": args.probe,
                "checkpoint": str(output_dir / "best_model.pth"),
                "parameter_audit": audit,
                "results": metrics,
            })


if __name__ == "__main__":
    main()
