"""Unified two-GPU training for M0 through M4 Middle Teachers."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from src.data.middle_teacher import create_middle_teacher_train_dataset_and_loader
from src.middle_teacher.pure_adsd_checkpoint import ADSDCheckpointController
from src.middle_teacher.distributed import barrier, initialize_distributed, rank, world_size
from src.middle_teacher.e3_model import build_stage3_model
from src.middle_teacher.pure_adsd_config import load_formal_config, to_runtime_config, ROOT
from src.middle_teacher.pure_adsd_distillation import ADSDDistillationRuntime
from src.middle_teacher.adsd_supervision import (load_assets, MiddleSubspaceSupervision,
    calibration_descriptors, add_head_optimizer_groups, deployment_only)
from src.middle_teacher.model import build_middle_teacher
from src.middle_teacher.optimizer import build_middle_teacher_optimizer
from src.middle_teacher.runtime import initialize_deepspeed, warmup_cosine_scheduler
from src.middle_teacher.selection import select_and_save
from src.evaluation.precision_contract import selection_signature
from src.middle_teacher.adsd_sam import GradientSummary, sam_backward
from src.middle_teacher.losses.pair_infonce import pair_infonce
from src.utils.gather_features_and_labels_and_views import GatherLayer, concat_all_gather

FIRST_MIDDLE_SELECTION_EPOCH = 6


def compute_objective(engine, model, kd, images, ids, config, step):
    local = config["data"]["local_pair_batch"]
    semantic = "adaptive_bridge_v2" in config["distillation"]
    output = engine(images, return_layer_features=semantic)
    descriptor = output["final_descriptor"] if semantic else output
    if descriptor.dtype != torch.float32 or descriptor.shape != (2 * local, 768):
        raise RuntimeError("Middle descriptor contract")
    md = torch.cat(GatherLayer.apply(descriptor[:local]), dim=0)
    ms = torch.cat(GatherLayer.apply(descriptor[local:]), dim=0)
    global_ids = concat_all_gather(ids)
    if global_ids.unique().numel() != config["data"]["global_pair_batch"]:
        raise RuntimeError("Global pair identity collision")
    task, d2s, s2d = pair_infonce(md, ms, model.logit_scale)
    if kd is None:
        return task, task, d2s, s2d, {}
    total, stats = kd.compose_all(
        task, md, ms, images, global_ids, model, step,
        output if semantic else None,
    )
    return total, task, d2s, s2d, stats



def should_select_epoch(epoch: int) -> bool:
    """Epochs 1–5 train only; U1652 selection starts at epoch 6."""
    return epoch >= FIRST_MIDDLE_SELECTION_EPOCH


def run(config_path, expected_gpus, teacher_chunk_size, smoke_output=None, smoke_steps=0):
    gpu_ids = expected_gpus.split(",")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpus
            or len(gpu_ids) != 2 or len(set(gpu_ids)) != 2
            or not all(g.isdigit() for g in gpu_ids)):
        raise ValueError("Exactly two explicit training GPUs are required")
    path, public = load_formal_config(config_path)
    if not public.get("output_dir"):
        raise ValueError("Set output_dir in the public config")
    if "adsd" in public and not public.get("teacher_checkpoint"):
        raise ValueError("Set teacher_checkpoint in the KD config")
    if "adsd" in public and not Path(public["teacher_checkpoint"]).is_file():
        raise FileNotFoundError(public["teacher_checkpoint"])
    data_root = Path(public["data_dir"])
    if not (data_root / "train" / "satellite").is_dir() or not (data_root / "train" / "drone").is_dir():
        raise FileNotFoundError(f"Middle training pairs under {data_root}")
    local_rank = initialize_distributed()
    if world_size() != 2:
        raise RuntimeError("Formal Middle training requires exactly two ranks")
    config = to_runtime_config(path)
    if smoke_steps and (smoke_steps!=3 or not smoke_output or Path(smoke_output).resolve().parent!=Path('/tmp')):
        raise ValueError('Smoke requires 3 steps and an isolated /tmp output')
    if smoke_output:
        public['output_dir']=smoke_output;config['checkpoint']['output_dir']=smoke_output
    output = Path(public["output_dir"]).resolve()
    if rank() == 0:
        if output.exists() and any(item.name != "train.log" for item in output.iterdir()):
            raise FileExistsError(f"Output directory is not fresh: {output}")
        if Path(os.environ.get("MIDDLE_TRAIN_LOG", "")).resolve() != (output / "train.log").resolve():
            raise RuntimeError("Training log/output directory mismatch")
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(public, indent=2) + "\n")
    barrier()
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device=torch.device('cuda',local_rank)
    tensors,asset_audit=load_assets(public)
    from src.source_contract import source_identity
    from src.middle_teacher.checkpoint import sha256
    source=source_identity('t2m')
    for path in (ROOT/'src/middle_teacher').glob('adsd_*.py'):source[str(path.relative_to(ROOT))]=sha256(path)
    for path in (ROOT/'src/middle_teacher').glob('pure_adsd_*.py'):source[str(path.relative_to(ROOT))]=sha256(path)
    for name in ('src/student/adsd.py','src/student/part2.py','src/student/formal_supervision.py',
                 'src/student/direct_teacher_assets.py','src/student/subspace_utils.py'):
        source[name]=sha256(ROOT/name)
    semantic = "adaptive_bridge_v2" in config["distillation"]
    model = build_stage3_model(config) if semantic else build_middle_teacher(config)
    model.to(device).bfloat16()
    calibration=calibration_descriptors(model,config,device)
    optimizer, audit = build_middle_teacher_optimizer(model, config["optimizer"])
    with torch.random.fork_rng(devices=[local_rank]):
        model.adsd_supervision=MiddleSubspaceSupervision(tensors,calibration,config['adsd']['head_seed']).to(device)
    add_head_optimizer_groups(model,optimizer,config)
    if rank()==0:
        torch.save(calibration,output/'middle_top_calibration.pt')
        (output/'adsd_asset_identity.json').write_text(json.dumps(asset_audit,indent=2)+'\n')
        from src.student.subspace_utils import tensor_sha256
        (output/'adsd_initialization.json').write_text(json.dumps({
            'calibration_sha256':tensor_sha256(calibration),
            'heads':{k:tensor_sha256(v.cpu()) for k,v in model.adsd_supervision.state_dict().items()},
            'calibration':model.adsd_supervision.initialization_audit},indent=2)+'\n')
    _, loader = create_middle_teacher_train_dataset_and_loader(config)
    total_steps = len(loader) * config["experiment"]["epochs"]
    warmup_steps = int(total_steps * public["warmup_ratio"])
    config["scheduler"] = {
        "type": "cosine",
        "warmup_steps": warmup_steps,
        "total_optimizer_steps": total_steps,
    }
    scheduler = warmup_cosine_scheduler(optimizer, total_steps, warmup_steps)
    engine, optimizer, scheduler, _ = initialize_deepspeed(model, optimizer, scheduler, config)
    with deployment_only(engine.module):
        selection_signature(engine.module,'middle',public['img_size'],selection_batch_size=16)
    device = torch.device("cuda", local_rank)
    kd = None
    if "adsd" in public:
        with torch.random.fork_rng(devices=[local_rank]):
            kd = ADSDDistillationRuntime(
                config, public["teacher_checkpoint"], device, teacher_chunk_size,
            )
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if any(id(p) in optimizer_ids for p in kd.teacher.parameters()):
            raise RuntimeError("Teacher parameters entered Middle optimizer")
        engine.module.distillation_teacher_identity = {
            "checkpoint": kd.audit["checkpoint"],
            "sha256": kd.audit["sha256"],
            "checkpoint_metadata": kd.audit["checkpoint_metadata"],
        }
    engine.module.selection_image_size = public["img_size"]
    gate_optimizer=torch.optim.AdamW(kd.gate.parameters(),lr=config['adsd']['gate_lr'],weight_decay=0.)
    gate_scheduler=warmup_cosine_scheduler(gate_optimizer,total_steps,warmup_steps)
    controller = ADSDCheckpointController(output,config,public_config=public,
        supervision=engine.module.adsd_supervision,gate=kd.gate,assets=asset_audit,source=source)
    if rank() == 0:
        print("MIDDLE_RUNTIME=" + json.dumps({
            "experiment": public["experiment_id"], "image_size": public["img_size"],
            "training_world_size": world_size(), "local_pair_batch": public["local_pair_batch"],
            "global_pair_batch": public["global_pair_batch"],
            "teacher_checkpoint": public.get("teacher_checkpoint"),
            "sam": config["sam"],
            "distillation_components": sorted(config["distillation"]) + ["adsd"],
            "optimizer": audit, "scheduler": config["scheduler"],
            "selection": "U1652_D2S_R1+U1652_S2D_R1",
            "selection_start_epoch": FIRST_MIDDLE_SELECTION_EPOCH,
            "selection_world_size": 1,
            "selection_batch_size": config["checkpoint"]["selection_eval_batch_size"],
        }), flush=True)

    step = 0
    for epoch in range(1, config["experiment"]["epochs"] + 1):
        kd.epoch=epoch
        loader.batch_sampler.set_epoch(epoch - 1)
        engine.train()
        running = 0.0
        sam_summary = GradientSummary() if "sam" in public else None
        for batch_index, (drone, satellite, labels, _pids) in enumerate(loader):
            images = torch.cat((drone, satellite), 0).to(
                device=device, dtype=next(engine.module.parameters()).dtype,
            )
            expected = (2 * public["local_pair_batch"], 3, public["img_size"], public["img_size"])
            if tuple(images.shape) != expected:
                raise RuntimeError(f"Middle image geometry mismatch: {tuple(images.shape)}")
            ids = torch.as_tensor(labels, device=device, dtype=torch.long)
            gate_optimizer.zero_grad(set_to_none=True)
            before_steps = engine.global_steps
            before_schedule = scheduler.last_epoch
            if sam_summary is None:
                engine.zero_grad()
                loss, task, d2s, s2d, stats = compute_objective(
                    engine, engine.module, kd, images, ids, config, step,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite Middle objective")
                engine.backward(loss)
            else:
                loss, task, d2s, s2d, stats, sam_stats = sam_backward(
                    engine, kd, images, ids, step, config["sam"], audit_ranks=step<3,
                )
                sam_summary.add(sam_stats)
            if kd.gate.d.grad is not None:raise RuntimeError('GBW leaked into main backward')
            engine.step()
            kd.gate_loss.backward()
            if kd.gate.d.grad is None or not torch.isfinite(kd.gate.d.grad):raise FloatingPointError('Gate gradient invalid')
            dist.all_reduce(kd.gate.d.grad);kd.gate.d.grad.div_(world_size())
            gate_optimizer.step();gate_scheduler.step()
            gate_ref=kd.gate.d.detach().clone();dist.broadcast(gate_ref,0)
            if not torch.equal(gate_ref,kd.gate.d):raise RuntimeError('GBW rank mismatch')
            step += 1
            if engine.global_steps != before_steps + 1 or scheduler.last_epoch != before_schedule + 1:
                raise RuntimeError("Expected one optimizer and scheduler step per logical batch")
            with torch.no_grad():
                engine.module.logit_scale.clamp_(0, math.log(100))
            running += float(loss.detach())
            if rank() == 0 and (batch_index < 3 or step % 20 == 0):
                print("MIDDLE_STEP=" + json.dumps({
                    "experiment": public["experiment_id"], "epoch": epoch, "step": step,
                    "loss": float(loss.detach()), "task": float(task.detach()),
                    "d2s": float(d2s.detach()), "s2d": float(s2d.detach()),
                    "kd": stats,
                    **({"sam": sam_stats} if sam_summary is not None else {}),
                }), flush=True)
            if smoke_steps and step>=smoke_steps:
                if rank()==0:
                    # Synthetic selection is only a serialization/reload smoke, never a result.
                    engine.module.sam_epoch_diagnostics=sam_summary.result()
                    engine.eval()
                    with deployment_only(engine.module):
                        from src.evaluation.model_loader import EvaluationEncoder
                        eval_images=images[:16].float()
                        with torch.no_grad():expected_descriptor=EvaluationEncoder(engine.module,768).eval()(eval_images)
                        metrics={'D2S_R1':0.,'D2S_R5':0.,'D2S_AP':0.,'S2D_R1':0.,'S2D_R5':0.,'S2D_AP':0.,'R1_sum':0.}
                        controller.save_best_if_improved(engine,6,step,metrics)
                        from src.evaluation.model_loader import load_encoder
                        from src.middle_teacher.pure_adsd_checkpoint import validate_adsd_checkpoint
                        payload=torch.load(output/'best_model.pth',map_location='cpu',weights_only=True)
                        validate_adsd_checkpoint(payload)
                        reloaded,_=load_encoder('middle',str(output/'best_model.pth'),None,device,image_size=256)
                        if any(not torch.equal(v,reloaded.model.state_dict()[n]) for n,v in engine.module.state_dict().items()):
                            raise RuntimeError("Smoke reloaded weights changed")
                        with torch.no_grad():actual=reloaded(eval_images)
                        if not torch.equal(actual,expected_descriptor):raise RuntimeError('Smoke strict reload descriptor mismatch')
                    (output/'SMOKE_ONLY.json').write_text(json.dumps({'synthetic_selection':True,'not_a_trained_result':True,'strict_reload_exact':True}))
                    print('MIDDLE_ADSD_STRICT_RELOAD_EXACT=PASS',flush=True)
                    print('MIDDLE_ADSD_THREE_BATCH_SMOKE=PASS',flush=True)
                barrier();dist.destroy_process_group();return
        if sam_summary is not None:
            engine.module.sam_epoch_diagnostics = sam_summary.result()
        metrics, improved = None, False
        if should_select_epoch(epoch):
            engine.eval()
            with deployment_only(engine.module):
                metrics, improved = select_and_save(engine,controller,config,epoch,step,device)
        if rank() == 0:
            print("MIDDLE_EPOCH=" + json.dumps({
                "experiment": public["experiment_id"], "epoch": epoch,
                "train_loss": running / len(loader),
                "selection": metrics if metrics is not None else "NOT_RUN",
                "best_epoch": controller.best_epoch, "improved": improved,
            }), flush=True)
        barrier()
    if rank() == 0 and (not 6 <= controller.best_epoch <= config["experiment"]["epochs"]
                        or not (output / "best_model.pth").is_file()):
        raise RuntimeError("Formal Middle run ended without a valid epoch-6+ best checkpoint")
    barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-gpus", required=True)
    parser.add_argument("--teacher-chunk-size", type=int, default=4)
    parser.add_argument("--smoke-output")
    parser.add_argument("--smoke-steps",type=int,default=0)
    args = parser.parse_args()
    run(args.config, args.expected_gpus, args.teacher_chunk_size,args.smoke_output,args.smoke_steps)


if __name__ == "__main__":
    main()
