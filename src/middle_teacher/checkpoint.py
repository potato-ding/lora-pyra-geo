"""Portable model checkpoints and DeepSpeed resume state for Middle Teacher."""
from __future__ import annotations

import hashlib
import inspect
import math
import os
import shutil
from pathlib import Path
import torch

from .distributed import barrier, rank


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_load(path, map_location="cpu"):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def unwrap_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "net"):
            if isinstance(payload.get(key), dict):
                return payload[key]
    return payload


def strip_distributed_prefix(state):
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def raw_model(model_or_engine):
    return getattr(model_or_engine, "module", model_or_engine)


def portable_state(model_or_engine):
    return {key: value.detach().cpu() for key, value in raw_model(model_or_engine).state_dict().items()}


def load_middle_teacher_checkpoint(model, path, strict=True):
    payload = safe_load(path, "cpu")
    state = strip_distributed_prefix(unwrap_state_dict(payload))
    model_keys, state_keys = set(model.state_dict()), set(state)
    missing, unexpected = sorted(model_keys-state_keys), sorted(state_keys-model_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(f"strict checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    result = model.load_state_dict(state, strict=strict)
    return {"checkpoint": str(Path(path).resolve()), "sha256": sha256(path),
            "state_key_count": len(state), "parameter_count": sum(p.numel() for p in model.parameters()),
            "missing_keys": missing, "unexpected_keys": unexpected,
            "post_missing_keys": list(result.missing_keys), "post_unexpected_keys": list(result.unexpected_keys)}


def selection_score(metrics):
    return float(metrics["D2S_R1"]) + float(metrics["S2D_R1"])


class CheckpointController:
    def __init__(self, output_dir, objective="pair_infonce_hierarchical_distillation"):
        self.output_dir = Path(output_dir); self.objective = objective
        self.best_score = -math.inf; self.best_epoch = 0; self.best_metrics = None

    def _save_portable(self, model_or_engine, filename, epoch, global_step, metrics=None):
        if rank() != 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": portable_state(model_or_engine), "epoch": int(epoch),
                    "global_step": int(global_step), "train_objective": self.objective,
                    "best_score": float(self.best_score), "best_epoch": int(self.best_epoch),
                    "best_metrics": self.best_metrics, "metrics": metrics}, self.output_dir / filename)

    def save_last(self, model_or_engine, epoch, global_step, metrics=None):
        self._save_portable(model_or_engine, "last_model.pth", epoch, global_step, metrics)

    def save_best_if_improved(self, model_or_engine, epoch, global_step, metrics):
        score = selection_score(metrics)
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch, self.best_metrics = score, int(epoch), dict(metrics)
            self._save_portable(model_or_engine, "best_model.pth", epoch, global_step, metrics)
        return improved

    def save_resume(self, engine, epoch, global_step):
        root = self.output_dir / "resume_state"; tag = f"step_{int(global_step)}"
        client = {"epoch": int(epoch), "global_step": int(global_step),
                  "best_score": float(self.best_score), "best_epoch": int(self.best_epoch),
                  "best_metrics": self.best_metrics}
        engine.save_checkpoint(str(root), tag=tag, client_state=client, save_latest=True)
        barrier()
        if rank() == 0 and root.exists():
            for candidate in root.glob("step_*"):
                if candidate.name != tag and candidate.is_dir(): shutil.rmtree(candidate)
        barrier()

    def resume_training(self, engine, path=None):
        root = str(path or self.output_dir / "resume_state")
        load_path, client = engine.load_checkpoint(root, load_optimizer_states=True,
                                                   load_lr_scheduler_states=True)
        if not load_path: raise RuntimeError(f"no DeepSpeed resume state under {root}")
        self.best_score = float(client.get("best_score", -math.inf))
        self.best_epoch = int(client.get("best_epoch", 0)); self.best_metrics = client.get("best_metrics")
        return client


safe_torch_load = safe_load
