"""Formal construction isolation and exact knowledge-buffer runtime checks."""
from contextlib import contextmanager
import torch

@contextmanager
def construction_rng(cfg, device):
    if cfg.get('artifact_contract')=='STUDENT_BEST_ONLY_V1':
        devices=[torch.device(device).index or 0] if torch.device(device).type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            yield
    else:
        yield

def snapshot_assets(supervision):
    if supervision is None:return {}
    return {name:value.detach().cpu().clone() for name,value in supervision.named_buffers()
            if (name in ('teacher_mean','top32_basis','random32_basis','random_b_basis') or name.startswith('bandwidth_'))}

def assert_assets_preserved(supervision, originals):
    current={} if supervision is None else dict(supervision.named_buffers())
    for name,original in originals.items():
        value=current[name]
        if original.dtype!=torch.float32 or value.dtype!=torch.float32 or not torch.equal(original,value.cpu()):
            raise RuntimeError('Knowledge asset changed during precision preparation: '+name)
