"""Shared Student model wrapper and DeepSpeed validation state contract."""
import json
import torch
import torch.distributed as dist
from torch import nn

class StudentTrainingModel(nn.Module):
    def __init__(self,student,supervision=None):
        super().__init__();self.student=student;self.stst=supervision
    def forward(self,images):return self.student(images)

def deepspeed_config(cfg=None):
    batch=32 if cfg is None else cfg['batch_size']
    world=1 if cfg is None else cfg['world_size']
    accumulation=1 if cfg is None else cfg['grad_accum_steps']
    precision='bfloat16' if cfg is None else cfg['precision']
    return {'train_batch_size':batch*world*accumulation,'train_micro_batch_size_per_gpu':batch,'gradient_accumulation_steps':accumulation,
        'zero_optimization':{'stage':1},'zero_allow_untested_optimizer':True,
        'bf16':{'enabled':precision=='bfloat16'},'fp16':{'enabled':precision=='float16'},'gradient_clipping':0.0,'steps_per_print':1000000}

@torch.no_grad()
def sync_student_buffers_from_rank0(student):
    """Choose rank0 buffers before validation; never synchronize parameters here."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    for buffer in student.buffers():
        dist.broadcast(buffer, src=0)
    dist.barrier()

@torch.no_grad()
def assert_student_validation_state_synced(student, epoch):
    """Fail collectively before selection if parameters or buffers diverge."""
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if not distributed:
        return dict(epoch=epoch, bn_protocol="single_rank_native_bn", cross_rank_buffer_sync=False, assertion_noop=True)
    maxima = {}
    for name, tensors in (("parameter", student.parameters()), ("buffer", student.buffers())):
        groups = {}
        for tensor in tensors:
            groups.setdefault((tensor.device, tensor.dtype), []).append(tensor.detach().reshape(-1))
        maximum = 0.0
        for values in groups.values():
            local = torch.cat(values)
            low, high = local.clone(), local.clone()
            if distributed:
                dist.all_reduce(low, op=dist.ReduceOp.MIN)
                dist.all_reduce(high, op=dist.ReduceOp.MAX)
            difference = (high.double() - low.double()).abs()
            if not torch.isfinite(difference).all():
                raise RuntimeError("Nonfinite Student validation " + name + " state")
            maximum = max(maximum, difference.max().item() if difference.numel() else 0.0)
        maxima[name + "_rank_max_diff"] = maximum
    if any(value != 0 for value in maxima.values()):
        raise RuntimeError("Student validation state differs across ranks: " + repr(maxima))
    record = dict(epoch=epoch, buffers_synced=True, canonical_buffer_source="rank0", **maxima)
    if not distributed or dist.get_rank() == 0:
        print("[ValidationState] " + json.dumps(record), flush=True)
    return record
