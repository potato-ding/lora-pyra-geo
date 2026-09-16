"""Fixed, training-only BN cross-view geometry consistency control."""
from contextlib import contextmanager
import json
from pathlib import Path
import torch
from torch import nn

NAME = 'FINAL-ADUAL-BNCC-S0'
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)


def validate_config(cfg):
    from .artifacts import ROOT
    reference = json.loads((ROOT/'configs/student/certified_r224/p2_top_rmlp_s0.json').read_text())
    extra = {'bncc_enabled', 'bncc_lambda'}
    identity = {'experiment_name', 'output_dir', 'sealed_provenance_file'}
    if cfg.get('bncc_enabled') is not True or type(cfg.get('bncc_lambda')) not in (int,float) or cfg['bncc_lambda'] != 1.0:
        raise ValueError('Only enabled BNCC with fixed lambda=1.0 is certified')
    if cfg.get('experiment_name') != NAME or Path(cfg['output_dir']).resolve() != ROOT/'src/checkpoint/student/CERTIFIED_R224'/NAME:
        raise ValueError('Only FINAL-ADUAL-BNCC-S0 is certified')
    if {k:v for k,v in cfg.items() if k not in extra|identity} != {k:v for k,v in reference.items() if k not in identity}:
        raise ValueError('BNCC must inherit every Final Top-RMLP S0 field exactly')
    from .part2_integration import validate_config as validate_reference
    validate_reference(reference)
    return True


@contextmanager
def running_stat_shadow(student):
    """Use current post-MAIN running buffers and restore every module's mode."""
    modes = [(m,m.training) for m in student.modules()]
    buffers = [(m,{k:v.detach().clone() for k,v in m.named_buffers(recurse=False)})
               for m in student.modules() if isinstance(m,BN_TYPES)]
    if not buffers:
        raise ValueError('Expected Student BatchNorm modules')
    try:
        student.eval()
        assert all(not m.training for m in student.modules())
        with torch.no_grad():
            yield
    finally:
        unchanged = all(torch.equal(dict(m.named_buffers(recurse=False))[k],v) for m,b in buffers for k,v in b.items())
        for m,mode in modes:
            m.training = mode
        if not unchanged:
            raise RuntimeError('BN_SHADOW_BUFFER_IMMUTABILITY_PASS=False')


def geometry_loss(batch_descriptor, running_descriptor, pairs):
    if batch_descriptor.shape != (2*pairs,512) or running_descriptor.shape != batch_descriptor.shape:
        raise ValueError('BNCC expects paired normalized 512D descriptors')
    if batch_descriptor.dtype != torch.float32 or running_descriptor.dtype != torch.float32:
        raise ValueError('BNCC geometry must remain FP32')
    batch = batch_descriptor[:pairs] @ batch_descriptor[pairs:].T
    target = (running_descriptor[:pairs] @ running_descriptor[pairs:].T).detach()
    gap = batch-target
    loss = gap.square().mean()
    # Unit-normalized dot products are in [-1,1], hence MSE cannot exceed 4.
    if not torch.isfinite(loss) or float(loss.detach()) > 4.001:
        raise FloatingPointError('Nonfinite or out-of-bound BNCC geometry loss')
    return loss, gap.detach().abs().mean()


def loss_with_shadow(student, images, main_descriptor, pairs):
    if pairs != 32 or images.shape != (64,3,224,224):
        raise ValueError('BNCC requires canonical mixed-view N64')
    if not student.training or not all(m.training for m in student.modules() if isinstance(m,BN_TYPES)):
        raise RuntimeError('MAIN Student and BN must retain train mode')
    version = images._version
    with running_stat_shadow(student):
        shadow = student(images).detach()
    assert images._version == version
    assert not shadow.requires_grad and shadow.grad_fn is None
    loss,gap = geometry_loss(main_descriptor,shadow,pairs)
    return loss,dict(loss_bncc=loss.detach(),mean_abs_similarity_gap=gap,
        shadow_buffer_immutable=True,shadow_stop_grad=True,main_batch_stat=True)


class EpochLog:
    """Detached sums for logging only; never affects training reductions."""
    KEYS = ('loss_total','loss_infonce','loss_top','loss_random','loss_bncc','mean_abs_similarity_gap')
    def __init__(self):
        self.total = None; self.count = 0
    def add(self, loss, parts):
        values = [loss,parts['infonce'],parts['top_loss'],parts['random_loss'],parts['loss_bncc'],parts['mean_abs_similarity_gap']]
        row = torch.stack([v.detach().float() for v in values])
        if not torch.isfinite(row).all() or not parts['shadow_buffer_immutable']:
            raise FloatingPointError('BNCC epoch logging sanity failed')
        self.total = row if self.total is None else self.total+row
        self.count += 1
    def finish(self,epoch,lr,alpha):
        if not self.count:
            raise RuntimeError('No training batches')
        return dict(record='BNCC_EPOCH',epoch=epoch,steps=self.count,lr=lr,alpha_top=float(alpha.detach()),
            **dict(zip(self.KEYS,(self.total/self.count).cpu().tolist())),
            SHADOW_BUFFER_IMMUTABLE=True,SHADOW_GRADIENT_TO_BACKBONE=False,BNCC_LAMBDA=1.0)
