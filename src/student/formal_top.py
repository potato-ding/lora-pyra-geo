"""Top128 residual head installation and FP32 precision groups for S3."""
import torch
from .part2 import install_residual_top,trainable_count
from .core_config import validate_config


def prepare_top(supervision,cfg):
    validate_config(cfg)
    calibration=torch.load(cfg['p2_calibration_path'],map_location='cpu',weights_only=True)
    assert calibration.dtype==torch.float32 and calibration.shape==(768,512) and torch.isfinite(calibration).all()
    supervision.bfloat16()
    install_residual_top(supervision,'rmlp',calibration)


def prepare_precision_groups(model,optimizer,cfg):
    if optimizer.state:raise RuntimeError('Precision grouping must precede the first optimizer step')
    model.bfloat16()
    groups=[]
    for group in optimizer.param_groups:
        by_dtype={}
        for parameter in group['params']:
            by_dtype.setdefault(parameter.dtype,[]).append(parameter)
        for dtype,parameters in by_dtype.items():
            groups.append(dict(group,params=parameters,name=group['name']+'_'+str(dtype)))
    optimizer.param_groups[:]=groups


def assert_precision(engine):
    supervision=engine.module.stst
    assert all(p.dtype==torch.bfloat16 for p in engine.module.student.parameters())
    assert all(p.dtype==torch.bfloat16 for p in supervision.projector_top.linear.parameters())
    assert all(p.dtype==torch.bfloat16 for p in supervision.projector_random.parameters())
    assert all(p.dtype==torch.float32 for p in supervision.projector_top.residual.parameters())
    assert supervision.projector_top.alpha.dtype==torch.float32


def metadata(supervision):
    top=supervision.projector_top
    result=dict(part='Part-II',research_axis='top_alignment_interface',
        training_only_head_params=trainable_count(supervision),
        p2_top_params=trainable_count(top),p2_residual_params=trainable_count(top.residual),
        p2_alpha_init=.001,p2_alpha_learnable=True,p2_alpha_weight_decay=0.,
        p2_mlp_hidden_dim=920,p2_initialization=top.initialization_audit,
        p2_parameter_storage='Student/base/Random BF16; residual/gate FP32',
        p2_optimizer_grouping='same AdamW decay/no-decay policy partitioned by dtype',
        p2_projector_compute='FP32',REFERENCE_USES_DEEPSPEED=True,P2_USES_DEEPSPEED=True)
    result.update(supervision.random_structure_metadata)
    result.update(random_training_params=trainable_count(supervision.projector_random),
        random_rmlp_calibration=None,random_parameter_storage='BF16',
        random_rmlp_beta_weight_decay=None)
    return result


def log_values(supervision):
    return dict(p2_alpha=float(supervision.projector_top.alpha.detach()))
