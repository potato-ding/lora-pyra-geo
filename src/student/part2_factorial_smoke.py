"""One real paired TRAIN step per fixed control on the original DeepSpeed path."""
import argparse, copy, json, math
from pathlib import Path
from types import SimpleNamespace
import torch
import deepspeed
from deepspeed.utils import safe_get_full_grad
from .artifacts import ROOT, file_sha256, write_json, deployment_state_dict
from .train import load_config, StudentTrainingModel, batch_loss, deepspeed_config
from .model import StudentModel
from .part1 import PartISupervision
from .part2_factorial import PREFLIGHT
from .part2_integration import prepare_top, prepare_precision_groups, assert_precision, metadata
from .data import create_student_train_dataset_and_loader
from .runtime import _seed_all, _seed_stst_worker
from .objective import PairInfoNCE
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from src.evaluation.model_loader import load_encoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    cfg=load_config(args.config);out=PREFLIGHT/Path(cfg['output_dir']).name
    assert not Path(cfg['output_dir']).exists()
    out.mkdir(parents=True,exist_ok=False)
    torch.cuda.set_device(0);torch.set_num_threads(4)
    deepspeed.init_distributed(dist_backend='nccl');_seed_all(0)
    protected={k:file_sha256(cfg[k]) for k in ['middle_checkpoint','student_pretrained','stst_asset','original_stst_asset','p2_calibration_path']}
    loader=create_student_train_dataset_and_loader(SimpleNamespace(**cfg));loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(ckpt_path=cfg['student_pretrained'],temperature=cfg['temperature']).cuda()
    sup=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],protected['middle_checkpoint'],128,'single32').cuda()
    original=copy.deepcopy(sup).bfloat16()
    teacher,_=load_encoder('middle',cfg['middle_checkpoint'],cfg['middle_config'],torch.device('cuda:0'))
    prepare_top(sup,cfg)
    for branch in ['top','random']:
        a=getattr(original,'projector_'+branch).linear;b=getattr(sup,'projector_'+branch).linear
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    model=StudentTrainingModel(student,sup).cuda()
    optimizer=build_student_optimizer(model,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    membership={id(p):{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups for p in g['params']}
    assert set(membership)=={id(p) for p in model.parameters() if p.requires_grad}
    prepare_precision_groups(model,optimizer,cfg)
    assert all({k:v for k,v in g.items() if k not in ['params','name']}=={k:v for k,v in membership[id(p)].items() if k!='name'} for g in optimizer.param_groups for p in g['params'])
    scheduler=build_student_scheduler(optimizer,SimpleNamespace(**cfg),steps_per_epoch=len(loader))
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=deepspeed_config())
    assert_precision(engine)
    groups={'student':student,'top_linear':sup.projector_top.linear,'random_linear':sup.projector_random.linear}
    capture={};hooks=[]
    for branch in ['top','random']:
        head=getattr(sup,'projector_'+branch)
        if hasattr(head,'residual'):
            groups[branch+'_residual']=head.residual;groups[branch+'_alpha']=head.gate
            assert membership[id(head.alpha)]['weight_decay']==0
        def hook(m,a,o,branch=branch):
            with torch.no_grad():
                z=a[0].detach().float();base=torch.nn.functional.linear(z,m.linear.weight.float(),m.linear.bias.float())
                diff=o[1].detach()-base;n=len(z)//2
                ratios=[float(diff[s].norm()/base[s].norm()) for s in [slice(0,n),slice(n,None)]]
                assert o[1].dtype==torch.float32 and all(math.isfinite(v) for v in ratios)
                active=getattr(m,'active_view','both') if hasattr(m,'residual') else 'none'
                for i,view in enumerate(['drone','satellite']):
                    if active in ('none', 'satellite' if view=='drone' else 'drone'):assert ratios[i]==0
                    else:assert ratios[i]>0
                capture[branch]={'ratios_drone_satellite':ratios,'active_view':active,'dtype':str(o[1].dtype)}
        hooks.append(head.register_forward_hook(hook))
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    engine.train();loader.batch_sampler.set_epoch(1);batch=next(iter(loader));images=torch.cat(batch[:2]).cuda()
    assert images.shape==(64,3,224,224)
    loss,parts=batch_loss(engine,teacher,images,32,PairInfoNCE(label_smoothing=cfg['label_smoothing']),cfg,1)
    assert loss.dtype==torch.float32 and torch.isfinite(loss)
    assert all(torch.isfinite(v).all() for v in parts.values() if torch.is_tensor(v))
    engine.backward(loss);grads={}
    for name,module in groups.items():
        gs=[safe_get_full_grad(p) for p in module.parameters() if p.requires_grad]
        assert gs and all(g is not None and torch.isfinite(g).all() for g in gs),name
        grads[name]=sum(float(g.float().square().sum()) for g in gs)**.5;assert grads[name]>0,name
    engine.step();assert_precision(engine)
    changed={name:sum(not torch.equal(before[n],p) for n,p in model.named_parameters() if id(p) in {id(q) for q in module.parameters()}) for name,module in groups.items()}
    assert all(v>0 for v in changed.values()),changed
    for h in hooks:h.remove()
    assert set(deployment_state_dict(engine))==set(student.state_dict())
    assert all(file_sha256(cfg[k])==h for k,h in protected.items())
    assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    report=dict(SMOKE_PASS=True,config=cfg,metadata=metadata(sup),protected_sha256=protected,
        loss=float(loss),components={k:None if v is None else float(v) for k,v in parts.items()},
        gradient_norms=grads,updated_parameter_counts=changed,view_masks=capture,
        trainable_parameter_counts={name:sum(p.numel() for p in module.parameters() if p.requires_grad) for name,module in groups.items()},
        trainable_total=sum(p.numel() for p in model.parameters() if p.requires_grad),
        optimizer_membership_complete=True,optimizer_policy_unchanged=True,optimizer_steps=1,
        deployment_strip_pass=True,teacher_grad=0,NaN=0,Inf=0,protected_unchanged=True,
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(out/'smoke_report.json',report);print('FACTORIAL_SMOKE_PASS=True',flush=True)
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()
