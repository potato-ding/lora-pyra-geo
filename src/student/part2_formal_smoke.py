"""One real TRAIN step using the formal P2 construction and unchanged DeepSpeed."""
import argparse,copy,json,math
from pathlib import Path
from types import SimpleNamespace
import torch
import deepspeed
from deepspeed.utils import safe_get_full_grad
from .artifacts import ROOT,file_sha256,write_json,deployment_state_dict
from .train import load_config,StudentTrainingModel,batch_loss,deepspeed_config
from .model import StudentModel
from .part1 import PartISupervision
from .data import create_student_train_dataset_and_loader
from .runtime import _seed_all,_seed_stst_worker
from .objective import PairInfoNCE
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .part2_integration import prepare_top,prepare_precision_groups,assert_precision,metadata
from src.evaluation.model_loader import load_encoder

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=load_config(args.config)
    preflight='P2_TOP_RMLP_MULTI_SEED' if cfg['seed'] in (1,2) else 'P2_S0'
    out=ROOT/'src/checkpoint/student/CERTIFIED_R224/_PREFLIGHT'/preflight/Path(cfg['output_dir']).name
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    assert not Path(cfg['output_dir']).exists()
    out.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(0);torch.set_num_threads(4)
    deepspeed.init_distributed(dist_backend='nccl');_seed_all(cfg['seed'])
    protected={k:file_sha256(cfg[k]) for k in ['middle_checkpoint','student_pretrained','stst_asset','original_stst_asset','p2_calibration_path']}
    loader=create_student_train_dataset_and_loader(SimpleNamespace(**cfg));loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(ckpt_path=cfg['student_pretrained'],temperature=cfg['temperature']).cuda()
    supervision=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],protected['middle_checkpoint'],128,'single32').cuda()
    original=copy.deepcopy(supervision).bfloat16()
    teacher,_=load_encoder('middle',cfg['middle_checkpoint'],cfg['middle_config'],torch.device('cuda:0'))
    prepare_top(supervision,cfg)
    assert all(torch.equal(v,supervision.projector_random.state_dict()[k]) for k,v in original.projector_random.state_dict().items())
    assert all(torch.equal(v,supervision.projector_top.linear.state_dict()[k]) for k,v in original.projector_top.linear.state_dict().items())
    model=StudentTrainingModel(student,supervision).cuda()
    optimizer=build_student_optimizer(model,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    membership={id(p):{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups for p in g['params']}
    prepare_precision_groups(model,optimizer,cfg)
    assert all({k:v for k,v in g.items() if k not in ['params','name']}=={k:v for k,v in membership[id(p)].items() if k!='name'} for g in optimizer.param_groups for p in g['params'])
    assert next(g for g in optimizer.param_groups if any(p is supervision.projector_top.alpha for p in g['params']))['weight_decay']==0
    scheduler=build_student_scheduler(optimizer,SimpleNamespace(**cfg),steps_per_epoch=len(loader))
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=deepspeed_config())
    assert_precision(engine)
    print('P2_DS_PRECISION_PASS=True',flush=True)
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    capture={}
    def capture_top(m,a,o):
        z=a[0].detach().float()
        base=torch.nn.functional.linear(z,m.linear.weight.float(),m.linear.bias.float())
        capture['ratio']=float((o[1].detach()-base).norm()/base.norm())
        capture['dtype']=str(o[1].dtype)
    hook=supervision.projector_top.register_forward_hook(capture_top)
    student_input_shapes=[]
    student_hook=student.register_forward_pre_hook(lambda module,args:student_input_shapes.append(tuple(args[0].shape)))
    engine.train();loader.batch_sampler.set_epoch(1);batch=next(iter(loader))
    images=torch.cat(batch[:2]).cuda();assert images.shape==(64,3,224,224)
    criterion=PairInfoNCE(label_smoothing=cfg['label_smoothing'])
    loss,parts=batch_loss(engine,teacher,images,32,criterion,cfg,1)
    student_hook.remove()
    assert student_input_shapes==[(64,3,224,224)]
    assert loss.dtype==torch.float32 and torch.isfinite(loss)
    assert all(torch.isfinite(v).all() for v in parts.values() if torch.is_tensor(v))
    assert capture['dtype']=='torch.float32' and math.isfinite(capture['ratio'])
    engine.backward(loss)
    groups={'student':student,'base':supervision.projector_top.linear,'random':supervision.projector_random,
            'residual':supervision.projector_top.residual,'alpha':supervision.projector_top.gate}
    grads={}
    for name,module in groups.items():
        gs=[safe_get_full_grad(p) for p in module.parameters() if p.requires_grad]
        assert gs and all(g is not None and torch.isfinite(g).all() for g in gs),name
        grads[name]=sum(float(g.float().square().sum()) for g in gs)**.5
        assert grads[name]>0,name
    assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    engine.step();assert_precision(engine)
    changed={name:sum(not torch.equal(before[n],p) for n,p in model.named_parameters() if id(p) in {id(q) for q in module.parameters()}) for name,module in groups.items()}
    assert all(n>0 for n in changed.values()),changed
    assert torch.isfinite(supervision.projector_top.alpha)
    hook.remove()
    state=deployment_state_dict(engine)
    assert set(state)==set(student.state_dict())
    assert all(file_sha256(cfg[k])==h for k,h in protected.items())
    report=dict(SMOKE_PASS=True,config=cfg,metadata=metadata(supervision),loss=float(loss),
        seed=cfg['seed'],sampler_seed=loader.batch_sampler.seed,
        student_forward_shapes=student_input_shapes,
        protected_sha256=protected,total_trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        components={k:None if v is None else float(v) for k,v in parts.items()},
        gradient_norms=grads,updated_parameter_counts=changed,teacher_grad=0,
        residual_base_ratio=capture['ratio'],alpha_after_step=float(supervision.projector_top.alpha),
        optimizer_steps=1,Student_BF16=True,projector_compute_FP32=True,alpha_weight_decay=0.,
        deployment_strip_pass=True,NaN=0,Inf=0,protected_unchanged=True,
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(out/'smoke_report.json',report);print(json.dumps(report),flush=True)
    torch.distributed.destroy_process_group()

if __name__=='__main__':main()
