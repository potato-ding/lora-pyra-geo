"""Fixed eight-batch gradient gate and separate fresh DeepSpeed smoke for BNCC."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import deepspeed
from deepspeed.utils import safe_get_full_grad
from .artifacts import ROOT,file_sha256,write_json,deployment_state_dict
from .train import load_config,StudentTrainingModel,batch_loss,deepspeed_config
from .model import StudentModel
from .part1 import PartISupervision
from .part2_integration import prepare_top,prepare_precision_groups,assert_precision
from .data import create_student_train_dataset_and_loader
from .runtime import _seed_all,_seed_stst_worker
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .objective import PairInfoNCE
from .bncc import geometry_loss
from src.evaluation.model_loader import load_encoder

OUT=ROOT/'src/checkpoint/student/CERTIFIED_R224/_PREFLIGHT/FINAL_ADUAL_BNCC_S0'


class GradientEngine(torch.nn.Module):
    """Same training module/precision without DeepSpeed's one-backward reduction hooks."""
    def __init__(self,module):
        super().__init__();self.module=module
    def forward(self,images):return self.module(images)


def summary(values):
    a=np.array(values,dtype=float)
    assert np.isfinite(a).all()
    return dict(mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()))


def grad_compare(a,b):
    dot=sum((x.float()*y.float()).sum(dtype=torch.float64) for x,y in zip(a,b))
    aa=sum(x.float().square().sum(dtype=torch.float64) for x in a)
    bb=sum(x.float().square().sum(dtype=torch.float64) for x in b)
    return dict(ratio=float((aa/bb).sqrt()),cosine=float(dot/(aa*bb).sqrt()),norm_a=float(aa.sqrt()),norm_b=float(bb.sqrt()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['gradient','smoke'],required=True)
    args=p.parse_args();cfg=load_config(args.config)
    assert not Path(cfg['output_dir']).exists()
    OUT.mkdir(parents=True,exist_ok=True)
    result_path=OUT/(args.mode+'_report.json');assert not result_path.exists()
    if args.mode=='smoke':assert json.loads((OUT/'gradient_report.json').read_text())['PREFLIGHT_PASS']
    torch.cuda.set_device(0);torch.set_num_threads(4);deepspeed.init_distributed(dist_backend='nccl');_seed_all(0)
    protected={k:file_sha256(cfg[k]) for k in ['middle_checkpoint','student_pretrained','stst_asset','original_stst_asset','p2_calibration_path']}
    assert protected['middle_checkpoint']=='1f5dd3a94e38d5e79bfff05b407959195eb59b9b9359f2380727f6a68fed3d78'
    assert protected['student_pretrained']=='d645a2de5481c9aac1639d0e97b04cd4bdb0df9d7347920b132dd0ed45de8b39'
    assert protected['stst_asset']==cfg['extended_stst_asset_sha256']
    loader=create_student_train_dataset_and_loader(SimpleNamespace(**cfg));loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(ckpt_path=cfg['student_pretrained'],temperature=cfg['temperature']).cuda()
    supervision=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],protected['middle_checkpoint'],128,'single32').cuda()
    teacher,_=load_encoder('middle',cfg['middle_checkpoint'],cfg['middle_config'],'cuda:0')
    prepare_top(supervision,cfg)
    model=StudentTrainingModel(student,supervision).cuda()
    optimizer=build_student_optimizer(model,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    prepare_precision_groups(model,optimizer,cfg)
    scheduler=build_student_scheduler(optimizer,SimpleNamespace(**cfg),len(loader))
    if args.mode=='smoke':
        engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=deepspeed_config())
    else:
        # autograd.grad of three independent objectives must not invoke DeepSpeed's
        # single-total-backward communication epilogue. No optimizer step is taken.
        engine=GradientEngine(model)
    assert_precision(engine);criterion=PairInfoNCE(cfg['label_smoothing'])
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    engine.train();loader.batch_sampler.set_epoch(1)
    batches=[];capture={};hooks=[]
    def student_hook(m,a,o):
        capture.setdefault('student',[]).append(dict(tensor=a[0],output=o,train=m.training,
            bn_modes=[x.training for x in m.modules() if isinstance(x,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d))]))
    hooks.append(student.register_forward_hook(student_hook))
    hooks.append(criterion.register_forward_hook(lambda m,a,o:capture.update(info=o)))
    hooks.append(supervision.register_forward_hook(lambda m,a,o:capture.update(kd=o[0],top=o[1]['top_loss'],random=o[1]['random_loss'])))
    def teacher_hook(m,a,o):capture['teacher_calls']=capture.get('teacher_calls',0)+1
    hooks.append(teacher.register_forward_hook(teacher_hook))
    backbone=list(student.backbone.parameters())
    for index,batch in enumerate(loader):
        capture.clear();images=torch.cat(batch[:2]).cuda()
        total,parts=batch_loss(engine,teacher,images,32,criterion,cfg,5 if args.mode=='gradient' else 1)
        main,shadow=capture['student'];assert len(capture['student'])==2 and capture['teacher_calls']==1
        assert main['tensor'].data_ptr()==shadow['tensor'].data_ptr() and torch.equal(main['tensor'],shadow['tensor'])
        assert main['train'] and all(main['bn_modes']) and not shadow['train'] and not any(shadow['bn_modes'])
        assert not shadow['output'].requires_grad and shadow['output'].grad_fn is None
        assert parts['shadow_buffer_immutable'] and parts['shadow_stop_grad']
        bncc,_=geometry_loss(main['output'],shadow['output'],32)
        assert torch.equal(bncc.detach(),parts['loss_bncc'])
        assert all(torch.isfinite(x) for x in [total,capture['info'],capture['top'],capture['random'],bncc])
        row=dict(batch=index,identities=list(batch[3]),loss_total=float(total.detach()),loss_infonce=float(capture['info'].detach()),
            loss_top=float(capture['top'].detach()),loss_random=float(capture['random'].detach()),loss_bncc=float(bncc.detach()),
            same_input_tensor=True,main_batch_stat=True,shadow_running_stat=True,shadow_stop_grad=True,shadow_buffer_immutable=True,
            student_forward_count=2,teacher_forward_count=1)
        if args.mode=='gradient':
            values=[capture['info'],.2*capture['kd'],bncc]
            grads=[torch.autograd.grad(value,backbone,retain_graph=j<2) for j,value in enumerate(values)]
            assert all(torch.isfinite(g).all() for group in grads for g in group)
            info=grad_compare(grads[2],grads[0]);kd=grad_compare(grads[2],grads[1])
            row.update(bncc_to_infonce_ratio=info['ratio'],bncc_to_kd_ratio=kd['ratio'],
                bncc_infonce_cosine=info['cosine'],bncc_kd_cosine=kd['cosine'],
                g_infonce_norm=info['norm_b'],g_kd_norm=kd['norm_b'],g_bncc_norm=info['norm_a'])
            del grads,values
        else:
            engine.backward(total)
            gradients=[safe_get_full_grad(p) for p in model.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            alpha_grad=safe_get_full_grad(supervision.projector_top.alpha)
            assert alpha_grad is not None and torch.isfinite(alpha_grad).all()
            row['alpha_gradient']=float(alpha_grad)
            row['student_gradient_norm']=sum(float(safe_get_full_grad(p).float().square().sum()) for p in student.parameters())**.5
            engine.step();assert_precision(engine)
            assert any(not torch.equal(before[k],v) for k,v in model.named_parameters() if k.startswith('student.'))
        assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        batches.append(row);print('BNCC_'+args.mode.upper()+'_BATCH_PASS='+str(index+1),flush=True)
        del main,shadow,total,bncc
        if index+1==(8 if args.mode=='gradient' else 1):break
    for hook in hooks:hook.remove()
    assert all(file_sha256(cfg[k])==h for k,h in protected.items())
    state=deployment_state_dict(engine);bare=StudentModel(ckpt_path=None);bare.load_state_dict(state,strict=True)
    assert set(state)==set(student.state_dict())
    report=dict(mode=args.mode,batches=batches,protected_sha256=protected,DEPLOYMENT_STRIP_PASS=True,
        TEACHER_FROZEN=True,SAME_INPUT_TENSOR_PASS=True,MAIN_BATCH_STAT_PASS=True,SHADOW_RUNNING_STAT_PASS=True,
        SHADOW_STOP_GRAD_PASS=True,SHADOW_BUFFER_IMMUTABILITY_PASS=True,ALL_LOSSES_FINITE=True,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,lambda_bn=1.,w_ref=1.,
        fresh_pretrained_initialization=True,formal_run_directory_created=False)
    report['gradient_runtime']='Identical Student/head precision and batch_loss, direct autograd.grad without DeepSpeed reduction hooks; real DeepSpeed used for separate smoke/formal training.'
    if args.mode=='gradient':
        assert all(torch.equal(before[k],v) for k,v in model.named_parameters())
        report['statistics']={k:summary([row[k] for row in batches]) for k in ['bncc_to_infonce_ratio','bncc_to_kd_ratio','bncc_infonce_cosine','bncc_kd_cosine','loss_bncc']}
        report['PREFLIGHT_PASS']=report['statistics']['bncc_to_infonce_ratio']['mean']<=1.0
        report['optimizer_steps']=0
    else:report.update(SMOKE_PASS=True,optimizer_steps=1)
    write_json(result_path,report)
    print('REPORT='+str(result_path),flush=True)
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()
