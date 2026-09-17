"""Four independent GPU workers, TRAIN-only; no optimizer or selection."""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT=Path('/home/dingyi/lora-pyra-geo')
BASE=ROOT/'src/checkpoint/student/CERTIFIED_R224'
OUT=BASE/'PARTIII-GROUP0-PREFLIGHT-V1'
OLD=BASE/'PARTIII-SPATIAL-INTERFACE-AUDIT-V1'
SEED=20260917
TEACHER=ROOT/'src/checkpoint/middle_teacher/CERTIFIED_R224/SAM-MABV2-RHO010-S0/best_model.pth'
TSHA='1f5dd3a94e38d5e79bfff05b407959195eb59b9b9359f2380727f6a68fed3d78'
PRETRAIN=ROOT/'src/models/repvit/repvit_m1_5_distill_450e.pth'
PSHA='d645a2de5481c9aac1639d0e97b04cd4bdb0df9d7347920b132dd0ed45de8b39'
P2=BASE/'P2-TOP-RMLP-S0/best_model.pth'
P2SHA='ae59d44befd2efb8b9353f6a58b79f1c68ea294fe3ee5be8772ef594e9f58ec8'
TRAIN=ROOT/'data/U1652/train'
CONFIG=ROOT/'configs/student/certified_r224/p2_top_rmlp_s0.json'
SHIFTS=[(16,0),(-16,0),(0,16),(0,-16)]


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(name,obj):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)


def read(name):return json.loads((OUT/name).read_text())


def csv_write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(values):
    import numpy as np
    a=np.asarray(values,dtype='float64')
    assert a.size and np.isfinite(a).all()
    return dict(n=a.size,mean=float(a.mean()),sample_std=float(a.std(ddof=1)) if a.size>1 else 0.,
        median=float(np.median(a)),p10=float(np.percentile(a,10)),p25=float(np.percentile(a,25)),
        p75=float(np.percentile(a,75)),p90=float(np.percentile(a,90)),negative_fraction=float((a<0).mean()))


def cosine(a,b):
    import numpy as np
    a=a.astype('float64',copy=False);b=b.astype('float64',copy=False)
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    if not denom>0:raise ValueError('Undefined zero gradient cosine')
    return float(np.dot(a,b)/denom)


def setup(gpu):
    assert gpu in range(4)
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='2'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    import torch
    assert torch.cuda.device_count()==1 and not torch.distributed.is_initialized()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def manifest():
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(OUT)
    assert sha(TEACHER)==TSHA and sha(PRETRAIN)==PSHA and sha(P2)==P2SHA
    OUT.mkdir(parents=True)
    from src.student.data import U1652PairDataset
    from src.dataset.teacher.datasets import CrossViewPairSampler
    data=U1652PairDataset(str(TRAIN))
    sampler=CrossViewPairSampler(data,32,seed=SEED);sampler.set_epoch(0)
    batches=[]
    for i,indices in enumerate(sampler):
        if i==16:break
        rows=[]
        for index in indices:
            identity,label,satellite,drone=data.pairs[index]
            assert Path(drone).resolve().is_relative_to(TRAIN.resolve()) and Path(satellite).resolve().is_relative_to(TRAIN.resolve())
            rows.append(dict(index=index,identity=identity,drone=drone,satellite=satellite,
                drone_sha256=sha(drone),satellite_sha256=sha(satellite)))
        assert len(rows)==32 and len({r['identity'] for r in rows})==32
        batches.append(dict(batch=i,augmentation_seed=SEED+i*1009,rows=rows))
    old=json.loads((OLD/'spatial_audit_manifest.json').read_text())
    assert old['images']==512 and old['identities']==256
    for row in old['records']:
        assert Path(row['image_path']).resolve().is_relative_to(TRAIN.resolve())
        assert sha(row['image_path'])==row['image_sha256']
    write('preflight_manifest.json',dict(seed=SEED,split='train',batch_count=16,pairs_per_batch=32,
        batches=batches,diagnostic_manifest_source=str(OLD/'spatial_audit_manifest.json'),
        diagnostic_manifest_sha256=sha(OLD/'spatial_audit_manifest.json'),diagnostic_records=old['records']))
    cfg=json.loads(CONFIG.read_text())
    assets={k:dict(path=cfg[k],sha256=sha(cfg[k])) for k in ['stst_asset','original_stst_asset','p2_calibration_path']}
    assert assets['stst_asset']['sha256']==cfg['extended_stst_asset_sha256']
    assert assets['original_stst_asset']['sha256']==cfg['original_stst_asset_sha256']
    assert assets['p2_calibration_path']['sha256']==cfg['p2_calibration_sha256']
    write('preflight_config.json',dict(task=OUT.name,teacher=str(TEACHER),teacher_sha256=TSHA,
        student_pretrained=str(PRETRAIN),student_pretrained_sha256=PSHA,p2_diagnostic=str(P2),p2_sha256=P2SHA,
        assets=assets,protocol='STU-1G-B32-R224-v1',gradient_graph='TRAIN_INIT_GRAPH',diagnostic_graph='P2_BEST_DIAGNOSTIC',
        initialization_seed=0,manifest_seed=SEED,training_steps=0,optimizer_steps=0,checkpoint_selection=False,
        lambda_parameter_set='ALL_BACKBONE',stage3_and_downstream='backbone.features indices 12 through 42, excluding neck and heads',
        parent_objective='InfoNCE + 0.2*(Top128_RMLP+Random32_Linear); full-strength w=1 diagnostic only; future w(e)=min(e/5,1)',
        point_projector='fresh Conv2d(256,768,1,bias=True); FP32 parameters/compute; seed20260917 isolated from parent RNG',
        step0_reset='Restore complete initial model/head/buffer state before each independent batch; no parameter update',
        gradient_precision='BF16 Student parameters/forward as current training; raw parameter gradients cast FP32; CPU float64 reductions',
        teacher_target='backbone.model.norm full sequence; remove 1 CLS + 4 storage tokens; row-major14x14',
        lambda_formula='min(median||g_kd||,0.25*median||g_ret||)/median||g_spatial||; 3 significant digits only within [0.01,1]',
        source_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        spatial_flags_default=False,formal_training_started=False))


def tensor_sha(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_hash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(tensor_sha(v).encode())
    return h.hexdigest()


def teacher_load():
    from src.evaluation.model_loader import load_encoder
    enc,audit=load_encoder('middle',TEACHER,TEACHER.parent/'run_config.json')
    assert audit['sha256']==TSHA
    core=enc.model.backbone.model
    assert core.n_storage_tokens==4 and core.embed_dim==768 and len(core.blocks)==12
    assert tuple(core.patch_embed.patch_size)==(16,16) and not core.untie_cls_and_patch_norms
    return enc.model


def capture_teacher(teacher,cache):
    from src.student.spatial_kd import spatial_tokens
    def capture(m,args,out):
        assert out.shape[1:]==(201,768)
        cache['teacher']=spatial_tokens(out,prefix_count=5,grid=(14,14))
    return teacher.backbone.model.norm.register_forward_hook(capture)


def training_input(batch):
    import numpy as np
    import torch
    from src.student.data import U1652PairDataset
    from src.dataset.transforms import get_train_transforms
    seed=batch['augmentation_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    _,sat,drone=get_train_transforms([224,224])
    # Both transforms share nested objects in the production constructor. Seed
    # them in the same order as _seed_stst_worker; execute dataset.__getitem__ unchanged.
    sat.set_random_seed(seed+17);drone.set_random_seed(seed+31)
    data=U1652PairDataset(str(TRAIN),sat_transforms=sat,drone_transforms=drone,prob_flip=.5,shuffle_batch_size=32)
    samples=[]
    for row in batch['rows']:
        p=data.pairs[row['index']]
        assert p[0]==row['identity'] and p[2]==row['satellite'] and p[3]==row['drone']
        samples.append(data[row['index']])
    x=torch.cat((torch.stack([r[0] for r in samples]),torch.stack([r[1] for r in samples])))
    ids=[r['drone'] for r in batch['rows']]+[r['satellite'] for r in batch['rows']]
    return x.cuda(),ids,tensor_sha(x)


def gradient_worker(gpu,pipe):
    import numpy as np
    import torch
    from src.student.model import StudentModel
    from src.student.part1 import PartISupervision
    from src.student.part2_integration import prepare_top
    from src.student.objective import PairInfoNCE
    from src.student.runtime import _seed_all
    from src.student.spatial_group0 import Stage3PointwiseSpatialKD,CenteredSpatialRelationKD
    from src.models.repvit_backbone import RepViTBackbone
    cfg=json.loads(CONFIG.read_text());_seed_all(0)
    student=StudentModel(ckpt_path=str(PRETRAIN)).cuda()
    raw=RepViTBackbone._unwrap_state_dict(RepViTBackbone._safe_torch_load(PRETRAIN))
    features={RepViTBackbone._normalize_key(k):v for k,v in raw.items() if RepViTBackbone._normalize_key(k).startswith('features.')}
    assert len(features)==1131 and set(features)==set(student.backbone.state_dict())
    assert all(torch.equal(v,student.backbone.state_dict()[k].cpu()) for k,v in features.items())
    del raw,features
    supervision=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],TSHA,128,'single32').cuda()
    teacher=teacher_load()
    prepare_top(supervision,cfg);student.bfloat16();supervision.bfloat16()
    assert supervision.projector_top.residual.hidden_dim==920
    assert supervision.projector_top.alpha.dtype==torch.float32 and supervision.projector_random.linear.weight.dtype==torch.bfloat16
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(SEED)
        spatial=(Stage3PointwiseSpatialKD() if gpu==0 else CenteredSpatialRelationKD()).cuda()
    initial={k:v.detach().clone() for k,v in student.state_dict().items()}
    head_state={k:v.detach().clone() for k,v in supervision.state_dict().items()}
    initial_hash=state_hash(student);heads_hash=state_hash(supervision);teacher_hash=state_hash(teacher)
    spatial_hash=state_hash(spatial)
    names,parameters=zip(*student.backbone.named_parameters())
    offset=0;ranges=[]
    for name,p in zip(names,parameters):
        if name.startswith('features.') and int(name.split('.')[1])>=12:ranges.append((offset,offset+p.numel()))
        offset+=p.numel()
    selections={'ALL_BACKBONE':np.arange(offset),'STAGE3_AND_DOWNSTREAM':np.concatenate([np.arange(a,b) for a,b in ranges])}
    layout=[dict(name=n,shape=list(p.shape),numel=p.numel()) for n,p in zip(names,parameters)]
    cache={};ht=capture_teacher(teacher,cache)
    hs=student.backbone.features[37].register_forward_hook(lambda m,args,out:cache.update(stage3=out))
    rows=[];inputs=[];bn_checks=[];head_grad_check=None
    for batch in read('preflight_manifest.json')['batches']:
        student.load_state_dict(initial,strict=True);supervision.load_state_dict(head_state,strict=True)
        student.train();supervision.train();student.zero_grad(set_to_none=True);supervision.zero_grad(set_to_none=True)
        x,ids,input_sha=training_input(batch)
        inputs.append(dict(batch=batch['batch'],input_sha256=input_sha,student_input_sha256=tensor_sha(x.bfloat16()),teacher_input_sha256=tensor_sha(x.bfloat16())))
        with torch.no_grad():td=teacher(x.bfloat16())
        shapes={};handles=[]
        def bn_hook(name):
            def f(m,args):shapes.setdefault(name,[]).append(len(args[0]))
            return f
        for name,m in student.named_modules():
            if isinstance(m,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d)):handles.append(m.register_forward_pre_hook(bn_hook(name)))
        descriptor=student(x.bfloat16())
        for h in handles:h.remove()
        assert len(shapes)==171 and all(v==[64] for v in shapes.values())
        bn_checks.append(True)
        s=cache['stage3'];t=cache['teacher'];assert s.shape==(64,256,14,14) and t.shape==(64,196,768)
        ret=PairInfoNCE(.1)(descriptor[:32],descriptor[32:],student.logit_scale.exp())
        dual,kda=supervision(descriptor,td,32);kd=.2*dual;base=ret+kd
        assert torch.allclose(dual,kda['top_loss']+kda['random_loss'])
        losses=spatial(s[:32],t[:32],s[32:],t[32:],drone_image_ids=ids[:32],teacher_drone_image_ids=ids[:32],
            satellite_image_ids=ids[32:],teacher_satellite_image_ids=ids[32:])
        sources=dict(ret=ret,kd=kd,base=base,spatial=losses['loss'],drone=losses['drone_loss'],satellite=losses['satellite_loss'])
        gradients={}
        for key,loss in sources.items():
            assert torch.isfinite(loss)
            gs=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            vector=torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1).cpu() for p,g in zip(parameters,gs)]).numpy()
            assert np.isfinite(vector).all()
            gradients[key]=vector
            del gs
        if gpu==0 and batch['batch']==0:
            gs=torch.autograd.grad(losses['loss'],tuple(spatial.parameters()),retain_graph=True)
            assert all(torch.isfinite(g).all() for g in gs) and sum(float(g.norm()) for g in gs)>0
            head_grad_check=dict(finite=True,nonzero=True,norms=[float(g.norm()) for g in gs]);del gs
        for set_name,index in selections.items():
            g={k:v[index] for k,v in gradients.items()}
            norms={k:float(np.linalg.norm(v.astype('float64'))) for k,v in g.items()}
            assert min(norms.values())>0
            rows.append(dict(batch=batch['batch'],parameter_set=set_name,**{k+'_norm':v for k,v in norms.items()},
                spatial_base_cos=cosine(g['spatial'],g['base']),spatial_ret_cos=cosine(g['spatial'],g['ret']),
                spatial_kd_cos=cosine(g['spatial'],g['kd']),drone_sat_cos=cosine(g['drone'],g['satellite']),
                drone_sat_norm_ratio=norms['drone']/norms['satellite'],
                **{k+'_loss':float(v.detach()) for k,v in sources.items()}))
        # Full gradients travel only through CPU IPC; never saved to any file.
        pipe.send(dict(kind='gradient',batch=batch['batch'],vector=gradients['spatial'],ranges=ranges,
                       layout=layout,input_sha=input_sha,initial_hash=initial_hash,heads_hash=heads_hash))
        assert all(p.grad is None for p in teacher.parameters())
        print(f'GPU{gpu} BATCH={batch["batch"]+1}/16 FINITE=True',flush=True)
        del descriptor,td,s,t,ret,dual,kda,kd,base,losses,sources,gradients,g,x
        cache.clear()
    student.load_state_dict(initial);supervision.load_state_dict(head_state)
    assert state_hash(student)==initial_hash and state_hash(supervision)==heads_hash and state_hash(teacher)==teacher_hash
    assert state_hash(spatial)==spatial_hash
    ht.remove();hs.remove()
    summary={key:{metric:stats([r[metric] for r in rows if r['parameter_set']==key]) for metric in rows[0] if metric not in ('batch','parameter_set')} for key in selections}
    a=summary['ALL_BACKBONE'];target=min(a['kd_norm']['median'],.25*a['ret_norm']['median'])
    raw=target/a['spatial_norm']['median'];warning=not (.01<=raw<=1.)
    lam=None if warning else float(format(raw,'.3g'))
    report=dict(pass_status=True,graph='TRAIN_INIT_GRAPH',batches=16,summary=summary,
        lambda_raw=raw,lambda_value=lam,scale_warning=warning,target_gradient=target,lambda_basis='ALL_BACKBONE',
        expected_raw_weighted_ret_ratio=raw*a['spatial_norm']['median']/a['ret_norm']['median'],
        expected_raw_weighted_kd_ratio=raw*a['spatial_norm']['median']/a['kd_norm']['median'],
        rounded_weighted_ret_ratio=None if lam is None else lam*a['spatial_norm']['median']/a['ret_norm']['median'],
        rounded_weighted_kd_ratio=None if lam is None else lam*a['spatial_norm']['median']/a['kd_norm']['median'],
        input_checksums=inputs,student_initial_hash=initial_hash,heads_initial_hash=heads_hash,state_restored=True,
        teacher_frozen=True,pretrained_load=dict(matched=1131,total=1131,missing=0,unexpected=0),
        student_forward_N64_all_batches=all(bn_checks),backbone_parameter_layout=layout,stage3_ranges=ranges,
        spatial_head_gradient_check=head_grad_check,no_cross_view_position_matching=True,optimizer_steps=0,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),teacher_sha256=sha(TEACHER),pretrain_sha256=sha(PRETRAIN))
    name='pointwise' if gpu==0 else 'relational'
    write(f'gpu{gpu}_{name}_gradient_audit.json',report);csv_write(f'gpu{gpu}_{name}_per_batch.csv',rows)


def diagnostic_models():
    from src.evaluation.model_loader import load_encoder
    teacher=teacher_load();student,sa=load_encoder('student',P2)
    assert sa['sha256']==P2SHA
    return teacher,student.model


def diagnostic_inputs(rows):
    import numpy as np
    import torch
    from PIL import Image
    from src.dataset.teacher.transforms import get_paired_cross_view_val_transforms
    tf=get_paired_cross_view_val_transforms([224,224]);images=[]
    for r in rows:
        p=Path(r['image_path']);assert p.resolve().is_relative_to(TRAIN.resolve()) and sha(p)==r['image_sha256']
        images.append(tf(image=np.array(Image.open(p).convert('RGB')))['image'])
    return torch.stack(images).cuda()


def stage4_worker():
    import torch
    from src.student.spatial_group0 import Stage4TeacherPooling
    from tools.audit.partiii_spatial_interface_audit import inventory,image_stats
    teacher,student=diagnostic_models();before=[state_hash(teacher),state_hash(student)]
    records=read('preflight_manifest.json')['diagnostic_records']
    x=diagnostic_inputs(records[:2]);ti,si,exact=inventory(teacher,student,x)
    stage=si['stages'][-1]
    assert [stage['H'],stage['W']]==[7,7] and stage['C']==512
    cache={};ht=capture_teacher(teacher,cache)
    hs=student.get_submodule(stage['module_name']).register_forward_hook(lambda m,a,o:cache.update(student=o.detach()))
    pool=Stage4TeacherPooling();rows=[]
    for start in range(0,len(records),8):
        batch=records[start:start+8];x=diagnostic_inputs(batch)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):teacher(x);student(x)
        raw=cache['teacher'].float()
        pooled=pool(raw,normalize=False)
        manual=raw.reshape(-1,7,2,7,2,768).mean((2,4)).reshape(-1,49,768)
        torch.testing.assert_close(pooled,manual,atol=2e-6,rtol=2e-6)
        sf=cache['student'].flatten(2).transpose(1,2)
        assert sf.shape[:2]==pooled.shape[:2]
        for r,t,s in zip(batch,pooled,sf):rows.append(dict(identity=r['identity'],view=r['view'],image_path=r['image_path'],**image_stats(t,s,(7,7))))
    ht.remove();hs.remove();after=[state_hash(teacher),state_hash(student)];assert before==after
    metrics=[k for k in rows[0] if k not in ('identity','view','image_path')]
    summary={view:{k:stats([r[k] for r in rows if r['view']==view]) for k in metrics} for view in ('drone','satellite')}
    no_exact_collapse=all(r['teacher_spatial_token_variance']>0 and r['student_spatial_token_variance']>0
                          and r['teacher_effective_rank']>1 and r['student_effective_rank']>1 for r in rows)
    write('gpu2_stage4_interface.json',dict(pass_status=True,graph='P2_BEST_DIAGNOSTIC',stage=stage,
        teacher_inventory=ti,student_inventory=si,teacher_7x7_mapping_pass=True,
        mapping='raw final-norm14x14 -> FP32 mean2x2 stride2 -> per-token FP32 L2',
        structural_interface_pass=bool(no_exact_collapse and ti['grid_order_pass'] and si['grid_order_pass']),
        collapse_definition='Exact degeneracy only: nonzero spatial variance and effective rank>1 for every image; no correlation threshold',
        geometry_signal=summary,images=len(rows),identities=len(set(r['identity'] for r in rows)),
        state_before=before,state_after=after,features_finite=True,formal_candidate_decision=None))
    csv_write('gpu2_stage4_per_image.csv',rows)


def shift_worker():
    import numpy as np
    import torch
    from torch.nn import functional as F
    from src.student.spatial_group0 import PatchAlignedShiftMapper,TeacherStableWeight
    teacher,student=diagnostic_models();before=[state_hash(teacher),state_hash(student)]
    cache={};ht=capture_teacher(teacher,cache)
    hs=student.backbone.features[37].register_forward_hook(lambda m,a,o:cache.update(student=o))
    records=read('preflight_manifest.json')['diagnostic_records']
    stable=TeacherStableWeight();rows=[];pooled={v:{str(shift):[] for shift in SHIFTS} for v in ('drone','satellite')}
    mappings={}
    ids=torch.arange(196).reshape(1,1,14,14).repeat_interleave(16,2).repeat_interleave(16,3).float()
    for dx,dy in SHIFTS:
        mapper=PatchAlignedShiftMapper(dx,dy);a,b=mapper.indices()
        shifted=mapper.translate(ids,padding=-1)
        token_ids=F.avg_pool2d(shifted,16,16).flatten()
        assert torch.equal(token_ids[b],torch.arange(196).float()[a]) and len(a.unique())==len(a)==len(b.unique())
        assert (token_ids[b]>=0).all()
        mappings[str((dx,dy))]=dict(valid_tokens=len(a),original_indices=a.tolist(),shifted_indices=b.tolist(),pass_status=True)
    for start in range(0,len(records),8):
        batch=records[start:start+8];x=diagnostic_inputs(batch)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):teacher(x)
        original=cache['teacher'].clone()
        for dx,dy in SHIFTS:
            mapper=PatchAlignedShiftMapper(dx,dy)
            # Padding normalized value 0 = ImageNet mean RGB. Supervision excludes it.
            shifted=mapper.translate(x,padding=0.)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):teacher(shifted)
            a,b=mapper.align(original,cache['teacher']);confidence,weights=stable(a,b)
            torch.testing.assert_close(weights.mean(1),torch.ones(len(batch),device='cuda'),atol=1e-6,rtol=1e-6)
            for r,c in zip(batch,confidence.cpu().numpy()):
                summary=stats(c)
                rows.append(dict(identity=r['identity'],view=r['view'],image_path=r['image_path'],dx=dx,dy=dy,
                                 valid_tokens=len(c),**{k:v for k,v in summary.items() if k not in ('n','negative_fraction')}))
                pooled[r['view']][str((dx,dy))].extend(c.tolist())
        if start%64==0:print(f'GPU3 SHIFT_IMAGES={min(start+8,len(records))}/{len(records)}',flush=True)
    # One backward-only Student smoke on two TRAIN images, kept in eval mode.
    x=diagnostic_inputs(records[:2]);mapper=PatchAlignedShiftMapper(16,0);shifted=mapper.translate(x)
    for p in student.backbone.parameters():p.requires_grad_(True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        teacher(x);t0=cache['teacher'].clone();teacher(shifted);t1=cache['teacher'].clone()
    ta,tb=mapper.align(t0,t1);c,w=stable(ta,tb)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        student(x);s0=cache['student'];student(shifted);s1=cache['student']
    a,b=mapper.align(s0.flatten(2).transpose(1,2),s1.flatten(2).transpose(1,2))
    loss=stable.weighted_loss(1-F.cosine_similarity(a.float(),b.float(),dim=-1),w)
    assert torch.isfinite(loss);loss.backward()
    grads=[p.grad for p in student.backbone.features[37].parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.float().norm()) for g in grads)>0
    assert all(p.grad is None for p in teacher.parameters())
    student.zero_grad(set_to_none=True);ht.remove();hs.remove()
    after=[state_hash(teacher),state_hash(student)];assert before==after
    write('gpu3_shift_mapping.json',dict(pass_status=True,graph='P2_BEST_DIAGNOSTIC',mapping=mappings,
        padding='constant0 in normalized tensor = ImageNet mean RGB; no wrap',coordinate_convention='dx right, dy down',
        mapping_unit_test_pass=True,overlap_excludes_padding=True,student_backward_smoke_pass=True,
        student_overlap_shape=list(a.shape),student_smoke_loss=float(loss.detach()),state_before=before,state_after=after,
        optimizer_steps=0,stable_weights_mean_one=True,no_cross_view_position_matching=True))
    summary={v:{k:stats(c) for k,c in shifts.items()} for v,shifts in pooled.items()}
    overall={v:stats([c for values in shifts.values() for c in values]) for v,shifts in pooled.items()}
    write('gpu3_teacher_stability.json',dict(pass_status=True,views_by_shift=summary,pooled_by_view=overall,
        aggregation='Equal valid-token weighting, all images and four cardinal shifts; std ddof=1',
        images=len(records),stability='(1+same-image aligned Teacher cosine)/2',hard_threshold=None))
    csv_write('gpu3_teacher_stability_per_image.csv',rows)


def entry(gpu,pipe):
    try:
        with (OUT/f'gpu{gpu}.log').open('x',buffering=1) as log:
            sys.stdout=log;sys.stderr=log;setup(gpu)
            if gpu in (0,1):gradient_worker(gpu,pipe)
            elif gpu==2:stage4_worker()
            else:shift_worker()
            pipe.send(dict(kind='done',gpu=gpu))
    except BaseException:
        error=traceback.format_exc()
        try:write(f'gpu{gpu}_failure.json',dict(error=error,pass_status=False));pipe.send(dict(kind='error',gpu=gpu,error=error))
        except Exception:pass
    finally:pipe.close()


def launch():
    import numpy as np
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
    ctx=mp.get_context('spawn');active={};processes=[];pending={};matches=[];done=[];busy=[]
    for gpu in range(4):
        if memory[gpu]>=100:busy.append(gpu);write(f'gpu{gpu}_failure.json',dict(pass_status=False,status='GPU_BUSY'));continue
        parent,child=ctx.Pipe(duplex=False);p=ctx.Process(target=entry,args=(gpu,child));p.start();child.close()
        active[parent]=gpu;processes.append(p)
    while active:
        for pipe in wait(list(active),timeout=5):
            gpu=active[pipe]
            try:message=pipe.recv()
            except EOFError:active.pop(pipe);pipe.close();continue
            if message['kind']=='gradient':
                key=message['batch'];pending.setdefault(key,{})[gpu]=message
                if set(pending[key])=={0,1}:
                    pair=pending.pop(key);a,b=pair[0],pair[1]
                    assert a['input_sha']==b['input_sha'] and a['initial_hash']==b['initial_hash'] and a['heads_hash']==b['heads_hash']
                    assert a['layout']==b['layout'] and a['ranges']==b['ranges']
                    full=cosine(a['vector'],b['vector'])
                    index=np.concatenate([np.arange(x,y) for x,y in a['ranges']])
                    stage=cosine(a['vector'][index],b['vector'][index])
                    matches.append(dict(batch=key,ALL_BACKBONE=full,STAGE3_AND_DOWNSTREAM=stage,input_match=True))
            elif message['kind']=='done':done.append(gpu)
            else:print(json.dumps(message),flush=True)
    for p in processes:p.join()
    if len(matches)==16:
        write('point_rel_matched_gradient_summary.json',dict(pass_status=True,point_rel_input_match_pass=True,
            per_batch=sorted(matches,key=lambda x:x['batch']),summary={k:stats([r[k] for r in matches]) for k in ('ALL_BACKBONE','STAGE3_AND_DOWNSTREAM')},
            merge='CPU float64 dot products of full matched gradients via memory-only IPC; no gradient/feature cache saved'))
    write('worker_status.json',dict(done=sorted(done),busy=busy,all_workers_pass=sorted(done)==[0,1,2,3] and len(matches)==16))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch']);a=p.parse_args()
    if a.mode=='prepare':manifest()
    else:launch()
