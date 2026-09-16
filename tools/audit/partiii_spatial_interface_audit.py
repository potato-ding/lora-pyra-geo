"""Frozen TRAIN-only spatial inventory/geometry; never a training entry point."""
from __future__ import annotations
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess

ROOT = Path('/home/dingyi/lora-pyra-geo')
STUDENT_ROOT = ROOT/'src/checkpoint/student/CERTIFIED_R224'
OUT = STUDENT_ROOT/'PARTIII-SPATIAL-INTERFACE-AUDIT-V1'
TEACHER = ROOT/'src/checkpoint/middle_teacher/CERTIFIED_R224/SAM-MABV2-RHO010-S0/best_model.pth'
STUDENT = STUDENT_ROOT/'P2-TOP-RMLP-S0/best_model.pth'
TEACHER_SHA = '1f5dd3a94e38d5e79bfff05b407959195eb59b9b9359f2380727f6a68fed3d78'
STUDENT_SHA = 'ae59d44befd2efb8b9353f6a58b79f1c68ea294fe3ee5be8772ef594e9f58ec8'
TRAIN = ROOT/'data/U1652/train'
SEED = 20260916


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def read(name): return json.loads((OUT/name).read_text())


def head(): return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def prepare():
    if OUT.exists() and any(OUT.iterdir()): raise RuntimeError('Refusing nonempty audit output')
    assert sha(TEACHER) == TEACHER_SHA and sha(STUDENT) == STUDENT_SHA
    ids = sorted(p.name for p in (TRAIN/'drone').iterdir() if p.is_dir() and (TRAIN/'satellite'/p.name).is_dir())
    rng = random.Random(SEED)
    selected = rng.sample(ids, 256)
    records = []
    for i, identity in enumerate(selected):
        for view in ['drone', 'satellite']:
            images = sorted(p for p in (TRAIN/view/identity).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
            p = rng.choice(images).resolve()
            assert p.is_relative_to(TRAIN.resolve())
            records.append(dict(identity=identity, view=view, image_path=str(p), worker_shard=i//128,
                                identity_index=i, image_sha256=sha(p)))
    write(OUT/'spatial_audit_manifest.json', dict(seed=SEED, split='train', identities=256,
          images=512, selection='random.Random(seed): sample sorted identities; choice sorted images per view', records=records))
    write(OUT/'audit_config.json', dict(task=OUT.name, teacher=str(TEACHER), teacher_sha256=TEACHER_SHA,
          student=str(STUDENT), student_sha256=STUDENT_SHA, train_root=str(TRAIN.resolve()), input_size=224,
          seed=SEED, identities=256, images=512, worker_gpus=[2,3], batch_size=4,
          teacher_parameters='BF16', student_parameters='FP32', forward='CUDA BF16 autocast',
          normalized_tokens_and_gram='FP32, TF32 disabled', statistics='CPU float64 reductions/ranks/eigensolver',
          spatial_kd_enabled_in_existing_training=False, source_parent_commit=head(),
          geometric_transform='existing deterministic val transform applied to TRAIN images only',
          normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]),
          formal_training=False, optimizer_steps=0, cross_view_position_matching=False))
    print('MANIFEST_READY=True', flush=True)


def runtime(gpu):
    if gpu not in (2,3) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise RuntimeError('Only an explicitly isolated GPU2/GPU3 process is allowed')
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.device_count() == 1


def models():
    from src.evaluation.model_loader import load_encoder
    teacher, ta = load_encoder('middle', TEACHER, TEACHER.parent/'run_config.json')
    student, sa = load_encoder('student', STUDENT)
    assert ta['sha256'] == TEACHER_SHA and sa['sha256'] == STUDENT_SHA
    assert not ta['missing'] and not ta['unexpected'] and not sa['missing'] and not sa['unexpected']
    return teacher.model, student.model, ta, sa


def tensor_sha(tensor):
    import torch
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor_sha(tensor).encode())
    return h.hexdigest()


def inputs(records):
    import numpy as np
    import torch
    from PIL import Image
    from src.dataset.teacher.transforms import get_paired_cross_view_val_transforms
    transform = get_paired_cross_view_val_transforms([224,224])
    xs, checks = [], []
    for row in records:
        p = Path(row['image_path'])
        assert p.resolve().is_relative_to(TRAIN.resolve()) and sha(p) == row['image_sha256']
        raw = np.array(Image.open(p).convert('RGB'))
        geometric = transform.transforms[0](image=raw)['image']
        # Compare independent applications, then share the exact input tensor.
        geometric_student = transform.transforms[0](image=raw.copy())['image']
        assert np.array_equal(geometric, geometric_student)
        x = transform(image=raw)['image']
        xs.append(x)
        checks.append(dict(image_path=str(p), pre_normalization_dtype=str(geometric.dtype),
            shape=list(geometric.shape), teacher_sha256=hashlib.sha256(geometric.tobytes()).hexdigest(),
            student_sha256=hashlib.sha256(geometric_student.tobytes()).hexdigest(),
            normalized_input_sha256=tensor_sha(x)))
    return torch.stack(xs).cuda(), checks


def describe(x): return dict(shape=list(x.shape), dtype=str(x.dtype))


def inventory(teacher, student, x):
    import torch
    import torch.nn.functional as F
    core = teacher.backbone.model
    shapes, values, handles = {}, {}, []
    def capture(name, retain=False):
        def fn(module, args, out):
            shapes[name] = dict(input=describe(args[0]), output=describe(out))
            if retain: values[name] = out.detach()
        return fn
    for name, module in [('patch_embed.proj',core.patch_embed.proj),('patch_embed',core.patch_embed),('norm',core.norm)]:
        handles.append(module.register_forward_hook(capture(name, True)))
    for i, module in enumerate(core.blocks): handles.append(module.register_forward_hook(capture('blocks.'+str(i))))
    def block_input(module, args): values['prepared_tokens'] = args[0].detach()
    handles.append(core.blocks[0].register_forward_pre_hook(block_input))
    for i, module in enumerate(student.backbone.features):
        handles.append(module.register_forward_hook(capture('student.backbone.features.'+str(i), i in student.backbone.out_indices)))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        td = teacher(x)
        sd = student(x)
    for handle in handles: handle.remove()
    grid = list(values['patch_embed'].shape[1:3])
    prefix = int(core.cls_token.shape[1] + core.storage_tokens.shape[1])
    assert prefix == core.n_storage_tokens+1
    assert not core.untie_cls_and_patch_norms, 'This audit requires observed shared full-sequence norm'
    patch = values['patch_embed'].flatten(1,2)
    with torch.no_grad(): expected = core.patch_embed.norm(values['patch_embed.proj'].flatten(2).transpose(1,2))
    teacher_order = torch.equal(patch, expected) and torch.equal(values['prepared_tokens'][:,prefix:], patch)
    assert values['norm'].shape[1] == prefix+grid[0]*grid[1]
    assert torch.equal(td,F.normalize(values['norm'][:,0].float(),dim=-1,eps=1e-6))
    stages, all_features, transitions = [], [], []
    previous = None
    for i in range(len(student.backbone.features)):
        name = 'student.backbone.features.'+str(i)
        shape = shapes[name]['output']['shape']
        record = dict(module_name='backbone.features.'+str(i), shape=shape, dtype=shapes[name]['output']['dtype'],
                      C=shape[1], H=shape[2], W=shape[3], stride=[224/shape[2],224/shape[3]])
        all_features.append(record)
        if shape[2:] != previous: transitions.append(record)
        previous = shape[2:]
        if i in student.backbone.out_indices: stages.append(record)
    matches = [r for r in stages if [r['H'],r['W']]==grid]
    all_matches = [r['module_name'] for r in all_features if [r['H'],r['W']]==grid]
    # The backbone's declared output boundaries, not an arbitrary internal block.
    primary = matches[-1] if matches else None
    student_order = True
    if primary:
        feature = values['student.'+primary['module_name']]
        flat = feature.flatten(2).transpose(1,2)
        student_order = all(torch.equal(flat[:,h*grid[1]+w],feature[:,:,h,w]) for h in range(grid[0]) for w in range(grid[1]))
    ti = dict(architecture=teacher.backbone_name, teacher_patch_size=list(core.patch_embed.patch_size),
         transformer_block_count=len(core.blocks), teacher_hidden_dim=int(core.embed_dim),
         teacher_prefix_token_count=prefix, cls_token_count=int(core.cls_token.shape[1]),
         register_storage_token_count=int(core.storage_tokens.shape[1]), extra_prefix_count=prefix-1-core.n_storage_tokens,
         teacher_spatial_token_count=grid[0]*grid[1], teacher_spatial_grid=grid,
         teacher_final_spatial_source_module='backbone.model.norm', full_sequence_final_norm=True,
         cls_extraction='forward_frozen_prefix: model.norm(tokens)[:,0] -> FP32 L2',
         hooks={k:v for k,v in shapes.items() if not k.startswith('student.')},
         ordering='patch Conv NCHW -> flatten(2).transpose(1,2) -> BHWC -> flatten(1,2); CLS/storage prepended only',
         patch_embed_source=inspect.getsource(type(core.patch_embed).forward),
         token_preparation_source=inspect.getsource(core.prepare_tokens_with_masks),
         grid_order_pass=teacher_order, normalized_cls_matches_descriptor=True,
         final_descriptor=describe(td))
    si = dict(architecture=student.BACKBONE_NAME, stages=stages, resolution_transitions=transitions,
              all_feature_blocks=all_features, grid_order_pass=student_order,
              ordering='NCHW flatten(2).transpose(1,2): index=h*W+w', final_descriptor=describe(sd),
              forward_source=inspect.getsource(type(student.backbone).forward))
    exact = dict(teacher_grid=grid, exact_grid_matches=matches, all_exact_grid_block_modules=all_matches,
                 primary=primary, selection='declared backbone stage output boundary; no fusion or interpolation',
                 exact_grid_match_found=bool(matches), teacher_grid_is_14x14=grid==[14,14])
    return ti, si, exact


def corr(a,b):
    import numpy as np
    a = np.asarray(a,dtype=np.float64); b = np.asarray(b,dtype=np.float64)
    a = a-a.mean(); b = b-b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    if denom == 0: raise ValueError('Undefined correlation: collapsed spatial geometry')
    return float(a@b/denom)


def image_stats(t,s,grid):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata
    t=t.float().cpu(); s=s.float().cpu()
    assert torch.isfinite(t).all() and torch.isfinite(s).all()
    tn=F.normalize(t,dim=-1); sn=F.normalize(s,dim=-1)
    gt=(tn@tn.T).numpy(); gs=(sn@sn.T).numpy()
    off=~np.eye(len(gt),dtype=bool)
    a,b=gt[off],gs[off]
    metrics=dict(pearson=corr(a,b), spearman=corr(rankdata(a),rankdata(b)),
                 centered_cosine=corr(a,b), mse=float(np.mean((a.astype('float64')-b)**2)))
    yy,xx=np.mgrid[:grid[0],:grid[1]]
    coords=np.stack([yy.ravel(),xx.ravel()],1)
    distance=np.abs(coords[:,None]-coords[None,:]).sum(-1)
    local=distance==1; far=distance>=max(2,(max(grid)+1)//2)
    tl=(gt*local).sum(1)/local.sum(1); sl=(gs*local).sum(1)/local.sum(1)
    tf=(gt*far).sum(1)/far.sum(1); sf=(gs*far).sum(1)/far.sum(1)
    metrics.update(teacher_local_mean=float(tl.mean()),student_local_mean=float(sl.mean()),
        teacher_far_mean=float(tf.mean()),student_far_mean=float(sf.mean()),
        teacher_local_minus_far=float((tl-tf).mean()),student_local_minus_far=float((sl-sf).mean()),
        local_geometry_correlation=corr(tl,sl))
    for name,raw,norm,gram in [('teacher',t,tn,gt),('student',s,sn,gs)]:
        vals=np.maximum(np.linalg.eigvalsh(gram.astype('float64')),0.)
        singular=np.sqrt(vals); prob=singular/singular.sum(); positive=prob[prob>0]
        feature_norm=raw.norm(dim=-1).numpy().astype('float64')
        metrics.update({name+'_offdiag_cosine':float(gram[off].astype('float64').mean()),
            name+'_spatial_token_variance':float(raw.double().var(dim=0,unbiased=True).mean()),
            name+'_normalized_token_variance':float(norm.double().var(dim=0,unbiased=True).mean()),
            name+'_effective_rank':float(np.exp(-np.sum(positive*np.log(positive)))),
            name+'_feature_norm_mean':float(feature_norm.mean()),name+'_feature_norm_std':float(feature_norm.std(ddof=1))})
    assert all(np.isfinite(v) for v in metrics.values())
    return metrics


def worker(shard):
    runtime(2+shard)
    import torch
    manifest=read('spatial_audit_manifest.json')
    rows=[r for r in manifest['records'] if r['worker_shard']==shard]
    assert len(rows)==256 and len({r['identity'] for r in rows})==128
    teacher,student,ta,sa=models()
    before=[state_sha(teacher),state_sha(student)]
    x,checks=inputs(rows[:2])
    ti,si,exact=inventory(teacher,student,x)
    records=[]; checksum_evidence=checks
    primary=exact['primary']; cache={}; handles=[]
    if primary:
        assert ti['grid_order_pass'] and si['grid_order_pass']
        def tc(m,args,out): cache['teacher']=out[:,ti['teacher_prefix_token_count']:].detach()
        def sc(m,args,out): cache['student']=out.detach()
        handles=[teacher.backbone.model.norm.register_forward_hook(tc),
                 student.get_submodule(primary['module_name']).register_forward_hook(sc)]
        for start in range(0,len(rows),4):
            batch=rows[start:start+4]; x,checks=inputs(batch)
            if start<8: checksum_evidence.extend(checks)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): teacher(x); student(x)
            t=cache['teacher']; s=cache['student'].flatten(2).transpose(1,2)
            assert t.shape[:2]==s.shape[:2]
            for row,tf,sf in zip(batch,t,s): records.append(dict(row,**image_stats(tf,sf,exact['teacher_grid'])))
            if (start+4)%32==0: print(f'SHARD={shard} IMAGES={start+4}/256',flush=True)
    for handle in handles:handle.remove()
    after=[state_sha(teacher),state_sha(student)]
    assert before==after and all(p.grad is None for m in [teacher,student] for p in m.parameters())
    assert sha(TEACHER)==TEACHER_SHA and sha(STUDENT)==STUDENT_SHA
    write(OUT/f'worker_{shard}.json',dict(shard=shard,physical_gpu=shard+2,pass_status=True,
          rows=records,teacher_inventory=ti,student_inventory=si,exact=exact,teacher_load=ta,student_load=sa,
          state_before=before,state_after=after,state_unchanged=True,geometric_checks=checksum_evidence,
          spatial_features_finite=bool(primary and len(records)==256),no_cross_view_position_matching=True))
    print(f'GPU{shard+2}_AUDIT_PASS=True',flush=True)


def summary(values):
    import numpy as np
    x=np.asarray(values,dtype='float64')
    return dict(n=len(x),mean=float(x.mean()),sample_std=float(x.std(ddof=1)),median=float(np.median(x)),
                p10=float(np.percentile(x,10)),p90=float(np.percentile(x,90)))


def merge():
    a,b=read('worker_0.json'),read('worker_1.json')
    assert a['pass_status'] and b['pass_status']
    assert a['exact']==b['exact'] and a['teacher_inventory']==b['teacher_inventory'] and a['student_inventory']==b['student_inventory']
    for filename,key in [('teacher_spatial_inventory.json','teacher_inventory'),('student_spatial_inventory.json','student_inventory'),('exact_grid_match.json','exact')]:
        write(OUT/filename,a[key])
    rows=a['rows']+b['rows']; manifest=read('spatial_audit_manifest.json')
    if not a['exact']['primary']:
        for name in ['spatial_geometry_summary.json','spatial_local_geometry.json','spatial_feature_diversity.json']:
            write(OUT/name,dict(status='NO_EXACT_GRID_NO_INTERPOLATION',statistics=None))
        return
    assert len(rows)==512 and {r['image_path'] for r in rows}=={r['image_path'] for r in manifest['records']}
    with (OUT/'spatial_geometry_per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    geom=['pearson','spearman','centered_cosine','mse']
    local=['teacher_local_mean','student_local_mean','teacher_far_mean','student_far_mean',
           'teacher_local_minus_far','student_local_minus_far','local_geometry_correlation']
    diversity=[k for k in rows[0] if k not in manifest['records'][0] and k not in geom+local]
    def summarize(keys):
        return {view:{key:summary([r[key] for r in rows if r['view']==view]) for key in keys} for view in ['drone','satellite']}
    write(OUT/'spatial_geometry_summary.json',dict(offdiagonal_only=True,ordered_pairs=True,
          statistic_reductions='float64; std ddof=1',views=summarize(geom),images=512))
    write(OUT/'spatial_local_geometry.json',dict(neighbors='Manhattan distance=1 (4-neighbor)',
          far='Manhattan distance >= ceil(max(H,W)/2), at least 2',
          local_geometry_correlation='Pearson across per-position mean 4-neighbor similarities',views=summarize(local)))
    write(OUT/'spatial_feature_diversity.json',dict(
          spatial_token_variance='Mean over channels of raw feature sample variance across positions (ddof=1)',
          normalized_token_variance='Same after per-token FP32 L2',
          effective_rank='exp(entropy(p)): p = singular values of per-token normalized uncentered token matrix / sum; sqrt(clipped FP32 Gram eigenvalues)',
          feature_norm_std='Sample std ddof=1 across positions before L2',views=summarize(diversity),
          pooled={k:summary([r[k] for r in rows]) for k in ['teacher_effective_rank','student_effective_rank']}))
    print('MERGE_PASS=True',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker','merge'])
    parser.add_argument('--shard',type=int,choices=[0,1]);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='worker':worker(args.shard)
    else:merge()
