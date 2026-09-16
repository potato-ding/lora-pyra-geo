"""U1652 TRAIN-only frozen-checkpoint BN context diagnostics; never an optimizer."""
import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
import random
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/home/dingyi/lora-pyra-geo')
sys.path.insert(0, str(ROOT))
BASE = ROOT/'src/checkpoint/student/CERTIFIED_R224'
OUT = BASE/'FINAL_ADUAL_BN_AUDIT_V1'
TEACHER_SHA = '1f5dd3a94e38d5e79bfff05b407959195eb59b9b9359f2380727f6a68fed3d78'
SEED = 20260916
CONDITIONS = ['C64_MIXED_CANONICAL', 'C32_MIXED', 'C32_VIEW_SEPARATED',
              'EVAL_RUNNING_STATS', 'C64_MATCHED8', 'C16_MIXED']
LOSSES = ['InfoNCE', 'Top_relational', 'Random_relational']
GROUPS = ['ALL_BACKBONE', 'BN_AFFINE_ONLY', 'NON_BN_BACKBONE_ONLY']
BN = (nn.BatchNorm1d, nn.BatchNorm2d)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    x = np.asarray([v for v in values if v is not None], dtype=np.float64)
    assert np.isfinite(x).all()
    if not len(x):
        return dict(n=0, mean=None, std=None, median=None, p10=None, p90=None)
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()),
                median=float(np.median(x)), p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)))


def corr(a, b, spearman=False):
    from scipy.stats import rankdata
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if spearman:
        a, b = rankdata(a), rankdata(b)
    a, b = a-a.mean(), b-b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a, b)/denom, -1, 1)) if denom else None


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        t = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def config(seed):
    return read(BASE/f'P2-TOP-RMLP-S{seed}'/'run_config.json')


def source_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def protected_assets():
    result = {}
    for s in range(3):
        run = BASE/f'P2-TOP-RMLP-S{s}'
        for p in run.rglob('*'):
            if p.is_file():
                result[str(p)] = sha(p)
    c = config(0)
    for key in ['middle_checkpoint', 'middle_config', 'stst_asset', 'original_stst_asset']:
        result[c[key]] = sha(c[key])
    return result


def prepare():
    assert not OUT.exists(), 'Refuse to overwrite an existing audit'
    from src.student.part1 import load_extended_asset
    from src.student.model import StudentModel
    from src.evaluation.model_loader import normalize_state
    from src.student.artifacts import validate_training_complete
    c = config(0)
    assert sha(c['middle_checkpoint']) == TEACHER_SHA
    assert sha(c['stst_asset']) == c['extended_stst_asset_sha256']
    load_extended_asset(c['stst_asset'], c['original_stst_asset'], TEACHER_SHA)
    train_source = (ROOT/'src/student/train.py').read_text()
    tree = ast.parse(train_source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'batch_loss')
    engine_calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'engine']
    assert len(engine_calls) == 1
    assert 'drone,satellite=batch[:2]' in train_source.replace(' ', '')
    assert 'images=torch.cat((drone,satellite)).to(device,non_blocking=True)' in train_source.replace(' ', '')
    head = {}; inventory = None; runtime_states = {}
    for s in range(3):
        cfg = config(s); run = BASE/f'P2-TOP-RMLP-S{s}'
        validate_training_complete(run)
        assert cfg['batch_size'] == 32 and cfg['cross_gpu_gather'] is False
        assert cfg['stst_asset'] == c['stst_asset'] and cfg['middle_checkpoint'] == c['middle_checkpoint']
        payload = torch.load(run/'best_model.pth', map_location='cpu', weights_only=True)
        state = normalize_state(payload)
        model = StudentModel(ckpt_path=None)
        model.load_state_dict(state, strict=True)
        # Training stores Student parameters/buffers in BF16; no value changes allowed by this cast.
        model.bfloat16()
        assert all(torch.equal(v, model.state_dict()[k].to(v.dtype)) for k, v in state.items())
        runtime_states[str(s)] = tensor_hash(model.state_dict())
        files = [str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()]
        weight_files = [f for f in files if f.endswith(('.pt', '.pth'))]
        assert set(weight_files) == {'best_model.pth', 'last_model.pth'}, 'Inspect additional head candidates first'
        for filename in weight_files:
            saved = torch.load(run/filename, map_location='cpu', weights_only=True)
            model.load_state_dict(normalize_state(saved), strict=True)
        head[str(s)] = dict(exact_best_head_recoverable=False, best_epoch=payload['epoch'],
            best_checkpoint=str(run/'best_model.pth'), best_sha256=sha(run/'best_model.pth'),
            checkpoint_keys=list(payload), state_keys=list(state), weight_files=weight_files,
            files_inspected=files, reason='Only strict bare Student best/last states persisted; no same-epoch projector/gate state.',
            gradient_type='PROJECTOR_FREE_RELATIONAL_DIAGNOSTIC',
            saver_source='src/student/canonical_selection.py:select_epoch -> canonical_state -> deployment_state_dict')
        rows = []
        for name, module in model.named_modules():
            if isinstance(module, BN):
                if name.startswith('backbone.features.'):
                    block = int(name.split('.')[2]); stage = next((f'stage{i+1}' for i, end in enumerate([5,11,37,42]) if block <= end), 'unknown')
                else:
                    stage = 'retrieval_neck'
                rows.append(dict(name=name, type=type(module).__name__, num_features=module.num_features,
                    affine=module.affine, track_running_stats=module.track_running_stats,
                    running_mean_shape=list(module.running_mean.shape), running_var_shape=list(module.running_var.shape), stage=stage))
        if inventory is None:
            inventory = rows
        else:
            assert rows == inventory
    train = Path(c['train_data_dir']).resolve()
    assert train == (ROOT/'data/U1652/train').resolve()
    identities = sorted(p.name for p in (train/'satellite').iterdir() if p.is_dir() and (train/'drone'/p.name).is_dir())
    assert len(identities) == 701
    rng = random.Random(SEED); batches = []
    for index in range(32):
        ids = rng.sample(identities, 32); entries = []
        for j, identity in enumerate(ids):
            paths = {}
            for view in ['drone', 'satellite']:
                images = sorted(p for p in (train/view/identity).iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png'])
                p = rng.choice(images) if view == 'drone' else images[0]
                assert p.resolve().is_relative_to(train)
                paths[view+'_path'] = str(p)
                paths[view+'_sha256'] = sha(p)
            entries.append(dict(identity=identity, batch_index=index, position=j,
                role='target' if j < 16 else 'context', matched8_role='target' if j < 8 else 'context', **paths))
        batches.append(dict(batch_index=index, entries=entries))
    OUT.mkdir(parents=True)
    write(OUT/'audit_sample_manifest.json', dict(seed=SEED, split='train', train_root=str(train),
        num_audit_batches=32, target_identities=16, context_identities=16, batches=batches))
    write(OUT/'bn_inventory.json', dict(per_seed_identical=True, layers=inventory, count=len(inventory)))
    write(OUT/'audit_head_recoverability.json', head)
    write(OUT/'audit_config.json', dict(source_commit=source_commit(), num_audit_batches=32, manifest_seed=SEED,
        CANONICAL_TRAIN_FORWARD_CONFIRMED=True, canonical_source_sha256=sha(ROOT/'src/student/train.py'),
        runtime_states=runtime_states, protected_assets=protected_assets(),
        conditions=CONDITIONS, teacher_path=c['middle_checkpoint'], teacher_sha256=TEACHER_SHA,
        stst_asset=c['stst_asset'], stst_asset_sha256=sha(c['stst_asset']),
        precision='Student BF16 storage/input/forward as training; descriptors, similarities, losses FP32; gradients accumulated as FP32 diagnostics',
        preprocessing='get_paired_cross_view_val_transforms(224) applied ONLY to U1652 TRAIN images; no augmentation',
        gradient_groups='ALL_BACKBONE excludes retrieval neck and logit_scale; BN_AFFINE_ONLY is the backbone BN subset. Per-layer BN affine audit additionally includes retrieval neck.',
        layer_output_drift='L2 distance of per-sample flattened unit-normalized BN outputs',
        batch_variance='biased variance used in BN forward; eval compares frozen running variance (unbiased estimator) as actually used',
        correlation='All target D2S entries including positive diagonal; constant-vector correlations reported null, never NaN',
        std_definition='population std (ddof=0); 3-seed summaries are unweighted statistics across per-seed summary values',
        n16_control='First 8 identities are targets; other 24 from the same 32-identity batch are context only',
        loss_definition='Exact PairInfoNCE; Top/Random are MSE of Student and frozen Teacher target similarity matrices, NOT original A-Dual-STST KD losses',
        bn_restore='Clone every registered BN buffer before each forward; restore by replacing buffer references immediately after forward, preserving autograd saved-tensor versions; verify again after backward and condition.'))
    print('PREPARE_PASS=True', flush=True)


def take_batch(batch, transform):
    from PIL import Image
    result = []
    for view in ['drone', 'satellite']:
        tensors = []
        for entry in batch['entries']:
            path = Path(entry[view+'_path'])
            assert sha(path) == entry[view+'_sha256']
            with Image.open(path) as image:
                tensors.append(transform(image=np.array(image.convert('RGB')))['image'])
        result.append(torch.stack(tensors).cuda().bfloat16())
    return result


def plans(condition):
    n = 8 if condition in ['C16_MIXED', 'C64_MATCHED8'] else 16
    if condition == 'C32_VIEW_SEPARATED':
        return n, [('Drone', list(range(32)), list(range(n)), []),
                   ('Satellite', list(range(32,64)), [], list(range(n)))]
    if condition in ['C64_MIXED_CANONICAL', 'C64_MATCHED8', 'EVAL_RUNNING_STATS']:
        return n, [('Mixed', list(range(64)), list(range(n)), list(range(32,32+n)))]
    return n, [('Mixed', list(range(n))+list(range(32,32+n)), list(range(n)), list(range(n,2*n)))]


class LayerCapture:
    def __init__(self, model):
        self.model = model; self.modules = {n:m for n,m in model.named_modules() if isinstance(m, BN)}
        self.hooks = [m.register_forward_hook(self.hook(n)) for n,m in self.modules.items()]
        self.reference = {}; self.rows = []; self.mode = None

    def start(self, condition, reference=None):
        self.condition = condition; self.rows = []; self.reference = reference or {}; self.new_reference = {}

    def hook(self, name):
        def call(module, inputs, output):
            with torch.no_grad():
                x = inputs[0].detach().float()
                dims = (0,)+tuple(range(2, x.ndim))
                var, mean = torch.var_mean(x, dim=dims, unbiased=False)
                target = {view:output[idx].detach().clone() for view,idx in self.target_indices.items() if idx}
                if self.condition in ['C64_MIXED_CANONICAL', 'C64_MATCHED8']:
                    self.new_reference[name] = dict(mean=mean, var=var, outputs=target)
                    return
                ref = self.reference[name]
                if self.condition == 'EVAL_RUNNING_STATS':
                    mean, var = module.running_mean.float(), module.running_var.float()
                a,b = mean.cpu().numpy(),ref['mean'].cpu().numpy()
                u,v = var.cpu().numpy(),ref['var'].cpu().numpy()
                row = dict(layer=name, forward_view=self.mode,
                    mean_abs_delta_mean=float((mean-ref['mean']).abs().mean()),
                    mean_abs_delta_var=float((var-ref['var']).abs().mean()),
                    relative_var_delta=float((var-ref['var']).abs().mean()/ref['var'].abs().mean().clamp_min(1e-12)),
                    channelwise_mean_correlation=corr(a,b), channelwise_variance_correlation=corr(u,v))
                for view,t in target.items():
                    a = F.normalize(t.float().flatten(1), dim=1)
                    b = F.normalize(ref['outputs'][view].float().flatten(1), dim=1)
                    row[view+'_normalized_l2_drift'] = float((a-b).norm(dim=1).mean())
                self.rows.append(row)
        return call

    def close(self):
        for hook in self.hooks:
            hook.remove()


def guarded_forward(model, images, batch_stats):
    modules = [m for m in model.modules() if isinstance(m, BN)]
    model.eval()
    for m in modules:
        m.train(batch_stats)
    assert all(not m.training for m in model.modules() if not isinstance(m, BN))
    before = [(m,{k:v.detach().clone() for k,v in m._buffers.items() if v is not None}) for m in modules]
    try:
        result = model(images)
    finally:
        for m, buffers in before:
            # Do not copy_ into tensors saved by autograd, which increments their versions.
            for key,value in buffers.items():
                m._buffers[key] = value
        assert all(torch.equal(m._buffers[k],v) for m,b in before for k,v in b.items())
    return result


def gradient_pair(a, b, indices):
    dot = torch.zeros((), device='cuda', dtype=torch.float64)
    aa = torch.zeros_like(dot); bb = torch.zeros_like(dot)
    for i in indices:
        # Tensor-wise streaming reduction, never flatten or persist the full gradient vector.
        dot += (a[i]*b[i]).sum(dtype=torch.float64)
        aa += a[i].square().sum(dtype=torch.float64); bb += b[i].square().sum(dtype=torch.float64)
    an,bn = float(aa.sqrt()),float(bb.sqrt())
    return dict(cosine=float((dot/(aa*bb).sqrt()).clamp(-1,1)) if an and bn else None,
                norm=an, reference_norm=bn, norm_ratio=an/bn if bn else None)


def geometry(z, ids, gallery_ids=None):
    n = len(ids); d,s = z[:n],z[n:]
    matrix = (d@s.T).detach().cpu().numpy()
    labels = np.asarray(ids)
    gallery = np.asarray(ids if gallery_ids is None else gallery_ids)
    positive = labels[:,None] == gallery[None,:]
    assert (positive.sum(1) == 1).all()
    pos = np.array([row[mask].mean() for row,mask in zip(matrix,positive)])
    neg = np.where(positive, -np.inf, matrix).max(1)
    return matrix, dict(positive_similarity=pos.tolist(), hardest_negative_similarity=neg.tolist(), margin=(pos-neg).tolist())


def worker(seed, limit=None, directory=None):
    destination = Path(directory) if directory else OUT
    from src.student.model import StudentModel
    from src.student.objective import PairInfoNCE
    from src.student.part1 import load_extended_asset
    from src.evaluation.model_loader import normalize_state, load_encoder
    from src.dataset.teacher.transforms import get_paired_cross_view_val_transforms
    torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.cuda.set_device(0)
    settings = read(OUT/'audit_config.json'); cfg = config(seed)
    assert source_commit() == settings['source_commit']
    assert not read(OUT/'audit_head_recoverability.json')[str(seed)]['exact_best_head_recoverable']
    checkpoint = BASE/f'P2-TOP-RMLP-S{seed}'/'best_model.pth'
    model = StudentModel(ckpt_path=None)
    model.load_state_dict(normalize_state(torch.load(checkpoint,map_location='cpu',weights_only=True)), strict=True)
    model.cuda().bfloat16().eval(); model.logit_scale.requires_grad_(False)
    initial = tensor_hash(model.state_dict())
    assert initial == settings['runtime_states'][str(seed)]
    initial_buffers = {k:v.detach().clone() for k,v in model.named_buffers()}
    teacher,_ = load_encoder('middle',cfg['middle_checkpoint'],cfg['middle_config'],'cuda:0')
    teacher.eval(); teacher_initial = tensor_hash(teacher.state_dict())
    asset = load_extended_asset(cfg['stst_asset'],cfg['original_stst_asset'],TEACHER_SHA)
    mean,top,rand = [asset[k].float().cuda() for k in ['teacher_mean','top128_basis','random32_A']]
    transform = get_paired_cross_view_val_transforms(img_size=[224,224])
    batches = read(OUT/'audit_sample_manifest.json')['batches'][:limit]
    capture = LayerCapture(model)
    parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    pnames = [n for n,p in parameters]; params = [p for n,p in parameters]
    bn_names = {n+'.'+a for n,m in capture.modules.items() for a in ['weight','bias'] if getattr(m,a) is not None}
    groups = dict(ALL_BACKBONE=[i for i,n in enumerate(pnames) if n.startswith('backbone.')],
        BN_AFFINE_ONLY=[i for i,n in enumerate(pnames) if n.startswith('backbone.') and n in bn_names],
        NON_BN_BACKBONE_ONLY=[i for i,n in enumerate(pnames) if n.startswith('backbone.') and n not in bn_names])
    criterion = PairInfoNCE(label_smoothing=cfg['label_smoothing'])
    data = dict(descriptor=[],retrieval=[],teacher=[],layers=[],gradient=[],interaction=[],affine=[],sanity=[])
    start = time.time()
    for batch in batches:
        b = batch['batch_index']; drone,sat = take_batch(batch,transform); images = torch.cat([drone,sat])
        assert images.shape == (64,3,224,224)
        with torch.no_grad():
            td = teacher(images).detach().float()
            projected = [F.normalize((td-mean)@basis,dim=1) for basis in [top,rand]]
        assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        reference = None; ref_grad = None; ref_z = None; ref_matrix = None; ref_geo = None
        for condition in CONDITIONS:
            n, forwards = plans(condition)
            is_reference = condition in ['C64_MIXED_CANONICAL','C64_MATCHED8']
            if is_reference:
                reference = None; ref_grad = None; ref_z = None; ref_matrix = None; ref_geo = None
            capture.start(condition,reference)
            outputs = {}; observed = []
            for mode,index,di,si in forwards:
                capture.mode = mode; capture.target_indices = dict(Drone=di,Satellite=si)
                z = guarded_forward(model,images[index],condition != 'EVAL_RUNNING_STATS')
                observed.append(len(index))
                if di: outputs['Drone'] = z[di]
                if si: outputs['Satellite'] = z[si]
            expected = [32,32] if condition=='C32_VIEW_SEPARATED' else [64 if condition in ['C64_MIXED_CANONICAL','C64_MATCHED8','EVAL_RUNNING_STATS'] else 2*n]
            assert observed == expected
            z = torch.cat([outputs['Drone'],outputs['Satellite']])
            assert z.shape == (2*n,512) and torch.isfinite(z).all()
            ids = [e['identity'] for e in batch['entries'][:n]]
            assert len(set(ids))==n
            # Same ordered identity lists are verified before exact training InfoNCE's diagonal labels.
            ss = z[:n]@z[n:].T
            targets = [t[:n]@t[32:32+n].T for t in projected]
            losses = [criterion(z[:n],z[n:],model.logit_scale.exp()),
                      (ss-targets[0]).square().mean(),(ss-targets[1]).square().mean()]
            gradients = []
            for j,loss in enumerate(losses):
                assert loss.dtype == torch.float32 and torch.isfinite(loss)
                gs = torch.autograd.grad(loss,params,retain_graph=j<2,allow_unused=False)
                gradients.append([g.detach().float() for g in gs])
                assert all(torch.isfinite(g).all() for g in gradients[-1])
            assert all(p.grad is None for p in model.parameters())
            assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in initial_buffers.items())
            matrix,geo = geometry(z.detach(),ids)
            if is_reference:
                reference = capture.new_reference; ref_grad = gradients
                ref_z = z.detach(); ref_matrix = matrix; ref_geo = geo
            metadata = dict(seed=seed,batch=b,condition=condition,target_pairs=n)
            data['sanity'].append(dict(**metadata,forward_sizes=observed,buffers_restored=True,only_target_in_loss=True))
            for view,sl in [('Drone',slice(0,n)),('Satellite',slice(n,2*n))]:
                cosine = (z.detach()[sl]*ref_z[sl]).sum(1).clamp(-1,1)
                values = dict(cosine=cosine.cpu().tolist(),angle_degree=(torch.acos(cosine)*180/math.pi).cpu().tolist(),
                    l2_distance=(z.detach()[sl]-ref_z[sl]).norm(dim=1).cpu().tolist())
                if not is_reference:
                    data['descriptor'].append(dict(**metadata,view=view,**values))
            retrieval = dict(**metadata,identities=ids,query_statistics=geo,
                pearson_vs_reference=corr(matrix,ref_matrix),spearman_vs_reference=corr(matrix,ref_matrix,True))
            for key,values in geo.items():
                retrieval['mean_'+key] = float(np.mean(values))
                retrieval['delta_'+key+'_vs_reference'] = float(np.mean(values)-np.mean(ref_geo[key]))
            data['retrieval'].append(retrieval)
            for label,target in zip(['Top128','Random32_A'],targets):
                t = target.detach().cpu().numpy()
                data['teacher'].append(dict(**metadata,branch=label,pearson=corr(matrix,t),spearman=corr(matrix,t,True)))
            data['layers'].extend(dict(**metadata,**row) for row in capture.rows)
            for j,lossname in enumerate(LOSSES):
                for group,idx in groups.items():
                    data['gradient'].append(dict(**metadata,loss=lossname,group=group,
                        loss_value=float(losses[j].detach()),**gradient_pair(gradients[j],ref_grad[j],idx)))
                all_bn_energy = sum(float(gradients[j][i].square().sum(dtype=torch.float64)) for i,name in enumerate(pnames) if name in bn_names)
                for name,module in capture.modules.items():
                    row = dict(**metadata,layer=name,loss=lossname)
                    energies = []
                    for a,label in [('weight','gamma'),('bias','beta')]:
                        i = pnames.index(name+'.'+a)
                        energy = float(gradients[j][i].square().sum(dtype=torch.float64)); energies.append(energy)
                        row['grad_norm_'+label] = math.sqrt(energy)
                        row[label+'_squared_norm_fraction_all_bn'] = energy/all_bn_energy if all_bn_energy else None
                    row['layer_squared_norm_fraction_all_bn'] = sum(energies)/all_bn_energy if all_bn_energy else None
                    row['layer_norm_fraction_all_bn_norm'] = math.sqrt(sum(energies)/all_bn_energy) if all_bn_energy else None
                    data['affine'].append(row)
            for i,j in [(0,1),(0,2),(1,2)]:
                for group,idx in groups.items():
                    data['interaction'].append(dict(**metadata,pair=LOSSES[i]+'__'+LOSSES[j],group=group,
                        cosine=gradient_pair(gradients[i],gradients[j],idx)['cosine']))
            del gradients,gs,losses,outputs,z,ss
        capture.reference={};capture.new_reference={};reference=None;ref_grad=None;ref_z=None
        write(destination/f'progress_s{seed}.json',dict(seed=seed,batches_completed=b+1,total_batches=len(batches),elapsed_seconds=time.time()-start))
        print(f'S{seed} BATCH_COMPLETE={b+1}/{len(batches)}',flush=True)
    capture.close(); model.eval()
    assert tensor_hash(model.state_dict()) == initial
    assert tensor_hash(teacher.state_dict()) == teacher_initial
    assert all(sha(path)==digest for path,digest in settings['protected_assets'].items())
    data['status'] = dict(seed=seed,audit_pass=True,batches=len(batches),BN_BUFFER_RESTORE_PASS=True,
        AUDIT_STATE_IMMUTABILITY_PASS=True,initial_state_hash=initial,final_state_hash=tensor_hash(model.state_dict()),
        teacher_state_unchanged=True,optimizer_steps=0,checkpoint_writes=0,source_commit=source_commit(),
        gradient_type='PROJECTOR_FREE_RELATIONAL_DIAGNOSTIC',peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    write(destination/f'seed_data_s{seed}.json',data)
    print(f'S{seed}_AUDIT_PASS=True',flush=True)


def summarize(rows, keys, fields):
    grouped = defaultdict(lambda:defaultdict(list))
    for row in rows:
        key = tuple(row[k] for k in keys)
        for field in fields:
            v = row.get(field)
            grouped[key][field].extend(v if isinstance(v,list) else [v])
    return [dict(zip(keys,key),statistics={f:stats(v) for f,v in fields_.items()}) for key,fields_ in grouped.items()]


def aggregate_seed_summaries(summary, keys):
    grouped = defaultdict(lambda:defaultdict(list))
    for row in summary:
        key = tuple(row[k] for k in keys if k!='seed')
        for metric,statistics in row['statistics'].items():
            for statistic,value in statistics.items():
                if statistic != 'n':
                    grouped[key][metric+'.'+statistic].append(value)
    return [dict(zip([k for k in keys if k!='seed'],key),across_seed_statistics={k:stats(v) for k,v in values.items()}) for key,values in grouped.items()]


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def finalize():
    settings = read(OUT/'audit_config.json')
    assert source_commit()==settings['source_commit']
    all_data = [read(OUT/f'seed_data_s{s}.json') for s in range(3)]
    assert all(d['status']['audit_pass'] and d['status']['batches']==32 for d in all_data)
    specs = {
      'descriptor':('audit_descriptor_bn_drift.json',['seed','condition','view'],['cosine','angle_degree','l2_distance']),
      'retrieval':('audit_retrieval_geometry.json',['seed','condition'],['mean_positive_similarity','mean_hardest_negative_similarity','mean_margin','delta_positive_similarity_vs_reference','delta_hardest_negative_similarity_vs_reference','delta_margin_vs_reference','pearson_vs_reference','spearman_vs_reference']),
      'teacher':('audit_teacher_geometry_bn.json',['seed','condition','branch'],['pearson','spearman']),
      'layers':('audit_bn_layer_drift.json',['seed','condition','layer','forward_view'],['mean_abs_delta_mean','mean_abs_delta_var','relative_var_delta','channelwise_mean_correlation','channelwise_variance_correlation','Drone_normalized_l2_drift','Satellite_normalized_l2_drift']),
      'gradient':('audit_gradient_context.json',['seed','condition','loss','group'],['cosine','norm','reference_norm','norm_ratio','loss_value']),
      'interaction':('audit_objective_interaction.json',['seed','condition','pair','group'],['cosine']),
      'affine':('audit_bn_affine_gradients.json',['seed','condition','layer','loss'],['grad_norm_gamma','grad_norm_beta','gamma_squared_norm_fraction_all_bn','beta_squared_norm_fraction_all_bn','layer_squared_norm_fraction_all_bn','layer_norm_fraction_all_bn_norm'])}
    combined = {}; seed_summary = {}; three_summary = {}
    for kind,(filename,keys,fields) in specs.items():
        rows = [row for d in all_data for row in d[kind]]
        per_seed = summarize(rows,keys,fields); three = aggregate_seed_summaries(per_seed,keys)
        result = dict(per_seed=per_seed,three_seed=three,per_batch=rows)
        if kind == 'layers':
            ranked = []
            by_layer = defaultdict(list)
            for row in per_seed:
                if row['condition']=='C16_MIXED':continue
                for view in ['Drone','Satellite']:
                    value = row['statistics'][view+'_normalized_l2_drift']['mean']
                    if value is not None:by_layer[row['layer']].append(value)
            ranked = sorted((dict(layer=k,mean_target_normalized_l2_drift=float(np.mean(v))) for k,v in by_layer.items()),key=lambda x:x['mean_target_normalized_l2_drift'],reverse=True)
            result['top10_context_sensitive_layers'] = ranked[:10]
            result['ranking_definition'] = 'Unweighted mean across seeds, primary non-reference contexts and observed views; N16 excluded.'
        write(OUT/filename,result);combined[kind]=rows;seed_summary[kind]=per_seed;three_summary[kind]=three
    write_csv(OUT/'audit_bn_layer_drift.csv',combined['layers'])
    write_csv(OUT/'audit_bn_affine_gradients.csv',combined['affine'])
    write(OUT/'audit_seed_summary.json',dict(status=[d['status'] for d in all_data],statistics=seed_summary))
    write(OUT/'audit_3seed_summary.json',dict(seed_count=3,std_ddof=0,statistics=three_summary))
    for s,d in enumerate(all_data):
        write_csv(OUT/f'per_batch_s{s}.csv',d['gradient']+d['interaction'])
    assert all(sha(path)==digest for path,digest in settings['protected_assets'].items())
    head = read(OUT/'audit_head_recoverability.json')
    text = [
        'Purpose: frozen Final A-Dual-STST BN context knowledge-absorption diagnostics on University-1652 TRAIN only.',
        'Final method: shared Top Linear(512,128) + shared residual MLP 512->920->128 GELU, learned alpha_top; shared Random Linear(512,32); bare RepViT-M1.5 deployment.',
        'Teacher: '+settings['teacher_path']+' SHA256='+settings['teacher_sha256'],
        'STST asset: '+settings['stst_asset']+' SHA256='+settings['stst_asset_sha256'],
        'Sample manifest seed='+str(SEED)+' NUM_AUDIT_BATCHES=32. All seeds use identical images and ordering.',
        'C64: Drone[target16+context16] then Satellite[target16+context16], one N64 forward; target16 loss only.',
        'C32_MIXED: target Drone16 then Satellite16, one N32 forward; identical target16 negatives.',
        'C32_VIEW_SEPARATED: Drone[target16+context16] N32 and Satellite[target16+context16] N32 in separate forwards; target16 losses only.',
        'EVAL_RUNNING_STATS: same input as C64, frozen running stats, target16 losses only.',
        'C16_MIXED: first8 target pairs, N16. C64_MATCHED8: same target8 plus other24 identities as context, N64. Only compare these matched8 conditions.',
        'BN-only mode: model.eval(), then only existing BatchNorm1d/2d modules.train(True) in batch-stat contexts. Other modules remain eval. No structural replacement.',
        settings['bn_restore'],settings['precision'],settings['preprocessing'],settings['gradient_groups'],settings['layer_output_drift'],settings['batch_variance'],settings['correlation'],settings['std_definition'],
        'No optimizer was constructed. torch.autograd.grad used solely for diagnostics; no parameter or projector update. No recalibration, new checkpoint, PCA, SVD or random basis generation.',
        'No test-split data or SUES/GTA used. No tensor caches saved.',
        'Source git commit='+settings['source_commit']]
    for s in range(3):
        h=head[str(s)];text += [f'S{s} checkpoint={h["best_checkpoint"]} SHA256={h["best_sha256"]} selected_epoch={h["best_epoch"]}',
            f'S{s} EXACT_BEST_HEAD_RECOVERABLE=False; GRADIENT_TYPE=PROJECTOR_FREE_RELATIONAL_DIAGNOSTIC']
    text += ['InfoNCE reuses src.student.objective.PairInfoNCE and persisted logit_scale; label_smoothing=0.1. Top/Random relational loss=mean((Student target D2S matrix - Teacher projected target D2S matrix)^2). These are NOT the original A-Dual-STST KD losses.',
             'All checkpoint/run/config/asset hashes unchanged; all Student and Teacher state tensors unchanged after runtime precision initialization.']
    (OUT/'README.txt').write_text('\n'.join(text)+'\n')
    # Intermediate seed JSONs contain scalar statistics only; final package uses an explicit deliverable allowlist.
    files = ['audit_config.json','audit_sample_manifest.json','bn_inventory.json','audit_head_recoverability.json',
             'audit_seed_summary.json','audit_3seed_summary.json','README.txt','audit_bn_layer_drift.csv','audit_bn_affine_gradients.csv']
    files += [v[0] for v in specs.values()]+[f'per_batch_s{s}.csv' for s in range(3)]
    assert len(files)==len(set(files))
    archive = OUT/'FINAL_ADUAL_BN_AUDIT_V1_RESULTS.tar.gz'
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tf:
            for filename in files:tf.add(OUT/filename,arcname=filename,recursive=False)
    with tarfile.open(archive) as tf:
        assert sorted(tf.getnames())==sorted(files)
        assert all(m.isfile() and (m.name.endswith(('.json','.csv')) or m.name=='README.txt') for m in tf.getmembers())
        for m in tf.getmembers():assert hashlib.sha256(tf.extractfile(m).read()).hexdigest()==sha(OUT/m.name)
    write(OUT/'package_status.json',dict(PASS=True,path=str(archive),size=archive.stat().st_size,sha256=sha(archive),file_count=len(files),source_commit=source_commit()))
    print('FINALIZE_PASS=True',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','finalize'])
    p.add_argument('--seed',type=int,choices=[0,1,2]);p.add_argument('--limit',type=int);p.add_argument('--directory')
    args=p.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='worker':worker(args.seed,args.limit,args.directory)
    else:finalize()
