"""TRAIN-only, immutable D0 knowledge absorption diagnostics. Never trains a model."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
STUDENT = ROOT / 'src/checkpoint/student/CERTIFIED_R224'
OUT = STUDENT / '_AUDITS/D0_KNOWLEDGE_ABSORPTION_V1'
TRAIN = ROOT / 'data/U1652/train'
TEACHER = ROOT / 'src/checkpoint/middle_teacher/CERTIFIED_R224/SAM-MABV2-RHO010-S0'
BANK = STUDENT / 'STST_ASSETS/SAM-MABV2-RHO010-S0_train_shared_k32.pt'
SEED = 20260914
LAMBDAS = [0., 1e-8, 1e-6, 1e-4, 1e-2, 1.]
DOMAINS = ('drone', 'satellite')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def pair_report(name, value):
    write(name + '.json', value)
    (OUT / (name + '.txt')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def train_guard(path, root=TRAIN):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Only University-1652 TRAIN paths are allowed: ' + str(path))
    return path


def stable_ids(ids, salt='outer'):
    return sorted(ids, key=lambda pid: (hashlib.sha256(f'{SEED}:{salt}:{pid}'.encode()).hexdigest(), pid))


def identity_split(ids):
    ids = stable_ids(ids)
    if len(ids) != 701 or len(set(ids)) != 701:
        raise ValueError('Exactly 701 unique TRAIN identities required')
    fit, held = ids[:560], ids[560:]
    internal = stable_ids(fit, 'internal')
    result = dict(seed=SEED, sort='SHA256(seed:salt:identity), identity tie-break',
                  fit=fit, heldout=held, internal_fit=internal[:448], internal_val=internal[448:])
    result['identity_overlap'] = len(set(fit) & set(held))
    result['sha256'] = digest(result)
    return result


def rows_from_train():
    rows, identity_sets = [], []
    for domain in DOMAINS:
        ids = sorted(x.name for x in (TRAIN / domain).iterdir() if x.is_dir())
        identity_sets.append(ids)
        for pid in ids:
            paths = sorted(p for p in (TRAIN / domain / pid).iterdir()
                           if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
            if not paths or (domain == 'satellite' and len(paths) != 1):
                raise ValueError('Invalid TRAIN identity images')
            rows.extend(dict(path=str(train_guard(p)), domain=domain, pid=pid) for p in paths)
    assert identity_sets[0] == identity_sets[1]
    return rows, identity_sets[0]


def run_dir(method, seed):
    if method not in ('B0','D0') or seed not in range(3):
        raise ValueError('Only certified B0/D0 seeds 0,1,2')
    return STUDENT / (
        f'B0-BASELINE-S{seed}' if method == 'B0' else f'D0-DUAL-STST-S{seed}')


def state_hash(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextmanager
def restored_bn(model, batch_statistics=True):
    """Only BN is in train mode. Restore buffers AND all mode flags, even on errors."""
    modes = {m: m.training for m in model.modules()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    model.eval()
    if batch_statistics:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.train()
    try:
        yield
    finally:
        with torch.no_grad():
            for k, v in model.named_buffers():
                v.copy_(buffers[k])
        for module, mode in modes.items():
            module.training = mode
        assert all(torch.equal(v, buffers[k]) for k, v in model.named_buffers())


def prepare():
    from src.evaluation.model_loader import normalize_state
    from src.student.dual_stst import load_stst_asset
    if (OUT / 'AUDIT_CONFIG.json').exists():
        raise FileExistsError('Audit already prepared; reuse explicit downstream stages')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / '_CACHE').mkdir(exist_ok=True)
    protected, certified, heads = {}, {}, {}
    teacher_sha, bank_sha = sha(TEACHER / 'best_model.pth'), sha(BANK)
    asset = load_stst_asset(BANK, expected_teacher_sha256=teacher_sha)
    for method in ('B0', 'D0'):
        summary = read(STUDENT / f'{method}_3SEED_SUMMARY.json')
        assert summary[f'{method}_3SEED_CERTIFIED'] is True
        certified[method] = True
        for seed in range(3):
            run = run_dir(method, seed)
            cfg, best = read(run / 'run_config.json'), read(run / 'best_metrics.json')
            assert cfg['protocol_id'] == 'STU-1G-B32-R224-v1'
            assert cfg['seed'] == seed and cfg['method'] == ('baseline' if method == 'B0' else 'dual_stst')
            train_guard(cfg['train_data_dir'])
            if method == 'D0':
                for k, v in dict(top_dim=32, random_dim=32, stst_weight=.2, stst_warmup_epochs=5).items():
                    assert cfg[k] == v
                assert cfg['middle_teacher_sha256'] == teacher_sha and cfg['stst_asset_sha256'] == bank_sha
            formal = read(run / 'test_1652_best.json')
            cksha = sha(run / 'best_model.pth')
            assert formal['checkpoint_sha256'] == cksha
            for direction in ('D2S', 'S2D'):
                for a, b in [('R1', 'R@1'), ('R5', 'R@5'), ('AP', 'AP')]:
                    assert best[f'{direction}_{a}'] == formal['results'][direction][b]
            assert summary['seeds'][seed]['train_pass'] and summary['seeds'][seed]['best_reload_match']
            for name in ['best_model.pth', 'last_model.pth', 'run_config.json', 'best_metrics.json',
                         'epoch_metrics.json', 'train.log']:
                path = run / name
                protected[str(path)] = dict(size=path.stat().st_size, sha256=sha(path))
            if method == 'D0':
                heads[str(seed)] = {}
                for name in ('best_model.pth', 'last_model.pth'):
                    state = normalize_state(torch.load(run / name, map_location='cpu', weights_only=True))
                    top = [k for k in state if 'projector_top' in k or 'top_head' in k]
                    rnd = [k for k in state if 'projector_random' in k or 'random_head' in k]
                    heads[str(seed)][name] = dict(TRAINED_TOP_HEAD_AVAILABLE=bool(top),
                        TRAINED_RANDOM_HEAD_AVAILABLE=bool(rnd), matching_keys=top+rnd,
                        all_keys=list(state), tensor_count=len(state))
    for path in [TEACHER / 'best_model.pth', TEACHER / 'run_config.json', BANK]:
        protected[str(path)] = dict(size=path.stat().st_size, sha256=sha(path))
    rows, ids = rows_from_train()
    split = identity_split(ids)
    write('TRAIN_ID_SPLIT.json', split)
    write('_CACHE/TRAIN_IMAGE_INDEX.json', dict(AUDIT_ONLY=True, NOT_FOR_TRAINING=True,
        TRAIN_ONLY=True, rows=rows, rows_sha256=digest(rows)))
    write('HEAD_PERSISTENCE.json', heads)
    config = dict(AUDIT_ONLY=True, NOT_FOR_TRAINING=True, TRAIN_ONLY=True,
        protocol_id='STU-1G-B32-R224-v1', dataset_root=str(TRAIN), fit_ids=560, heldout_ids=141,
        internal_fit_ids=448, internal_val_ids=112, split_seed=SEED, lambda_candidates=LAMBDAS,
        probe='FP64 centered analytical ridge with unregularized intercept; FP32 official targets',
        lambda_selection='highest equal-view mean cosine on internal 112 TRAIN IDs; first candidate wins ties',
        fit_weighting='half total weight per domain, uniform image weights within domain',
        report_weighting='combined pools individual images; view-balanced cosine also reported',
        parameter_storage='Student FP32, CUDA BF16 autocast; Teacher certified BF16',
        bn_parameter_storage='Student FP32 CUDA BF16 autocast (isolate BN mode/context at fixed precision)',
        image_size=224, extraction_batch=32, contexts_per_anchor=8, bn_total_image_sizes=[16,32,64],
        bn_anchor_design='141 identity pairs, one stable drone and the unique satellite per identity; '
        'half a batch is fixed anchor pairs, half varying companions; each anchor receives 8 contexts',
        primary_bn_total_images=64, bn_simulation_sizes=[16,32],
        decision_rules=dict(probe='heldout pooled cosine mean increases in all three matched seeds',
            subspace_geometry='mean Pearson across drone/satellite/cross-view increases in all three seeds',
            global_geometry='D2S matrix Pearson increases in all three seeds',
            drift='canonical N=64 pooled anchor/context mean 1-cos decreases in all three seeds',
            sensitivity_present='each seed N=64 mean drift > 1e-6; descriptive numerical threshold'),
        protected_files=protected, certified=certified,
        source_commit_before_audit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        target_asset_metadata=asset['metadata'], formal_training_started=False,
        limitations=['Heldout identities are disjoint for fresh probe fitting only. Both backbones and the '
          'fixed STST bank previously saw all 701 TRAIN identities. This is not unseen-identity generalization.',
          'Checkpoint selection previously used U1652 TEST under the approved protocol; no test images or '
          'test metric values are used to choose any audit hyperparameter.'])
    write('AUDIT_CONFIG.json', config)
    (OUT / 'SOURCE_PROVENANCE.txt').write_text(json.dumps(config, indent=2)+'\n')
    print('PREPARED', len(rows), 'images', flush=True)


class AuditImages(Dataset):
    def __init__(self, rows):
        from src.dataset.transforms import get_test_transforms
        self.rows = rows
        self.transform = get_test_transforms([224,224])
        for row in rows:
            train_guard(row['path'])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.rows[index]['path']) as image:
            array = np.array(image.convert('RGB'))
        return self.transform(image=array)['image'], index


def cache_rows():
    return read(OUT / '_CACHE/TRAIN_IMAGE_INDEX.json')['rows']


def instrument():
    from src.student.train import StudentTrainingModel, batch_loss
    from src.student.model import StudentModel
    from src.student.data import create_student_train_dataset_and_loader
    from src.dataset.teacher.datasets import CrossViewPairSampler
    from src.student.objective import PairInfoNCE
    from src.evaluation.model_loader import normalize_state
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    cfg = read(run_dir('D0',0) / 'run_config.json')
    loader = create_student_train_dataset_and_loader(SimpleNamespace(**{**cfg,'num_workers':0}))
    sampler = CrossViewPairSampler(loader.dataset,batch_size=32,shuffle=True,seed=0)
    sampler.set_epoch(1)
    batch = next(iter(DataLoader(loader.dataset,batch_sampler=sampler,num_workers=0)))
    student = StudentModel(ckpt_path=None)
    student.load_state_dict(normalize_state(torch.load(run_dir('D0',0)/'best_model.pth',
                                                     map_location='cpu',weights_only=True)),strict=True)
    student.cuda().bfloat16().train()
    module = StudentTrainingModel(student)
    class ForwardOnlyEngine:
        def __init__(self): self.module = module
        def parameters(self): return self.module.parameters()
        def __call__(self, images): return self.module(images)
    bn_events, forward_events, backbone_events, f4_events, handles = [], [], [], [], []
    for name,m in student.named_modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):
            def hook(mod, args, output, name=name):
                shape = list(args[0].shape)
                bn_events.append(dict(module=name,input_shape=shape,output_shape=list(output.shape),
                    N=shape[0],statistics_elements_per_channel=int(np.prod([shape[0]]+shape[2:]))))
            handles.append(m.register_forward_hook(hook))
    handles.append(student.register_forward_hook(lambda m,a,o:forward_events.append(dict(
        input_shape=list(a[0].shape), output_shape=list(o.shape)))))
    handles.append(student.backbone.register_forward_hook(lambda m,a,o:backbone_events.append(dict(
        backbone_input_shape=list(a[0].shape), f4_output_shape=list(o[-1].shape)))))
    handles.append(student.backbone.features[student.backbone.out_indices[-1]].register_forward_hook(
        lambda m,a,o:f4_events.append(dict(f4_block_input_shape=list(a[0].shape),f4_block_output_shape=list(o.shape)))))
    before = state_hash(student)
    buffers = {k:v.clone() for k,v in student.named_buffers()}
    images = torch.cat(batch[:2]).cuda()
    # The common formal batch_loss student call is identical before the mode branch.
    # Stop at the baseline return: no fresh training heads are created for this instrumentation.
    try:
        with torch.no_grad():
            loss,_ = batch_loss(ForwardOnlyEngine(),None,images,32,PairInfoNCE(),{'mode':'baseline'},1)
    finally:
        for h in handles: h.remove()
        with torch.no_grad():
            for k,v in student.named_buffers(): v.copy_(buffers[k])
    after = state_hash(student)
    assert before == after and len(forward_events) == 1 and all(e['N']==64 for e in bn_events)
    pair_report('ACTUAL_BN_FORWARD',dict(ACTUAL_STUDENT_FORWARD_PATTERN='one concatenated [drone32; satellite32] Student forward',
        DRONE_SAT_FORWARD_MODE='concatenated_drone_then_satellite',BN_BATCH_SIZE_UNIQUE_VALUES=sorted({e['N'] for e in bn_events}),
        FIRST_BN_INPUT_SHAPE=bn_events[0]['input_shape'],LAST_BN_INPUT_SHAPE=bn_events[-1]['input_shape'],
        forward_events=forward_events,backbone_events=backbone_events,f4_events=f4_events,bn_events=bn_events,
        sampled_identity_names=list(batch[3]),loss_finite=bool(torch.isfinite(loss)),
        parameter_dtype='bfloat16',input_dtype='bfloat16',BN_STATE_EXACT_RESTORED=True,
        STATE_HASH_BEFORE=before,STATE_HASH_AFTER=after,OPTIMIZER_STEP_CALLS=0,
        instrumentation='real train dataset + CrossViewPairSampler epoch1; exact batch_loss common Student forward; '
                        'baseline early return omits only subsequent teacher/head computations'))
    print('BN_BATCH_SIZE_UNIQUE_VALUES=[64]', flush=True)


def extract(label):
    from src.evaluation.model_loader import load_encoder
    rows = cache_rows()
    if label == 'teacher':
        model,audit = load_encoder('middle',TEACHER/'best_model.pth',TEACHER/'run_config.json','cuda:0')
    else:
        method,seed = label.split('-')
        model,audit = load_encoder('student',run_dir(method,int(seed))/'best_model.pth',device='cuda:0')
    before = state_hash(model)
    dataset = AuditImages(rows)
    loader = DataLoader(dataset,batch_size=32,shuffle=False,num_workers=6,pin_memory=True)
    chunks=[]; start=time.time()
    with torch.inference_mode():
        for step,(images,index) in enumerate(loader):
            desc=model(images.cuda(non_blocking=True)).cpu().numpy()
            assert np.isfinite(desc).all()
            chunks.append(desc)
            if step%100 == 0:
                print(json.dumps(dict(stage='extract',label=label,done=sum(len(x) for x in chunks),
                                      total=len(rows),seconds=time.time()-start)),flush=True)
    values=np.concatenate(chunks)
    assert before==state_hash(model)
    path=OUT/'_CACHE'/f'{label}_train_descriptors.npy'
    with path.open('xb') as f: np.save(f,values)
    write(f'_CACHE/{label}_train_descriptors.meta.json',dict(AUDIT_ONLY=True,NOT_FOR_TRAINING=True,
        TRAIN_ONLY=True,rows_sha256=digest(rows),descriptor_sha256=sha(path),shape=list(values.shape),
        strict_load=audit,model_state_unchanged=True,seconds=time.time()-start))
    print('EXTRACTION_COMPLETE='+label,flush=True)


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def descriptive(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('Nonfinite/empty statistics')
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std(ddof=1)) if len(x)>1 else 0.,
        median=float(np.median(x)),P90=float(np.percentile(x,90)),P95=float(np.percentile(x,95)),max=float(x.max()))


def probe_metrics(pred,target):
    pred,target=unit(pred),unit(target)
    cosine=np.clip((pred*target).sum(1),-1,1)
    sq=(pred-target)**2
    den=((target-target.mean(0))**2).sum()
    return dict(cosine_similarity_mean=float(cosine.mean()),cosine_similarity_std=float(cosine.std(ddof=1)),
        median_cosine=float(np.median(cosine)),normalized_MSE=float(sq.mean()),
        normalized_squared_L2=float(sq.sum(1).mean()),R2=float(1-sq.sum()/den) if den>0 else None,
        angular_error_deg_mean=float(np.degrees(np.arccos(cosine)).mean()),n=len(pred))


def view_weights(domains):
    domains=np.asarray(domains)
    weights=np.zeros(len(domains),dtype=np.float64)
    for domain in DOMAINS:
        mask=domains==domain
        if not mask.any(): raise ValueError('Both TRAIN domains required for probe fit')
        weights[mask]=.5/mask.sum()
    return weights


def ridge_fit(x,y,lam,weights=None):
    """Minimize weighted squared residual + lambda ||W||F^2, unpenalized intercept."""
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    w=np.ones(len(x))/len(x) if weights is None else np.asarray(weights,dtype=np.float64)
    w=w/w.sum()
    xm,ym=w@x,w@y
    xc,yc=x-xm,y-ym
    gram=xc.T@(w[:,None]*xc)
    cross=xc.T@(w[:,None]*yc)
    vals,vec=np.linalg.eigh(gram)
    if lam==0:
        cutoff=np.finfo(np.float64).eps*max(x.shape)*max(float(vals.max()),1e-30)
        inv=np.zeros_like(vals)
        np.divide(1,vals,out=inv,where=vals>cutoff)
    else: inv=1/(np.maximum(vals,0)+lam)
    coef=(vec*inv)@(vec.T@cross)
    return coef,ym-xm@coef


def fit_fresh_probe(x,y,rows,split):
    for row in rows:
        if 'path' in row: train_guard(row['path'])
    if set(split['internal_fit']) & set(split['internal_val']): raise ValueError('Internal probe leakage')
    if set(split['internal_fit']) | set(split['internal_val']) != set(split['fit']):
        raise ValueError('Internal split must exactly partition fit identities')
    ids=np.array([r['pid'] for r in rows]); domains=np.array([r['domain'] for r in rows])
    masks={k:np.isin(ids,split[k]) for k in ['fit','heldout','internal_fit','internal_val']}
    if set(split['fit'])&set(split['heldout']): raise ValueError('Probe identity leakage')
    candidate=[]
    for lam in LAMBDAS:
        m=masks['internal_fit']
        coef,bias=ridge_fit(x[m],y[m],lam,view_weights(domains[m]))
        v=masks['internal_val']
        pred=unit(x[v]@coef+bias); target=unit(y[v])
        cos=(pred*target).sum(1)
        score=float(view_weights(domains[v])@cos)
        candidate.append(dict(lambda_value=lam,internal_val_equal_view_cosine=score))
    chosen=max(range(len(candidate)),key=lambda i:candidate[i]['internal_val_equal_view_cosine'])
    lam=candidate[chosen]['lambda_value']; m=masks['fit']
    coef,bias=ridge_fit(x[m],y[m],lam,view_weights(domains[m]))
    pred=unit(x@coef+bias)
    held=masks['heldout']
    metrics={domain:probe_metrics(pred[held & (domains==domain)],y[held & (domains==domain)]) for domain in DOMAINS}
    metrics['combined']=probe_metrics(pred[held],y[held])
    metrics['view_balanced_cosine']=float(np.mean([metrics[d]['cosine_similarity_mean'] for d in DOMAINS]))
    return coef,bias,pred,dict(selected_lambda=lam,internal_selection=candidate,heldout_metrics=metrics)


def centroid(values,rows,ids,domain):
    labels=np.array([r['pid'] for r in rows]); domains=np.array([r['domain'] for r in rows])
    return unit(np.stack([values[(labels==pid)&(domains==domain)].mean(0) for pid in ids]))


def geometry_metrics(pred,teacher,within=False):
    from scipy.stats import spearmanr
    pred,teacher=np.asarray(pred,dtype=np.float64),np.asarray(teacher,dtype=np.float64)
    n=len(pred)
    mask=np.triu(np.ones_like(pred,dtype=bool),1) if within else np.ones_like(pred,dtype=bool)
    x,y=pred[mask],teacher[mask]
    pearson=float(np.corrcoef(x,y)[0,1]); spearman=float(spearmanr(x,y).statistic)
    xp,yp=pred.copy(),teacher.copy()
    if within:
        np.fill_diagonal(xp,-np.inf);np.fill_diagonal(yp,-np.inf)
    a=np.argsort(xp,axis=1)[:,::-1];b=np.argsort(yp,axis=1)[:,::-1]
    overlaps={str(k):float(np.mean([len(set(a[i,:k])&set(b[i,:k]))/k for i in range(n)])) for k in [1,5,10]}
    return dict(Pearson=pearson,Spearman=spearman,pairwise_cosine_RMSE=float(np.sqrt(np.mean((x-y)**2))),
        top_k_neighbor_overlap=overlaps,pair_count=int(mask.sum()),exclude_self=within)


def retrieval_structure(matrix):
    matrix=np.asarray(matrix,dtype=np.float64)
    pos=np.diag(matrix)
    neg=matrix.copy();np.fill_diagonal(neg,-np.inf)
    hardest=neg.max(1)
    return dict(positive_similarity=descriptive(pos),hardest_negative_similarity=descriptive(hardest),
                positive_minus_hard_negative_margin=descriptive(pos-hardest))


def numeric_delta(a,b):
    """b minus a, recursively; retain nonnumeric metadata outside the delta."""
    if isinstance(a,dict): return {k:numeric_delta(v,b[k]) for k,v in a.items() if k in b}
    if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)): return b-a
    return None


def dynamics():
    names=dict(retrieval_loss='infonce',top32_loss='top32_loss',random32_loss='random32_loss',
               dual_stst_loss='dual_stst',weighted_stst_loss='weighted_stst_loss',total_loss='loss')
    all_seeds={}
    for seed in range(3):
        records=[]
        for line in (run_dir('D0',seed)/'train.log').read_text().splitlines():
            try: row=json.loads(line)
            except (ValueError,TypeError): continue
            if 'step' in row and 'top32_loss' in row: records.append(row)
        epochs={}
        for epoch in range(1,31):
            chosen=[r for r in records if r['epoch']==epoch]
            assert chosen and len({r['step'] for r in chosen})==len(chosen)
            means={k:float(np.mean([r[v] for r in chosen])) for k,v in names.items()}
            assert all(np.isfinite(list(means.values())))
            means['random_top_ratio']=means['random32_loss']/means['top32_loss']
            epochs[str(epoch)]=dict(sample_count=len(chosen),steps=[r['step'] for r in chosen],**means)
        phases={}
        for phase,span in {'early':range(1,6),'middle':range(10,21),'late':range(25,31)}.items():
            means={k:float(np.mean([epochs[str(e)][k] for e in span])) for k in names}
            means['random_top_ratio']=means['random32_loss']/means['top32_loss']
            phases[phase]=means
        all_seeds[str(seed)]=dict(epochs=epochs,phases=phases)
    milestones={str(e):{k:descriptive([all_seeds[str(s)]['epochs'][str(e)][k] for s in range(3)])
        for k in list(names)+['random_top_ratio']} for e in [1,5,10,15,20,25,30]}
    phases={p:{k:descriptive([all_seeds[str(s)]['phases'][p][k] for s in range(3)])
        for k in list(names)+['random_top_ratio']} for p in ['early','middle','late']}
    result=dict(aggregation='logged sample means, NOT complete epoch loss means; seed std ddof=1',
        phase_epochs=dict(early=[1,5],middle=[10,20],late=[25,30]),seeds=all_seeds,milestones=milestones,phases=phases,
        TOP_BRANCH_ACTIVE=all(all_seeds[str(s)]['phases']['late']['top32_loss']>0 and
            all_seeds[str(s)]['phases']['late']['top32_loss']<all_seeds[str(s)]['phases']['early']['top32_loss'] for s in range(3)),
        RANDOM_BRANCH_ACTIVE=all(all_seeds[str(s)]['phases']['late']['random32_loss']>0 and
            all_seeds[str(s)]['phases']['late']['random32_loss']<all_seeds[str(s)]['phases']['early']['random32_loss'] for s in range(3)),
        RANDOM_LOSS_HIGHER_THAN_TOP_3OF3=all(all_seeds[str(s)]['epochs'][str(e)]['random_top_ratio']>1 for s in range(3) for e in range(1,31)),
        branch_active_definition='finite nonzero branch loss and early-to-late reduction; not gradient or causal evidence')
    pair_report('DUAL_BRANCH_DYNAMICS',result)
    return result


def probe():
    from src.student.dual_stst import DualSTSTSupervision
    rows,split=cache_rows(),read(OUT/'TRAIN_ID_SPLIT.json')
    for label in ['teacher']+[f'{m}-{s}' for m in ['B0','D0'] for s in range(3)]:
        meta=read(OUT/f'_CACHE/{label}_train_descriptors.meta.json')
        assert meta['AUDIT_ONLY'] and meta['NOT_FOR_TRAINING'] and meta['TRAIN_ONLY']
        assert meta['rows_sha256']==digest(rows)
        assert meta['descriptor_sha256']==sha(OUT/f'_CACHE/{label}_train_descriptors.npy')
    teacher=np.load(OUT/'_CACHE/teacher_train_descriptors.npy')
    # Reuse the official method directly with only its fixed buffers; do not instantiate/train a head.
    asset=torch.load(BANK,map_location='cpu',weights_only=True)
    target_owner=SimpleNamespace(**{k:asset[k].float() for k in ['teacher_mean','top32_basis','random32_basis']})
    targets=DualSTSTSupervision.teacher_targets(target_owner,torch.from_numpy(teacher))
    target_values={'top':targets[0][0].numpy(),'random':targets[1][0].numpy()}
    fit_results,geometry,global_results={},{},{}
    t_global={d:centroid(teacher,rows,split['heldout'],d) for d in DOMAINS}
    t_matrix=t_global['drone']@t_global['satellite'].T
    t_sub={b:{d:centroid(y,rows,split['heldout'],d) for d in DOMAINS} for b,y in target_values.items()}
    for seed in range(3):
        for method in ('B0','D0'):
            label=f'{method}-{seed}'; x=np.load(OUT/f'_CACHE/{label}_train_descriptors.npy')
            fit_results[label]={};geometry[label]={}; arrays={}
            for branch,y in target_values.items():
                coef,bias,pred,result=fit_fresh_probe(x,y,rows,split)
                fit_results[label][branch]=result
                arrays[branch+'_coef']=coef;arrays[branch+'_bias']=bias
                s={d:centroid(pred,rows,split['heldout'],d) for d in DOMAINS}; t=t_sub[branch]
                g={d:geometry_metrics(s[d]@s[d].T,t[d]@t[d].T,True) for d in DOMAINS}
                g['cross_view_D2S']=geometry_metrics(s['drone']@s['satellite'].T,t['drone']@t['satellite'].T)
                g['primary_mean_Pearson']=float(np.mean([g[d]['Pearson'] for d in [*DOMAINS,'cross_view_D2S']]))
                geometry[label][branch]=g
            np.savez(OUT/f'_CACHE/{label}_fresh_probe.npz',AUDIT_ONLY=True,NOT_FOR_TRAINING=True,**arrays)
            s={d:centroid(x,rows,split['heldout'],d) for d in DOMAINS}
            sm=s['drone']@s['satellite'].T
            global_results[label]={direction:dict(comparison=geometry_metrics(a,b),
                student=retrieval_structure(a),teacher=retrieval_structure(b))
                for direction,a,b in [('D2S',sm,t_matrix),('S2D',sm.T,t_matrix.T)]}
            print('PROBE_COMPLETE='+label,flush=True)
    flags={}
    for branch in ['top','random']:
        flags[branch.upper()+'_PROBE_GAIN_3OF3']=all(fit_results[f'D0-{s}'][branch]['heldout_metrics']['combined']['cosine_similarity_mean']>
             fit_results[f'B0-{s}'][branch]['heldout_metrics']['combined']['cosine_similarity_mean'] for s in range(3))
        flags[branch.upper()+'_GEOMETRY_GAIN_3OF3']=all(geometry[f'D0-{s}'][branch]['primary_mean_Pearson']>
             geometry[f'B0-{s}'][branch]['primary_mean_Pearson'] for s in range(3))
    flags['GLOBAL_GEOMETRY_GAIN_3OF3']=all(global_results[f'D0-{s}']['D2S']['comparison']['Pearson']>
        global_results[f'B0-{s}']['D2S']['comparison']['Pearson'] for s in range(3))
    pair_report('FRESH_PROBE_RESULTS',dict(TRAIN_ONLY=True,fit_ids=560,heldout_ids=141,identity_overlap=0,
        semantics=dict(TEACHER_CENTERING='descriptor - exact STST asset teacher_mean',
            TOP_TARGET_NORMALIZATION='FP32 F.normalize((descriptor-mean) @ top32_basis)',
            RANDOM_TARGET_NORMALIZATION='FP32 F.normalize((descriptor-mean) @ random32_basis)',
            STUDENT_HEAD_NORMALIZATION='FP32 linear with bias, then F.normalize',
            COSINE_LOSS_SEMANTICS='per branch 0.5*drone mean(1-cos) + 0.5*satellite mean(1-cos); branches summed',
            fresh_probe='analytical ridge approximates original normalized targets; normalized predictions; not recovered trained heads',
            normalized_MSE='mean squared error over samples and 32 coordinates after L2 normalization',
            R2='1 - summed normalized-prediction residual squares / target centered total squares'),
        models=fit_results,matched_deltas={str(s):numeric_delta(fit_results[f'B0-{s}'],fit_results[f'D0-{s}']) for s in range(3)},flags=flags))
    pair_report('TEACHER_GEOMETRY_TRANSFER',dict(TRAIN_ONLY=True,heldout_ids=141,
        identity_centroid='normalized mean per-image normalized descriptors (or projected targets/predictions) per domain/identity',
        within_view='upper triangle excludes self; nearest neighbors exclude self',
        cross_view='all 141x141 pairs including same-identity positives',subspace=geometry,global_geometry=global_results,
        subspace_matched_delta={str(s):numeric_delta(geometry[f'B0-{s}'],geometry[f'D0-{s}']) for s in range(3)},
        global_matched_delta={str(s):numeric_delta(global_results[f'B0-{s}'],global_results[f'D0-{s}']) for s in range(3)},flags=flags))
    dynamics()


def make_contexts(rows,heldout,size):
    """Real forward pattern: one concatenation, equally many matched drone/satellite pairs."""
    if size not in [16,32,64]: raise ValueError('Unsupported simulation image count')
    pairs=size//2; anchors_per_group=pairs//2
    lookup={pid:{d:[] for d in DOMAINS} for pid in heldout}
    for i,row in enumerate(rows):
        if row['pid'] in lookup: lookup[row['pid']][row['domain']].append(i)
    anchors={pid:{d:sorted(lookup[pid][d],key=lambda i:digest([SEED,'anchor',rows[i]['path']]))[0] for d in DOMAINS} for pid in heldout}
    result=[]
    for start in range(0,len(heldout),anchors_per_group):
        group=heldout[start:start+anchors_per_group]
        candidates=[pid for pid in heldout if pid not in group]
        for k in range(8):
            rng=random.Random(SEED+size*100003+start*101+k)
            companions=rng.sample(candidates,pairs-len(group))
            pair_ids=group+companions
            d=[anchors[pid]['drone'] for pid in group]+[rng.choice(lookup[pid]['drone']) for pid in companions]
            s=[anchors[pid]['satellite'] for pid in group]+[lookup[pid]['satellite'][0] for pid in companions]
            result.append(dict(size=size,group_start=start,k=k,anchor_ids=group,pair_ids=pair_ids,
                               row_indices=d+s,anchor_positions=list(range(len(group)))+list(range(pairs,pairs+len(group)))))
    return anchors,result


def drift_arrays(context,reference):
    context,reference=unit(context),unit(reference)
    cosine=np.clip((context*reference[:,None,:]).sum(-1),-1,1)
    pairwise=np.einsum('akd,ald->akl',context,context)
    upper=np.triu_indices(context.shape[1],1)
    return dict(cos_context_eval=cosine,one_minus_cosine=1-cosine,
        descriptor_L2_drift=np.linalg.norm(context-reference[:,None,:],axis=-1),
        pairwise_context_cosine=np.clip(pairwise[:,upper[0],upper[1]],-1,1))


def summarize_drift(context,reference):
    n=len(reference)//2
    output={}
    for domain,sl in [('drone',slice(0,n)),('satellite',slice(n,None)),('combined',slice(None))]:
        arrays=drift_arrays(context[sl],reference[sl])
        output[domain]={k:descriptive(v) for k,v in arrays.items()}
        output[domain]['per_anchor_mean_one_minus_cosine']=descriptive(arrays['one_minus_cosine'].mean(1))
    return output


def bn_audit(label):
    from functools import lru_cache
    from src.evaluation.model_loader import load_encoder
    rows,split=cache_rows(),read(OUT/'TRAIN_ID_SPLIT.json')
    actual=read(OUT/'ACTUAL_BN_FORWARD.json')
    assert actual['BN_BATCH_SIZE_UNIQUE_VALUES']==[64]
    if label=='teacher':
        wrapper,audit=load_encoder('middle',TEACHER/'best_model.pth',TEACHER/'run_config.json','cuda:0')
    else:
        method,seed=label.split('-')
        wrapper,audit=load_encoder('student',run_dir(method,int(seed))/'best_model.pth',device='cuda:0')
    model=wrapper.model
    before=state_hash(model)
    dataset=AuditImages(rows)
    @lru_cache(maxsize=512)
    def get_image(index): return dataset[index][0]
    def forward(images):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            return model(images).float().cpu().numpy()
    result={}; n=len(split['heldout']); dim=768 if label=='teacher' else 512
    for size in [64,16,32]:
        anchors,contexts=make_contexts(rows,split['heldout'],size)
        context=np.zeros((2*n,8,dim),np.float32); reference=np.zeros((2*n,dim),np.float32)
        eval_control=np.zeros_like(reference); start=time.time()
        for step,c in enumerate(contexts):
            images=torch.stack([get_image(i) for i in c['row_indices']]).cuda()
            start_id=c['group_start']; count=len(c['anchor_ids'])
            out_indices=list(range(start_id,start_id+count))+list(range(n+start_id,n+start_id+count))
            if c['k']==0:
                model.eval();reference[out_indices]=forward(images)[c['anchor_positions']]
            if c['k']==7:
                model.eval();eval_control[out_indices]=forward(images)[c['anchor_positions']]
            if label=='teacher':
                model.eval(); values=forward(images)
            else:
                with restored_bn(model):
                    assert all(not m.training for m in model.modules() if not isinstance(m,nn.modules.batchnorm._BatchNorm))
                    values=forward(images)
            context[out_indices,c['k']]=values[c['anchor_positions']]
            if step%50==0:
                print(json.dumps(dict(stage='bn',label=label,N=size,done=step+1,total=len(contexts),seconds=time.time()-start)),flush=True)
        assert np.isfinite(context).all() and np.all(np.linalg.norm(context,axis=-1)>.99)
        assert before==state_hash(model)
        np.savez(OUT/f'_CACHE/{label}_BN_N{size}.npz',AUDIT_ONLY=True,NOT_FOR_TRAINING=True,
                 context=context,reference=reference,eval_control=eval_control)
        metrics=summarize_drift(context,reference)
        metrics['eval_only_context_control']=descriptive(1-np.clip((unit(reference)*unit(eval_control)).sum(1),-1,1))
        metrics['SIMULATION_ONLY']=size!=64
        metrics['total_images']=size;metrics['pair_batch']=size//2
        metrics['BN_STATE_EXACT_RESTORED']=True
        if label!='teacher':
            probe_weights=np.load(OUT/f'_CACHE/{label}_fresh_probe.npz')
            metrics['knowledge']={}
            for branch in ['top','random']:
                coef,bias=probe_weights[branch+'_coef'],probe_weights[branch+'_bias']
                metrics['knowledge'][branch]=summarize_drift(context@coef+bias,reference@coef+bias)
        result[str(size)]=metrics
        write(f'_CACHE/{label}_bn_partial.json',result)
        if label=='teacher':
            write(f'_CACHE/BN_CONTEXT_N{size}_MANIFEST.json',dict(AUDIT_ONLY=True,NOT_FOR_TRAINING=True,
                anchors=anchors,contexts=contexts,sha256=digest(contexts)))
    pair_report(f'BN_{label}',dict(label=label,strict_load=audit,results=result,BN_STATE_EXACT_RESTORED=True,
        STATE_HASH_BEFORE=before,STATE_HASH_AFTER=state_hash(model),non_BN_modules_eval=True,OPTIMIZER_STEP_CALLS=0))
    print('BN_COMPLETE='+label,flush=True)


def validate_final_schema(result):
    required=['AUDIT_SOURCE_COMMIT','TRAIN_ONLY','FIT_IDS','HELDOUT_IDS','IDENTITY_OVERLAP',
        'ACTUAL_STUDENT_FORWARD_PATTERN','BN_BATCH_SIZE_UNIQUE_VALUES','TOP_PROBE_GAIN_3OF3',
        'RANDOM_PROBE_GAIN_3OF3','D0_REDUCES_BN_DRIFT_3OF3','OPTIMIZER_STEP_CALLS',
        'TRAINING_CHECKPOINT_CREATED','CHECKPOINTS_UNCHANGED','STST_ASSET_UNCHANGED',
        'FORMAL_TRAINING_STARTED','AUDIT_COMPLETE','FACT','SUPPORTED_INTERPRETATION','UNRESOLVED']
    if any(k not in result for k in required): raise ValueError('Incomplete final audit schema')
    if not (result['TRAIN_ONLY'] and result['FIT_IDS']==560 and result['HELDOUT_IDS']==141 and
            result['IDENTITY_OVERLAP']==0 and result['OPTIMIZER_STEP_CALLS']==0 and
            result['TRAINING_CHECKPOINT_CREATED']==0 and result['CHECKPOINTS_UNCHANGED'] and
            result['STST_ASSET_UNCHANGED'] and not result['FORMAL_TRAINING_STARTED']):
        raise ValueError('Audit safety gate failed')


def finalize():
    config=read(OUT/'AUDIT_CONFIG.json'); split=read(OUT/'TRAIN_ID_SPLIT.json')
    forward=read(OUT/'ACTUAL_BN_FORWARD.json')
    probes=read(OUT/'FRESH_PROBE_RESULTS.json'); geo=read(OUT/'TEACHER_GEOMETRY_TRANSFER.json')
    dyn=read(OUT/'DUAL_BRANCH_DYNAMICS.json'); heads=read(OUT/'HEAD_PERSISTENCE.json')
    labels=[f'{method}-{s}' for method in ['B0','D0'] for s in range(3)]
    bn={label:read(OUT/f'BN_{label}.json') for label in [*labels,'teacher']}
    assert all(v['BN_STATE_EXACT_RESTORED'] for v in bn.values())
    assert not any(c['TRAINED_TOP_HEAD_AVAILABLE'] or c['TRAINED_RANDOM_HEAD_AVAILABLE']
        for seed in heads.values() for c in seed.values()), 'Available trained heads require additional evaluation'
    after={p:dict(sha256=sha(p),size=Path(p).stat().st_size) for p in config['protected_files']}
    assert after==config['protected_files'], 'Protected source assets changed'
    created=list(OUT.rglob('*.pth'))+list(OUT.rglob('*.pt'))
    assert not created, 'Audit directory must contain no training checkpoint or STST bank'
    git_clean=not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert git_clean, 'Finalize only after tested audit source commit and clean status'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tests=read(OUT/'REGRESSION_TEST_STATUS.json')
    assert tests['pytest_exit_code']==0 and tests['git_diff_check_exit_code']==0
    result=dict(AUDIT_SOURCE_COMMIT=commit,TRAIN_ONLY=True,FIT_IDS=560,HELDOUT_IDS=141,IDENTITY_OVERLAP=0,
        ACTUAL_STUDENT_FORWARD_PATTERN=forward['ACTUAL_STUDENT_FORWARD_PATTERN'],
        BN_BATCH_SIZE_UNIQUE_VALUES=forward['BN_BATCH_SIZE_UNIQUE_VALUES'],
        TOP_BRANCH_ACTIVE=dyn['TOP_BRANCH_ACTIVE'],RANDOM_BRANCH_ACTIVE=dyn['RANDOM_BRANCH_ACTIVE'],
        RANDOM_LOSS_HIGHER_THAN_TOP_3OF3=dyn['RANDOM_LOSS_HIGHER_THAN_TOP_3OF3'],
        TRAINED_HEAD_EVAL_SKIPPED=True,BRANCH_GRADIENT_AUDIT_SKIPPED=True,
        SKIP_REASON='All six D0 best/last checkpoints contain only deployment Student tensors; no trained heads',
        OPTIMIZER_STEP_CALLS=0,SCHEDULER_STEP_CALLS=0,TRAINING_CHECKPOINT_CREATED=0,
        CHECKPOINTS_UNCHANGED=True,STST_ASSET_UNCHANGED=True,FORMAL_TRAINING_STARTED=False,
        BN_STATE_EXACT_RESTORED=True,REGRESSION_TESTS_PASS=True,GIT_DIFF_CHECK_PASS=True,
        GIT_STATUS_CLEAN=git_clean,**probes['flags'])
    result.update({k:forward[k] for k in ['DRONE_SAT_FORWARD_MODE','FIRST_BN_INPUT_SHAPE','LAST_BN_INPUT_SHAPE']})
    result.update({k:v for k,v in probes['semantics'].items() if k.isupper()})
    for branch in ['top','random']:
        for method in ['B0','D0']:
            result[f'{branch.upper()}_PROBE_{method}_MEAN']=float(np.mean([
                probes['models'][f'{method}-{s}'][branch]['heldout_metrics']['combined']['cosine_similarity_mean'] for s in range(3)]))
            result[f'{branch.upper()}_GEOMETRY_{method}']=float(np.mean([
                geo['subspace'][f'{method}-{s}'][branch]['primary_mean_Pearson'] for s in range(3)]))
    for method in ['B0','D0']:
        result[f'GLOBAL_GEOMETRY_{method}']=float(np.mean([
            geo['global_geometry'][f'{method}-{s}']['D2S']['comparison']['Pearson'] for s in range(3)]))
        result[f'BN_DRIFT_{method}_MEAN']=float(np.mean([
            bn[f'{method}-{s}']['results']['64']['combined']['one_minus_cosine']['mean'] for s in range(3)]))
        result[f'BN_SENSITIVITY_PRESENT_{method}']=all(
            bn[f'{method}-{s}']['results']['64']['combined']['one_minus_cosine']['mean']>1e-6 for s in range(3))
        for branch in ['top','random']:
            result[f'{branch.upper()}_KNOWLEDGE_DRIFT_{method}']=float(np.mean([
                bn[f'{method}-{s}']['results']['64']['knowledge'][branch]['combined']['one_minus_cosine']['mean'] for s in range(3)]))
    result['D0_REDUCES_BN_DRIFT_3OF3']=all(bn[f'D0-{s}']['results']['64']['combined']['one_minus_cosine']['mean']<
        bn[f'B0-{s}']['results']['64']['combined']['one_minus_cosine']['mean'] for s in range(3))
    for seed in range(3):
        for method in ['B0','D0']:
            result[f'BN_DRIFT_{method}_S{seed}']=bn[f'{method}-{seed}']['results']['64']['combined']['one_minus_cosine']['mean']
        result[f'BN_DRIFT_DELTA_S{seed}']=result[f'BN_DRIFT_D0_S{seed}']-result[f'BN_DRIFT_B0_S{seed}']
    for branch in ['top','random']:
        result[f'D0_REDUCES_{branch.upper()}_KNOWLEDGE_DRIFT_3OF3']=all(
            bn[f'D0-{s}']['results']['64']['knowledge'][branch]['combined']['one_minus_cosine']['mean']<
            bn[f'B0-{s}']['results']['64']['knowledge'][branch]['combined']['one_minus_cosine']['mean'] for s in range(3))
    result['TEACHER_CONTEXT_DRIFT_MEAN']=bn['teacher']['results']['64']['combined']['one_minus_cosine']['mean']
    result['TEACHER_CONTEXT_DRIFT_P95']=bn['teacher']['results']['64']['combined']['one_minus_cosine']['P95']
    for phase in ['early','middle','late']:
        result[f'TOP_LOSS_{phase.upper()}']=dyn['phases'][phase]['top32_loss']['mean']
        result[f'RANDOM_LOSS_{phase.upper()}']=dyn['phases'][phase]['random32_loss']['mean']
        result[f'RANDOM_TOP_RATIO_{phase.upper()}']=dyn['phases'][phase]['random_top_ratio']['mean']
    result['FACT']=[
        'All mechanism measurements use the fixed 141 heldout TRAIN identities; probes use 560 other TRAIN identities.',
        'Instrumented formal Student forward concatenates drone32 and satellite32; all BN layers see N=64.',
        f"Heldout Top cosine B0={result['TOP_PROBE_B0_MEAN']:.8f}, D0={result['TOP_PROBE_D0_MEAN']:.8f}; "
        f"3/3 gain={result['TOP_PROBE_GAIN_3OF3']}.",
        f"Heldout Random cosine B0={result['RANDOM_PROBE_B0_MEAN']:.8f}, D0={result['RANDOM_PROBE_D0_MEAN']:.8f}; "
        f"3/3 gain={result['RANDOM_PROBE_GAIN_3OF3']}.",
        f"BN context mean drift B0={result['BN_DRIFT_B0_MEAN']:.8f}, D0={result['BN_DRIFT_D0_MEAN']:.8f}; "
        f"3/3 decrease={result['D0_REDUCES_BN_DRIFT_3OF3']}.",
        'Training-only heads are absent; trained-head evaluation and branch gradient diagnostics were skipped.']
    result['SUPPORTED_INTERPRETATION']=[
        'A higher heldout fresh-probe score indicates more linearly recoverable fixed Teacher subspace information '
        'in the frozen Student descriptor under this probe protocol; it does not isolate either training branch causally.',
        ('Teacher-guided Dual-STST is associated with lower BN-context representation sensitivity under the matched '
         'single-GPU protocol.' if result['D0_REDUCES_BN_DRIFT_3OF3'] else
         'This audit does not establish a consistent three-seed reduction in BN-context representation sensitivity.'),
        'Higher Random residual is consistent with a harder target fit; loss magnitudes alone do not measure gradient contribution.']
    result['UNRESOLVED']=config['limitations']+[
        'No causal claim that BN causes the D0 retrieval gain; no branch ablation or controlled BN training intervention was performed.',
        'Canonical BN contexts use fixed anchor groups and varying matched companions, with deterministic preprocessing, '
        'Student FP32 storage/BF16 autocast, and only BN train mode. They isolate BN sensitivity, not all augmented BF16-storage training effects.',
        'Within-context anchors share companions; repeated contexts are correlated. Seed and sample descriptive statistics are not significance tests.',
        'No recovered trained heads: the fresh ridge probes are analytical measurements, not the original learned projectors.']
    result['AUDIT_COMPLETE']=True
    validate_final_schema(result)
    pair_report('BN_CONTEXT_SENSITIVITY',dict(TRAIN_ONLY=True,heldout_ids=141,anchors_per_domain=141,K=8,
        primary_N=64,simulation_N=[16,32],models=bn,
        matched_deltas={str(s):numeric_delta(bn[f'B0-{s}']['results'],bn[f'D0-{s}']['results']) for s in range(3)},
        flags={k:v for k,v in result.items() if 'DRIFT' in k or 'SENSITIVITY' in k},BN_STATE_EXACT_RESTORED=True))
    pair_report('D0_KNOWLEDGE_ABSORPTION_AUDIT',result)
    write('STATE_IMMUTABILITY.json',dict(before=config['protected_files'],after=after,all_unchanged=True,
        checkpoints_created=0,optimizer_step_calls=0,scheduler_step_calls=0))
    config.update(AUDIT_SOURCE_COMMIT=commit,AUDIT_COMPLETE=True)
    write('AUDIT_CONFIG.json',config)
    (OUT/'SOURCE_PROVENANCE.txt').write_text(json.dumps(dict(config=config,
        source_sha256=sha(Path(__file__)),tests_sha256=sha(ROOT/'tests/test_d0_knowledge_absorption_audit.py'),
        all_assets_unchanged=True),indent=2)+'\n')
    report=['# D0 Knowledge Absorption Audit v1','',f'Audit source commit: `{commit}`','',
        'TRAIN-only；560 identities 拟合、141 identities 报告；没有训练或 optimizer/scheduler step。','',
        '## 核心结果','', '| 指标 | B0 三 seed 均值 | D0 三 seed 均值 | 3/3 改善 |',
        '|---|---:|---:|---|']
    for title,left,right,flag in [
        ('Top probe cosine','TOP_PROBE_B0_MEAN','TOP_PROBE_D0_MEAN','TOP_PROBE_GAIN_3OF3'),
        ('Random probe cosine','RANDOM_PROBE_B0_MEAN','RANDOM_PROBE_D0_MEAN','RANDOM_PROBE_GAIN_3OF3'),
        ('Top geometry Pearson','TOP_GEOMETRY_B0','TOP_GEOMETRY_D0','TOP_GEOMETRY_GAIN_3OF3'),
        ('Random geometry Pearson','RANDOM_GEOMETRY_B0','RANDOM_GEOMETRY_D0','RANDOM_GEOMETRY_GAIN_3OF3'),
        ('Global geometry Pearson','GLOBAL_GEOMETRY_B0','GLOBAL_GEOMETRY_D0','GLOBAL_GEOMETRY_GAIN_3OF3'),
        ('BN drift (1-cos) ↓','BN_DRIFT_B0_MEAN','BN_DRIFT_D0_MEAN','D0_REDUCES_BN_DRIFT_3OF3'),
        ('Top knowledge drift ↓','TOP_KNOWLEDGE_DRIFT_B0','TOP_KNOWLEDGE_DRIFT_D0','D0_REDUCES_TOP_KNOWLEDGE_DRIFT_3OF3'),
        ('Random knowledge drift ↓','RANDOM_KNOWLEDGE_DRIFT_B0','RANDOM_KNOWLEDGE_DRIFT_D0','D0_REDUCES_RANDOM_KNOWLEDGE_DRIFT_3OF3')]:
        report.append(f'| {title} | {result[left]:.8f} | {result[right]:.8f} | {result[flag]} |')
    report+=['','## 按 seed 的核心测量','',
        '| Seed | Model | Top cosine | Random cosine | Global Pearson | BN drift N64 |',
        '|---|---|---:|---:|---:|---:|']
    for s in range(3):
        for m in ['B0','D0']:
            label=f'{m}-{s}'
            vals=[probes['models'][label][b]['heldout_metrics']['combined']['cosine_similarity_mean'] for b in ['top','random']]
            vals+=[geo['global_geometry'][label]['D2S']['comparison']['Pearson'],bn[label]['results']['64']['combined']['one_minus_cosine']['mean']]
            report.append('| '+f'{s} | {m} | '+' | '.join(f'{v:.8f}' for v in vals)+' |')
    report+=['',f"Teacher context drift: mean={result['TEACHER_CONTEXT_DRIFT_MEAN']:.10g}, P95={result['TEACHER_CONTEXT_DRIFT_P95']:.10g}.",'']
    for section in ['FACT','SUPPORTED_INTERPRETATION','UNRESOLVED']:
        report+=['## '+section,'']+['- '+x for x in result[section]]+['']
    report+=['## 结果文件','',
        '- `FRESH_PROBE_RESULTS.json/.txt`：三域全部 probe 指标、内部 lambda 选择和匹配 delta。',
        '- `TEACHER_GEOMETRY_TRANSFER.json/.txt`：Top/Random 子空间、直接全局几何、邻居重合、正负 margin。',
        '- `BN_CONTEXT_SENSITIVITY.json/.txt`：三种 N、所有域、完整分布统计和知识 drift。',
        '- `DUAL_BRANCH_DYNAMICS.json/.txt`：日志采样逐 epoch/阶段统计，early=1–5、middle=10–20、late=25–30。',
        '- `_CACHE/`：AUDIT_ONLY=True / NOT_FOR_TRAINING=True，包含图像顺序和 context manifest。',
        '- `STATE_IMMUTABILITY.json`：原始文件前后 SHA；`HEAD_PERSISTENCE.json`：best/last 实际 keys。','',
        '未启动任何新的正式实验。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    lines=['========== D0 KNOWLEDGE ABSORPTION AUDIT v1 ==========']
    for k,v in result.items():
        if k not in ['FACT','SUPPORTED_INTERPRETATION','UNRESOLVED']: lines.append(f'{k}={v}')
    (OUT/'FINAL_OUTPUT.txt').write_text('\n'.join(lines)+'\n')
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='RESULT_MANIFEST.txt')
    (OUT/'RESULT_MANIFEST.txt').write_text('AUDIT_ONLY=True\nNOT_FOR_TRAINING=True\n'+
        '\n'.join(f'{sha(p)}  {p.stat().st_size}  {p.relative_to(OUT)}' for p in files)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','instrument','extract','probe','bn','finalize'])
    parser.add_argument('--label',default='teacher')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    if args.stage=='prepare': prepare()
    elif args.stage=='instrument': instrument()
    elif args.stage=='extract': extract(args.label)
    elif args.stage=='probe': probe()
    elif args.stage=='bn': bn_audit(args.label)
    elif args.stage=='finalize': finalize()


if __name__=='__main__': main()
