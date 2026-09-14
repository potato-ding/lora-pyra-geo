"""TRAIN-only Part-I.5 measurements; immutable checkpoints, no optimization updates."""
from __future__ import annotations
import argparse,hashlib,json,random,subprocess,tarfile,time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from src.tools import audit_d0_knowledge_absorption as old
from src.tools.audit_t128_r32_complementarity import split_guard,score_prediction
from src.student.dual_stst import DualSTSTSupervision
from src.student.part1 import load_extended_asset
from src.student.artifacts import source_identity
ROOT=old.ROOT;STUDENT=old.STUDENT;TRAIN=old.TRAIN
OUT=STUDENT/'_AUDITS/P1_5_T128_R32_MECHANISM_V1'
PREVIOUS=old.OUT
TEACHER=old.TEACHER
ASSET=STUDENT/'STST_ASSETS/SAM-MABV2-RHO010-S0_train_extended_p1_v1.pt'
SEED=20260914
LAMBDAS=old.LAMBDAS
sha=old.sha;read=old.read;digest=old.digest;train_guard=old.train_guard
def run(seed):return STUDENT/f'P1-T128-R32-S{seed}'
def write(name,data):
    path=OUT/name
    if not path.resolve().is_relative_to(OUT.resolve()):raise ValueError('Audit output only')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def pair(name,data):
    write(name+'.json',data)
    (OUT/(name+'.txt')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def snapshot(paths):return {str(p):dict(size=p.stat().st_size,sha256=sha(p)) for p in paths}
def unchanged():
    before=read(OUT/'AUDIT_CONFIG.json')['protected_files']
    assert snapshot([Path(p) for p in before])==before,'Protected assets changed'
def stats(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Empty/nonfinite distribution')
    med=float(np.median(x));q=np.percentile(x,[5,10,25,75,90,95])
    variance=float(x.var(ddof=1)) if len(x)>1 else 0.
    return dict(n=len(x),mean=float(x.mean()),std=variance**.5,variance=variance,median=med,
        P05=float(q[0]),P10=float(q[1]),P25=float(q[2]),P75=float(q[3]),P90=float(q[4]),P95=float(q[5]),
        IQR=float(q[3]-q[2]),MAD=float(np.median(np.abs(x-med))))
def angles(cos):return np.degrees(np.arccos(np.clip(cos,-1,1)))
def ratio(a,b):return float(a/b) if b!=0 else None
def gradient_arrays(gt,gr):
    gt=np.asarray(gt,dtype=np.float64);gr=np.asarray(gr,dtype=np.float64)
    nt=np.linalg.norm(gt,axis=-1);nr=np.linalg.norm(gr,axis=-1)
    if np.any(nt==0) or np.any(nr==0):raise ValueError('Zero gradient requires explicit undefined statistics')
    cosine=np.clip(np.sum(gt*gr,axis=-1)/(nt*nr),-1,1)
    return dict(top_norm=nt,random_norm=nr,random_top_ratio=nr/nt,grad_cos=cosine,
        combined_to_sum_ratio=np.linalg.norm(gt+gr,axis=-1)/(nt+nr))
def summarize_grad(a):
    out={k:stats(v) for k,v in a.items()}
    c=a['grad_cos']
    out['fractions']=dict(negative=float(np.mean(c<0)),below_minus025=float(np.mean(c<-.25)),
        near_zero_abs_le01=float(np.mean(np.abs(c)<=.1)),positive=float(np.mean(c>0)),above025=float(np.mean(c>.25)))
    return out
@contextmanager
def restored_state(model,training):
    modes={m:m.training for m in model.modules()}
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    model.train(training)
    try:yield
    finally:
        with torch.no_grad():
            for n,b in model.named_buffers():b.copy_(buffers[n])
        for m,value in modes.items():m.training=value
        assert all(torch.equal(b,buffers[n]) for n,b in model.named_buffers())
def official_targets(z,asset):
    holder=SimpleNamespace(teacher_mean=asset['teacher_mean'].to(z.device).float(),
        top32_basis=asset['top128_basis'].to(z.device).float(),random32_basis=asset['random32_A'].to(z.device).float())
    t,r=DualSTSTSupervision.teacher_targets(holder,z)
    return t[0],r[0]
def tensor_inventory(payload,prefix=''):
    result={}
    if torch.is_tensor(payload):result[prefix]=dict(shape=list(payload.shape),dtype=str(payload.dtype))
    elif isinstance(payload,dict):
        for k,v in payload.items():result.update(tensor_inventory(v,prefix+'/'+str(k)))
    elif isinstance(payload,(list,tuple)):
        for i,v in enumerate(payload):result.update(tensor_inventory(v,prefix+'/'+str(i)))
    return result
def make_batches(rows,split):
    split_guard(rows,split)
    lookup={pid:{d:[] for d in old.DOMAINS} for pid in split['heldout']}
    for i,row in enumerate(rows):
        if row['pid'] in lookup:lookup[row['pid']][row['domain']].append(i)
    rng=random.Random(SEED);batches=[]
    for b in range(32):
        ids=rng.sample(sorted(lookup),32)
        indices=[rng.choice(lookup[pid][d]) for d in old.DOMAINS for pid in ids]
        batches.append(dict(batch=b,identity_ids=ids,row_indices=indices,paths=[rows[i]['path'] for i in indices],
            audit_rng_seed=SEED+b,augmentation=False,augmentation_rng_seed=None))
    return dict(seed=SEED,N_BATCHES=32,pair_batch=32,total_images=64,
        data='heldout TRAIN only',preprocessing='old.AuditImages deterministic get_test_transforms([224,224])',
        same_all_seeds=True,batches=batches,sha256=digest(batches))
def make_relations(rows,held,z):
    lookup={pid:{d:[] for d in old.DOMAINS} for pid in held}
    for i,row in enumerate(rows):
        if row['pid'] in lookup:lookup[row['pid']][row['domain']].append(i)
    ids=sorted(held);dq=[];dg=[];sat=[]
    for pid in ids:
        drones=sorted(lookup[pid]['drone'],key=lambda i:digest([SEED,'relation',rows[i]['path']]))
        assert len(drones)>=2 and len(lookup[pid]['satellite'])==1
        dq.append(drones[0]);dg.append(drones[1]);sat.append(lookup[pid]['satellite'][0])
    output={}
    for domain,queries,gallery in [('D2S',dq,sat),('DRONE_DRONE',dq,dg),('SAT_SAT',sat,sat)]:
        sim=old.unit(z[queries])@old.unit(z[gallery]).T
        groups={'positive':[],'hard_negative_top1':[],'hard_negative_top5':[],'random_negative':[]}
        rng=random.Random(SEED)
        for i,q in enumerate(queries):
            if domain!='SAT_SAT':groups['positive'].append([q,gallery[i]])
            rank=np.argsort(sim[i])[::-1];neg=[int(j) for j in rank if j!=i]
            groups['hard_negative_top1'].append([q,gallery[neg[0]]])
            groups['hard_negative_top5'].extend([[q,gallery[j]] for j in neg[:5]])
            j=rng.choice([j for j in range(len(ids)) if j!=i])
            groups['random_negative'].append([q,gallery[j]])
        output[domain]=groups
    return dict(heldout_ids=ids,query_identities=141,anchor_selection='SHA256(seed,relation,path); first drone as query, second drone as within-drone gallery',
        mining='full768 normalized Teacher deterministic TRAIN descriptors only; argsort descending; identity excluded',
        SAT_SAT_positive='not defined: only one satellite image per identity; self-pairs excluded',
        pairs=output,sha256=digest(output),row_index_source=str(PREVIOUS/'_CACHE/TRAIN_IMAGE_INDEX.json'))
def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    splitpath=PREVIOUS/'TRAIN_ID_SPLIT.json'
    if not splitpath.is_file():raise FileNotFoundError('Original split required; no regeneration')
    split=read(splitpath);index=read(PREVIOUS/'_CACHE/TRAIN_IMAGE_INDEX.json');rows=index['rows']
    split_guard(rows,split)
    assert len(split['fit'])==560 and len(split['heldout'])==141
    assert all(train_guard(x['path']).is_file() for x in rows)
    cache=PREVIOUS/'_CACHE/teacher_train_descriptors.npy';meta=read(cache.with_suffix('.meta.json'))
    tsha=sha(TEACHER/'best_model.pth')
    assert meta['strict_load']['sha256']==tsha and meta['descriptor_sha256']==sha(cache)
    assert meta['TRAIN_ONLY'] and meta['model_state_unchanged']
    assert digest(rows)==index['rows_sha256']==meta['rows_sha256']
    asset=load_extended_asset(ASSET,old.BANK,tsha)
    protected=[TEACHER/'best_model.pth',TEACHER/'run_config.json',ASSET,old.BANK,splitpath,cache,cache.with_suffix('.meta.json'),PREVIOUS/'_CACHE/TRAIN_IMAGE_INDEX.json']
    recovery={}
    for seed in range(3):
        cfg=read(run(seed)/'run_config.json')
        assert cfg['top_dim']==128 and cfg['random_layout']=='single32' and cfg['random_total_dim']==32
        assert cfg['middle_teacher_sha256']==tsha and cfg['extended_asset_sha256']==sha(ASSET)
        files=sorted(p for p in run(seed).rglob('*') if p.is_file());protected+=files
        rec=dict(files=[dict(path=str(p),size=p.stat().st_size) for p in files],state_files={},archives={})
        for p in files:
            if p.suffix in ['.pt','.pth','.bin','.ckpt']:
                data=torch.load(p,map_location='cpu',weights_only=True)
                inv=tensor_inventory(data)
                matching=[k for k in inv if any(t in k.lower() for t in ['projector_top','projector_random','top_head','random_head','stst'])]
                rec['state_files'][p.name]=dict(epoch=data.get('epoch'),all_tensors=inv,head_matching_keys=matching)
            elif p.name.endswith('.tar.gz'):
                with tarfile.open(p) as tar:rec['archives'][p.name]=tar.getnames()
        rec['TRAINED_TOP_HEAD_RECOVERABLE']=any('top' in k for f in rec['state_files'].values() for k in f['head_matching_keys'])
        rec['TRAINED_RANDOM_HEAD_RECOVERABLE']=any('random' in k for f in rec['state_files'].values() for k in f['head_matching_keys'])
        assert not rec['TRAINED_TOP_HEAD_RECOVERABLE'] and not rec['TRAINED_RANDOM_HEAD_RECOVERABLE'],'Recovered head requires trained-head implementation'
        assert not any(n.endswith(('.pth','.pt','.ckpt','.bin')) for names in rec['archives'].values() for n in names),'Inspect archived state before fallback'
        recovery[str(seed)]=rec
    OUT.mkdir(parents=True);(OUT/'_CACHE').mkdir()
    heads=dict(HEAD_SOURCE='RIDGE_DIAGNOSTIC',ORIGINAL_TRAINING_GRADIENTS_RECOVERABLE=False,seeds=recovery,
        schema='canonical_selection.canonical_state -> bare deployment_state_dict -> epoch/model/protocol_id; no training-only projector serialization',
        schema_source={n:sha(ROOT/n) for n in ['src/student/canonical_selection.py','src/student/artifacts.py','src/student/dual_stst.py']})
    pair('HEAD_RECOVERY_AUDIT',heads)
    config=dict(PART1_BASE_INTERFACE='T128+R32',TRAIN_ONLY=True,ZERO_TRAINING=True,HEAD_SOURCE='RIDGE_DIAGNOSTIC',
        ORIGINAL_TRAINING_GRADIENTS_RECOVERABLE=False,TRAIN_SPLIT_REUSED=True,FIT_ID_COUNT=560,HELDOUT_ID_COUNT=141,ID_OVERLAP=0,
        split_path=str(splitpath),split_sha256=sha(splitpath),rows_path=str(PREVIOUS/'_CACHE/TRAIN_IMAGE_INDEX.json'),
        teacher_cache=str(cache),teacher_cache_sha256=sha(cache),lambda_candidates=LAMBDAS,N_BATCHES=32,audit_seed=SEED,
        extraction='old.AuditImages; deterministic224; batch32; Student FP32 storage CUDA BF16 autocast; same as D0 Knowledge Absorption',
        gradient='actual StudentTrainingModel.forward output -> batch_loss descriptor.float() -> shared post-BN/post-L2 FP32 [64,512]; independent autograd.grad at this exact interface',
        context='Full Student train() versus eval(); BF16 parameters/input, native forward; every buffer and module mode restored after each context; frozen ridge coefficients FP32, FP32 loss. No backbone backward needed for dz.',
        precision_scope='Ridge fitted to legacy FP32-storage/autocast eval descriptors; gradient contexts both use formal BF16 storage. Fixed-head transfer between these numerical paths is a diagnostic limitation, not original training gradient recovery.',
        fit='FP64 centered ridge, unregularized intercept, equal total weight per view; internal448/112 FIT selection by equal-view cosine; refit560; held141 never fitted',
        gradient_reduction='Official _branch_loss: half each view mean = mean over64. Per-sample gradients are rows of mean-loss derivative, without multiplying64. Batch norm ratio is ||gR||F/||gT||F; primary ratio averages32 batch ratios; primary cos/conflict pool2048 sample rows.',
        summary_scope='Primary angular fields use D2S hard_negative_top1 only (no mixture of positives and negatives). Margin uses D2S positive vs hard_negative_top1. All domains/sets reported separately. Changes=train minus eval. Std is sample ddof1.',
        limitations=['Ridge diagnostic heads are not trained heads; original training gradients unavailable.',
        'All701TRAIN identities previously informed Teacher/Student/bank; heldout isolates ridge fit only.',
        'Deterministic preprocessing without augmentation isolates context with matched images; no weighting recommendation or method decision.',
        'Repeated satellite/images and batches are dependent; descriptive distributions are not independent-sample significance tests.',
        'Teacher distribution duplicated across seeds is the same fixed source, not independent Teacher estimates.'],
        protected_files=snapshot(protected),training_source_sha256=source_identity(),
        initial_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write('AUDIT_CONFIG.json',config)
    write('AUDIT_BATCH_MANIFEST.json',make_batches(rows,split))
    z=np.load(cache)
    write('RELATION_PAIR_MANIFEST.json',make_relations(rows,split['heldout'],z))
    (OUT/'SOURCE_PROVENANCE.txt').write_text(json.dumps(config,indent=2)+'\n')
    print('HEAD_SOURCE=RIDGE_DIAGNOSTIC; ORIGINAL_TRAINING_GRADIENTS_RECOVERABLE=False; TRAIN_SPLIT_REUSED=True',flush=True)
def extract(seed):
    from src.evaluation.model_loader import load_encoder
    cfg=read(OUT/'AUDIT_CONFIG.json');rows=read(cfg['rows_path'])['rows']
    wrapper,identity=load_encoder('student',run(seed)/'best_model.pth',device='cuda:0')
    before=old.state_hash(wrapper);parts=[]
    loader=DataLoader(old.AuditImages(rows),batch_size=32,shuffle=False,num_workers=6,pin_memory=True)
    with torch.inference_mode():
        for i,(images,indices) in enumerate(loader):
            parts.append(wrapper(images.cuda(non_blocking=True)).cpu().numpy())
            if i%200==0:print(f'S{seed} EXTRACT {min((i+1)*32,len(rows))}/{len(rows)}',flush=True)
    z=np.concatenate(parts);assert z.shape==(len(rows),512) and np.isfinite(z).all()
    assert before==old.state_hash(wrapper)
    path=OUT/f'_CACHE/student_S{seed}.npy'
    with path.open('xb') as f:np.save(f,z)
    write(f'_CACHE/student_S{seed}.meta.json',dict(strict_load=identity,sha256=sha(path),rows_sha256=digest(rows),
        state_before=before,state_after=old.state_hash(wrapper),STATE_EXACT=True))
    return z,wrapper
def conditioning(x,w,lam):
    x=np.asarray(x,dtype=np.float64);w=w/w.sum();x=x-w@x
    eigen=np.linalg.eigvalsh(x.T@(w[:,None]*x))
    tol=np.finfo(np.float64).eps*max(x.shape)*max(float(eigen[-1]),1e-30)
    pos=eigen[eigen>tol]
    return dict(gram_min_eigenvalue=float(eigen[0]),gram_max_eigenvalue=float(eigen[-1]),numerical_rank=len(pos),
        full_rank=len(pos)==x.shape[1],condition_on_nonzero_subspace=ratio(eigen[-1],pos[0]) if len(pos) else None,
        regularized_condition=ratio(max(eigen[-1],0)+lam,max(eigen[0],0)+lam),svd_cutoff=tol)
def fit_heads(x,targets,rows,split):
    split_guard(rows,split)
    ids=np.array([r['pid'] for r in rows]);domains=np.array([r['domain'] for r in rows])
    masks={k:np.isin(ids,split[k]) for k in ['fit','heldout','internal_fit','internal_val']}
    coef={};report={}
    for branch,target in targets.items():
        candidates=[]
        for lam in LAMBDAS:
            m=masks['internal_fit'];w=old.view_weights(domains[m]);c,b=old.ridge_fit(x[m],target[m],lam,w)
            v=masks['internal_val']
            score=float(old.view_weights(domains[v])@np.sum(old.unit(x[v]@c+b)*old.unit(target[v]),axis=1))
            candidates.append(dict(lam=lam,internal_equal_view_cosine=score))
        lam=max(candidates,key=lambda z:z['internal_equal_view_cosine'])['lam']
        m=masks['fit'];w=old.view_weights(domains[m]);c,b=old.ridge_fit(x[m],target[m],lam,w)
        coef[branch]=(c,b);held={}
        for domain in ['combined','drone','satellite']:
            h=masks['heldout'] if domain=='combined' else masks['heldout']&(domains==domain)
            held[domain]=score_prediction(x[h]@c+b,target[h])
        report[branch]=dict(lambda_value=lam,conditioning=conditioning(x[m],w,lam),candidates=candidates,
            heldout=held,FIT_ID_COUNT=560,HELDOUT_ID_COUNT=141,ID_OVERLAP=0,
            metric_note='Primary prediction L2 normalized; R2=1-SSE/SST;NMSE=SSE/sum(target^2);raw prediction alternatives also reported')
    return coef,report
def require_diagnostic_heads(source,heads):
    if source!='RIDGE_DIAGNOSTIC' or set(heads)!=set(['top','random']):raise ValueError('Random/untrained heads forbidden')
    for name,dim in [('top',128),('random',32)]:
        c,b=heads[name]
        if c.shape!=(512,dim) or b.shape!=(dim,) or not np.isfinite(c).all() or not np.isfinite(b).all():
            raise ValueError('Invalid fitted head')
def branch_gradients(z,t,r,heads):
    losses=[];grads=[]
    for name,target in [('top',t),('random',r)]:
        c,b=heads[name]
        prediction=F.normalize(z.float()@c+b,dim=1)
        loss=DualSTSTSupervision._branch_loss(prediction,target,z.shape[0]//2)[0]
        g=torch.autograd.grad(loss,z,retain_graph=True)[0]
        assert torch.isfinite(g).all() and torch.isfinite(loss)
        losses.append(float(loss));grads.append(g.detach().float().cpu().numpy())
    return losses,grads
def gradients(seed,model,heads,asset):
    from src.evaluation.model_loader import load_encoder
    from src.student.train import StudentTrainingModel
    require_diagnostic_heads('RIDGE_DIAGNOSTIC',heads)
    cfg=read(OUT/'AUDIT_CONFIG.json');rows=read(cfg['rows_path'])['rows']
    manifest=read(OUT/'AUDIT_BATCH_MANIFEST.json');dataset=old.AuditImages(rows)
    teacher,ta=load_encoder('middle',TEACHER/'best_model.pth',TEACHER/'run_config.json','cuda:0')
    tmodel=teacher.model;assert not tmodel.training and not any(p.requires_grad for p in tmodel.parameters())
    teacher_before=old.state_hash(tmodel)
    model.bfloat16().eval();initial=old.state_hash(model)
    # Backbone is a frozen value provider; requiring dz at its exact forward
    # boundary yields the same partial derivative without parameter gradients.
    wrapped=StudentTrainingModel(model)
    for p in model.parameters():p.requires_grad_(False)
    th={name:(torch.tensor(c,dtype=torch.float32,device='cuda'),torch.tensor(b,dtype=torch.float32,device='cuda'))
        for name,(c,b) in heads.items()}
    contexts={k:[] for k in ['train','eval']};raw={k:[] for k in contexts};input_hashes=[];bn_shapes=[]
    handles=[]
    for n,m in model.named_modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
            handles.append(m.register_forward_hook(lambda m,i,o,n=n:bn_shapes.append(dict(name=n,N=int(i[0].shape[0]),training=m.training))))
    try:
        for batch in manifest['batches']:
            random.seed(batch['audit_rng_seed']);np.random.seed(batch['audit_rng_seed']);torch.manual_seed(batch['audit_rng_seed'])
            images=torch.stack([dataset[i][0] for i in batch['row_indices']]).cuda()
            input_hashes.append(hashlib.sha256(images.cpu().numpy().tobytes()).hexdigest())
            assert images.shape==(64,3,224,224)
            with torch.no_grad():
                teacher_z=tmodel(images.bfloat16()).detach().float()
                t,r=official_targets(teacher_z,asset)
            for context in ['train','eval']:
                with restored_state(model,context=='train'):
                    with torch.no_grad():z=wrapped(images.bfloat16()).float()
                    assert z.dtype==torch.float32 and z.shape==(64,512)
                    z.requires_grad_(True)
                    losses,(gt,gr)=branch_gradients(z,t,r,th)
                    a=gradient_arrays(gt,gr);raw[context].append(a)
                    nt=float(np.linalg.norm(gt.astype(np.float64)));nr=float(np.linalg.norm(gr.astype(np.float64)))
                    record=dict(batch=batch['batch'],TOP_LOSS=losses[0],RANDOM_LOSS=losses[1],
                        TOP_GRAD_NORM=nt,RANDOM_GRAD_NORM=nr,G_R_OVER_G_T=nr/nt,
                        G_SUM_NORM=float(np.linalg.norm(gt.astype(np.float64)+gr)),
                        COMBINED_TO_SUM_RATIO=float(np.linalg.norm(gt.astype(np.float64)+gr)/(nt+nr)),
                        BATCH_GRAD_COS=float(np.sum(gt.astype(np.float64)*gr)/(nt*nr)),
                        per_sample={k:v.tolist() for k,v in a.items()})
                    contexts[context].append(record)
                assert all(p.grad is None for p in model.parameters())
            if batch['batch']%8==0:print(f'S{seed} GRAD_BATCH={batch["batch"]+1}/32',flush=True)
    finally:
        for h in handles:h.remove()
    assert initial==old.state_hash(model) and teacher_before==old.state_hash(tmodel)
    assert all(e['N']==64 for e in bn_shapes)
    summary={};stack={}
    for context in contexts:
        stack[context]={k:np.stack([a[k] for a in raw[context]]) for k in raw[context][0]}
        summary[context]={}
        for domain,sl in [('ALL',slice(None)),('DRONE',slice(0,32)),('SATELLITE',slice(32,64))]:
            summary[context][domain]=summarize_grad({k:v[:,sl].reshape(-1) for k,v in stack[context].items()})
        summary[context]['batch']={key:stats([b[key] for b in contexts[context]]) for key in [
            'TOP_LOSS','RANDOM_LOSS','TOP_GRAD_NORM','RANDOM_GRAD_NORM','G_R_OVER_G_T','G_SUM_NORM','COMBINED_TO_SUM_RATIO','BATCH_GRAD_COS']}
    diff={}
    for domain,sl in [('ALL',slice(None)),('DRONE',slice(0,32)),('SATELLITE',slice(32,64))]:
        diff[domain]={k:stats((stack['train'][k]-stack['eval'][k])[:,sl]) for k in stack['train']}
    output=dict(seed=seed,HEAD_SOURCE='RIDGE_DIAGNOSTIC',ORIGINAL_TRAINING_GRADIENTS_RECOVERABLE=False,
        contexts=contexts,summary=summary,TRAIN_EVAL_CONTEXT_GRADIENT_DIFF=diff,
        VIEW_RATIO_DIFFERENCE={c:summary[c]['DRONE']['random_top_ratio']['mean']-summary[c]['SATELLITE']['random_top_ratio']['mean'] for c in contexts},
        view_difference_definition='drone minus satellite mean per-sample ratio',
        BN_STATE_RESTORED=True,STATE_BEFORE=initial,STATE_AFTER=old.state_hash(model),
        TEACHER_STATE_BEFORE=teacher_before,TEACHER_STATE_AFTER=old.state_hash(tmodel),teacher_grad_count=0,
        shared_representation='StudentTrainingModel(images) / formal batch_loss descriptor.float(): post BN and post L2',
        shared_interface_source_sha256=sha(ROOT/'src/student/train.py'),bn_events=bn_shapes,
        matched_input_sha256=input_hashes,batch_manifest_sha256=sha(OUT/'AUDIT_BATCH_MANIFEST.json'),
        OPTIMIZER_STEP_CALLS=0,SCHEDULER_STEP_CALLS=0,TRAINING_STARTED=False)
    write(f'GRADIENT_S{seed}.json',output)
def relation_measure(target,pairs):
    out={}
    for domain,sets in pairs.items():
        out[domain]={}
        for name,indices in sets.items():
            if not indices:out[domain][name]=None;continue
            index=np.asarray(indices)
            cosine=np.clip(np.sum(old.unit(target[index[:,0]])*old.unit(target[index[:,1]]),axis=1),-1,1)
            out[domain][name]=dict(cosine=cosine,angle=angles(cosine))
    return out
def angular(seed,x,heads,targets):
    manifest=read(OUT/'RELATION_PAIR_MANIFEST.json');result={}
    for branch in ['top','random']:
        c,b=heads[branch];student=old.unit(x.astype(np.float64)@c+b)
        tm=relation_measure(targets[branch],manifest['pairs']);sm=relation_measure(student,manifest['pairs'])
        domains={}
        for domain,sets in tm.items():
            distributions={};margins={}
            for name,t in sets.items():
                if t is None:distributions[name]=None;continue
                v=sm[domain][name];entry={}
                for metric in ['cosine','angle']:
                    ts,ss=stats(t[metric]),stats(v[metric])
                    entry[metric]=dict(teacher=ts,student=ss,
                        student_minus_teacher_mean=ss['mean']-ts['mean'],student_over_teacher_std=ratio(ss['std'],ts['std']),
                        student_over_teacher_IQR=ratio(ss['IQR'],ts['IQR']),
                        Wasserstein_1D=float(wasserstein_distance(t[metric],v[metric])))
                distributions[name]=entry
            for negative in ['hard_negative_top1','hard_negative_top5','random_negative']:
                if sets['positive'] is None:margins[negative]=None;continue
                margins[negative]={}
                for who,m in [('teacher',tm),('student',sm)]:
                    p=m[domain]['positive'];n=m[domain][negative]
                    margins[negative][who]=dict(margin_cos=float(p['cosine'].mean()-n['cosine'].mean()),
                        margin_angle=float(n['angle'].mean()-p['angle'].mean()))
                margins[negative]['student_minus_teacher']={k:margins[negative]['student'][k]-margins[negative]['teacher'][k] for k in ['margin_cos','margin_angle']}
            domains[domain]=dict(distributions=distributions,margins=margins)
        result[branch]=domains
    ratios={}
    for domain in manifest['pairs']:
        ratios[domain]={}
        for name in manifest['pairs'][domain]:
            t=result['top'][domain]['distributions'][name];v=result['random'][domain]['distributions'][name]
            ratios[domain][name]=None if t is None else {
                who:dict(R32_OVER_TOP128_ANGLE_STD_RATIO=ratio(v['angle'][who]['std'],t['angle'][who]['std']),
                    R32_OVER_TOP128_ANGLE_IQR_RATIO=ratio(v['angle'][who]['IQR'],t['angle'][who]['IQR']))
                for who in ['teacher','student']}
    write(f'ANGULAR_SCALE_S{seed}.json',dict(seed=seed,HEAD_SOURCE='RIDGE_DIAGNOSTIC',branches=result,
        branch_dispersion_ratios=ratios,relation_manifest_sha256=sha(OUT/'RELATION_PAIR_MANIFEST.json'),
        scope='All relations fixed across branches/seeds; 141 heldout TRAIN identities; no branch-specific mining'))
def worker(seed):
    unchanged();cfg=read(OUT/'AUDIT_CONFIG.json')
    assert read(OUT/'HEAD_RECOVERY_AUDIT.json')['HEAD_SOURCE']=='RIDGE_DIAGNOSTIC'
    rows=read(cfg['rows_path'])['rows'];split=read(cfg['split_path'])
    asset=load_extended_asset(ASSET,old.BANK,sha(TEACHER/'best_model.pth'))
    teacher_z=np.load(cfg['teacher_cache'])
    t,r=official_targets(torch.from_numpy(teacher_z),asset)
    targets=dict(top=t.numpy(),random=r.numpy())
    x,wrapper=extract(seed)
    heads,probe=fit_heads(x,targets,rows,split)
    require_diagnostic_heads('RIDGE_DIAGNOSTIC',heads)
    write(f'_CACHE/RIDGE_S{seed}.json',probe)
    with (OUT/f'_CACHE/RIDGE_S{seed}.npz').open('xb') as f:
        np.savez(f,**{k+'_'+n:v for k,(c,b) in heads.items() for n,v in [('coef',c),('bias',b)]})
    print(f'S{seed} RIDGE_COMPLETE',flush=True)
    angular(seed,x,heads,targets)
    gradients(seed,wrapper.model,heads,asset)
    unchanged()
    print(f'S{seed} AUDIT_COMPLETE=True',flush=True)
def cross(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(S0=float(a[0]),S1=float(a[1]),S2=float(a[2]),mean=float(a.mean()),sample_std=float(a.std(ddof=1)))
def flatten_numeric(x,prefix=''):
    out={}
    if isinstance(x,dict):
        for k,v in x.items():out.update(flatten_numeric(v,prefix+'/'+str(k)))
    elif isinstance(x,(int,float)) and not isinstance(x,bool):out[prefix]=float(x)
    return out
def finalize():
    unchanged()
    cfg=read(OUT/'AUDIT_CONFIG.json');g=[read(OUT/f'GRADIENT_S{s}.json') for s in range(3)]
    a=[read(OUT/f'ANGULAR_SCALE_S{s}.json') for s in range(3)]
    assert all(v['BN_STATE_RESTORED'] and v['STATE_BEFORE']==v['STATE_AFTER'] for v in g)
    assert g[0]['matched_input_sha256']==g[1]['matched_input_sha256']==g[2]['matched_input_sha256']
    assert len(g[0]['matched_input_sha256'])==32
    assert len({x['relation_manifest_sha256'] for x in a})==1
    assert source_identity()==cfg['training_source_sha256']
    tests=read(OUT/'_CACHE/TEST_STATUS.json');assert tests['PASS']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    keys={}
    for name,fun in {
        'TOP_GRAD_NORM':lambda x:x['summary']['train']['batch']['TOP_GRAD_NORM']['mean'],
        'RANDOM_GRAD_NORM':lambda x:x['summary']['train']['batch']['RANDOM_GRAD_NORM']['mean'],
        'RANDOM_TOP_GRAD_RATIO':lambda x:x['summary']['train']['batch']['G_R_OVER_G_T']['mean'],
        'GRAD_COS_MEAN':lambda x:x['summary']['train']['ALL']['grad_cos']['mean'],
        'GRAD_CONFLICT_FRACTION':lambda x:x['summary']['train']['ALL']['fractions']['negative'],
        'DRONE_RANDOM_TOP_GRAD_RATIO':lambda x:x['summary']['train']['DRONE']['random_top_ratio']['mean'],
        'SAT_RANDOM_TOP_GRAD_RATIO':lambda x:x['summary']['train']['SATELLITE']['random_top_ratio']['mean'],
        'DRONE_GRAD_COS':lambda x:x['summary']['train']['DRONE']['grad_cos']['mean'],
        'SAT_GRAD_COS':lambda x:x['summary']['train']['SATELLITE']['grad_cos']['mean'],
        'TRAIN_EVAL_TOP_GRAD_CHANGE':lambda x:x['TRAIN_EVAL_CONTEXT_GRADIENT_DIFF']['ALL']['top_norm']['mean'],
        'TRAIN_EVAL_RANDOM_GRAD_CHANGE':lambda x:x['TRAIN_EVAL_CONTEXT_GRADIENT_DIFF']['ALL']['random_norm']['mean'],
        'TRAIN_EVAL_GRAD_COS_CHANGE':lambda x:x['TRAIN_EVAL_CONTEXT_GRADIENT_DIFF']['ALL']['grad_cos']['mean']}.items():
        keys[name]=cross([fun(x) for x in g])
    for branch,label in [('top','TOP128'),('random','R32')]:
        keys[label+'_TEACHER_ANGLE_STD']=cross([x['branches'][branch]['D2S']['distributions']['hard_negative_top1']['angle']['teacher']['std'] for x in a])
        for metric in ['cos','angle']:
            keys[label+'_RELATION_MARGIN_'+metric.upper()]=cross([x['branches'][branch]['D2S']['margins']['hard_negative_top1']['teacher']['margin_'+metric] for x in a])
        keys[label+'_STUDENT_TEACHER_WASSERSTEIN']=cross([x['branches'][branch]['D2S']['distributions']['hard_negative_top1']['angle']['Wasserstein_1D'] for x in a])
    keys['R32_TOP128_ANGLE_STD_RATIO']=cross([x['branch_dispersion_ratios']['D2S']['hard_negative_top1']['teacher']['R32_OVER_TOP128_ANGLE_STD_RATIO'] for x in a])
    consistency=dict(RANDOM_GRAD_NORM_GT_TOP_SEEDS=sum(keys['RANDOM_GRAD_NORM'][f'S{i}']>keys['TOP_GRAD_NORM'][f'S{i}'] for i in range(3)),
        GRAD_COS_POSITIVE_MEAN_SEEDS=sum(keys['GRAD_COS_MEAN'][f'S{i}']>0 for i in range(3)),
        SAT_RATIO_GT_DRONE_SEEDS=sum(keys['SAT_RANDOM_TOP_GRAD_RATIO'][f'S{i}']>keys['DRONE_RANDOM_TOP_GRAD_RATIO'][f'S{i}'] for i in range(3)),
        R32_ANGLE_STD_GT_TOP_SEEDS=sum(keys['R32_TEACHER_ANGLE_STD'][f'S{i}']>keys['TOP128_TEACHER_ANGLE_STD'][f'S{i}'] for i in range(3)))
    details={}
    for label,objects in [('gradient_summary',[x['summary'] for x in g]),('context_diff',[x['TRAIN_EVAL_CONTEXT_GRADIENT_DIFF'] for x in g]),
                          ('angular',[x['branches'] for x in a]),('angular_ratios',[x['branch_dispersion_ratios'] for x in a])]:
        flats=[flatten_numeric(x) for x in objects]
        assert all(set(f)==set(flats[0]) for f in flats)
        details[label]={k:cross([f[k] for f in flats]) for k in flats[0]}
    summary=dict(HEAD_SOURCE='RIDGE_DIAGNOSTIC',ORIGINAL_TRAINING_GRADIENTS_RECOVERABLE=False,
        key_measurements=keys,all_numeric_measurements=details,consistency=consistency,
        ridge={str(s):read(OUT/f'_CACHE/RIDGE_S{s}.json') for s in range(3)},
        scope=cfg['summary_scope'],gradient_reduction=cfg['gradient_reduction'],limitations=cfg['limitations'],
        BN_STATE_RESTORED=True,CHECKPOINTS_UNCHANGED=True,ASSETS_UNCHANGED=True,
        AUDIT_SOURCE_COMMIT=commit,TESTS_PASS=True,GIT_DIFF_CHECK_PASS=True,OPTIMIZER_STEP_CALLS=0,
        TRAINING_STARTED=False,TRAIN_ONLY=True,ZERO_TRAINING=True,AUDIT_COMPLETE=True)
    write('P1_5_MECHANISM_3SEED_SUMMARY.json',summary)
    lines=['========== PART-I.5 T128+R32 MECHANISM AUDIT ==========',
        'PART1_BASE_INTERFACE=T128+R32','TRAIN_ONLY=True','ZERO_TRAINING=True',
        'HEAD_SOURCE=RIDGE_DIAGNOSTIC','ORIGINAL_TRAINING_GRADIENTS_RECOVERABLE=False']
    for suffix,key in [('RANDOM_TOP_GRAD_RATIO','RANDOM_TOP_GRAD_RATIO'),('GRAD_COS_MEAN','GRAD_COS_MEAN'),('CONFLICT_FRACTION','GRAD_CONFLICT_FRACTION')]:
        for i in range(3):lines.append(f'S{i}_{suffix}={keys[key][f"S{i}"]}')
    for out,key,field in [
        ('MEAN_RANDOM_TOP_GRAD_RATIO','RANDOM_TOP_GRAD_RATIO','mean'),('STD_RANDOM_TOP_GRAD_RATIO','RANDOM_TOP_GRAD_RATIO','sample_std'),
        ('MEAN_GRAD_COS','GRAD_COS_MEAN','mean'),('STD_GRAD_COS','GRAD_COS_MEAN','sample_std'),
        ('DRONE_RANDOM_TOP_GRAD_RATIO_MEAN','DRONE_RANDOM_TOP_GRAD_RATIO','mean'),('SAT_RANDOM_TOP_GRAD_RATIO_MEAN','SAT_RANDOM_TOP_GRAD_RATIO','mean'),
        ('DRONE_GRAD_COS_MEAN','DRONE_GRAD_COS','mean'),('SAT_GRAD_COS_MEAN','SAT_GRAD_COS','mean'),
        ('TOP128_ANGLE_STD_MEAN','TOP128_TEACHER_ANGLE_STD','mean'),('R32_ANGLE_STD_MEAN','R32_TEACHER_ANGLE_STD','mean'),
        ('R32_OVER_TOP128_ANGLE_STD_RATIO','R32_TOP128_ANGLE_STD_RATIO','mean'),
        ('TOP128_MARGIN_COS','TOP128_RELATION_MARGIN_COS','mean'),('R32_MARGIN_COS','R32_RELATION_MARGIN_COS','mean'),
        ('TOP128_MARGIN_ANGLE','TOP128_RELATION_MARGIN_ANGLE','mean'),('R32_MARGIN_ANGLE','R32_RELATION_MARGIN_ANGLE','mean'),
        ('TOP128_TEACHER_STUDENT_WASSERSTEIN','TOP128_STUDENT_TEACHER_WASSERSTEIN','mean'),('R32_TEACHER_STUDENT_WASSERSTEIN','R32_STUDENT_TEACHER_WASSERSTEIN','mean')]:
        lines.append(f'{out}={keys[key][field]}')
    for k in ['TRAIN_EVAL_TOP_GRAD_CHANGE','TRAIN_EVAL_RANDOM_GRAD_CHANGE','TRAIN_EVAL_GRAD_COS_CHANGE']:lines.append(f'{k}={keys[k]["mean"]}')
    lines += [f'{k}={v}' for k,v in consistency.items()]
    archive=OUT/'P1_5_T128_R32_MECHANISM_V1_RESULTS.tar.gz'
    lines+=['OPTIMIZER_STEP_CALLS=0','TRAINING_STARTED=False','CHECKPOINTS_UNCHANGED=True','ASSETS_UNCHANGED=True',
        'BN_STATE_RESTORED=True','TESTS_PASS=True','GIT_DIFF_CHECK_PASS=True',f'AUDIT_SOURCE_COMMIT={commit}',f'RESULT_PACKAGE={archive}','AUDIT_COMPLETE=True']
    terminal='\n'.join(lines)+'\n'
    txt=terminal+'\nMEASUREMENT_SCOPE\n'+cfg['summary_scope']+'\n'+cfg['gradient_reduction']+'\n'+cfg['precision_scope']+'\n'
    txt+='\nLIMITATIONS\n'+'\n'.join(cfg['limitations'])+'\n\nALL KEY MEASUREMENTS: S0/S1/S2/mean/sample_std\n'+json.dumps(keys,indent=2)+'\n'
    (OUT/'P1_5_MECHANISM_3SEED_SUMMARY.txt').write_text(txt)
    provenance=dict(config=cfg,source_commit=commit,audit_source_sha256=sha(Path(__file__)),tests=tests,
        training_source_unchanged=True,protected_after=snapshot([Path(p) for p in cfg['protected_files']]))
    (OUT/'SOURCE_PROVENANCE.txt').write_text(json.dumps(provenance,indent=2)+'\n')
    logs=[OUT/'_CACHE/prepare.log']+[OUT/f'_CACHE/worker_S{s}.log' for s in range(3)]+[OUT/'_CACHE/pytest.log']
    (OUT/'audit.log').write_text('\n'.join(f'FILE={p.name}\n'+p.read_text(errors='replace') for p in logs)+'\n'+terminal)
    names=['AUDIT_CONFIG.json','SOURCE_PROVENANCE.txt','HEAD_RECOVERY_AUDIT.json','HEAD_RECOVERY_AUDIT.txt']+[
        f'{label}_S{s}.json' for label in ['GRADIENT','ANGULAR_SCALE'] for s in range(3)]+[
        'P1_5_MECHANISM_3SEED_SUMMARY.json','P1_5_MECHANISM_3SEED_SUMMARY.txt','AUDIT_BATCH_MANIFEST.json','RELATION_PAIR_MANIFEST.json','audit.log','RESULT_MANIFEST.txt']
    (OUT/'RESULT_MANIFEST.txt').write_text('AUDIT_SOURCE_COMMIT='+commit+'\n'+
        '\n'.join(f'{sha(OUT/n)}  {(OUT/n).stat().st_size}  {n}' for n in names if n!='RESULT_MANIFEST.txt')+'\n')
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as tar:
            for n in names:tar.add(OUT/n,arcname=n,recursive=False)
    with tarfile.open(archive) as tar:
        assert set(tar.getnames())==set(names) and len(tar.getmembers())==16
        for member in tar.getmembers():
            assert member.isfile() and not member.name.endswith(('.pth','.pt'))
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==sha(OUT/member.name)
    print(terminal,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','finalize']);p.add_argument('--seed',type=int,choices=range(3))
    args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    if args.stage=='prepare':prepare()
    elif args.stage=='worker':worker(args.seed)
    else:finalize()

