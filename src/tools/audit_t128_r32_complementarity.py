"""TRAIN-only immutable T128/R32 geometry and closed-form probe audit."""
from __future__ import annotations
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.stats import spearmanr
from src.tools import audit_d0_knowledge_absorption as legacy
from src.student.dual_stst import DualSTSTSupervision,load_stst_asset
from src.student.part1 import load_extended_asset,check_extended_tensors
ROOT=Path(__file__).resolve().parents[2]
STUDENT=ROOT/'src/checkpoint/student/CERTIFIED_R224'
OUT=STUDENT/'_AUDITS/T128_R32_COMPLEMENTARITY_V1'
PREVIOUS=STUDENT/'_AUDITS/D0_KNOWLEDGE_ABSORPTION_V1'
TRAIN=ROOT/'data/U1652/train'
TEACHER=ROOT/'src/checkpoint/middle_teacher/CERTIFIED_R224/SAM-MABV2-RHO010-S0/best_model.pth'
ORIGINAL=STUDENT/'STST_ASSETS/SAM-MABV2-RHO010-S0_train_shared_k32.pt'
EXTENDED=STUDENT/'STST_ASSETS/SAM-MABV2-RHO010-S0_train_extended_p1_v1.pt'
SEED=20260914
N_NULL=2000
LAMBDAS=[0.,1e-8,1e-6,1e-4,1e-2,1.]
def sha(p):return legacy.sha(p)
def read(p):return json.loads(Path(p).read_text())
def write(name,data):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def pair(name,data):
    write(name+'.json',data)
    (OUT/(name+'.txt')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def train_guard(path,root=TRAIN):
    p=Path(path).resolve();root=Path(root).resolve()
    if not p.is_relative_to(root):raise ValueError('Only University-1652 TRAIN is permitted')
    return p
def snapshot(paths):return {str(p):dict(size=p.stat().st_size,sha256=sha(p)) for p in paths}
def unchanged(protected):
    if snapshot([Path(p) for p in protected])!=protected:raise RuntimeError('Protected state mutation')
def basis_guard(b,dim=None):
    b=np.asarray(b,dtype=np.float64)
    if b.ndim!=2 or (dim is not None and b.shape!=dim) or not np.isfinite(b).all():raise ValueError('Basis shape/finite error')
    error=float(np.max(np.abs(b.T@b-np.eye(b.shape[1]))))
    if error>1e-5:raise ValueError('Basis is not orthonormal')
    return error
def overlap(u,r):
    basis_guard(u);basis_guard(r)
    if u.shape[0]!=r.shape[0]:raise ValueError('Ambient dimensions differ')
    g=u.T@r;s=np.linalg.svd(g,compute_uv=False)
    angles=np.degrees(np.arccos(np.clip(s,-1,1)))
    rho=float(np.sum(g*g)/r.shape[1])
    if rho < -1e-6 or rho>1+1e-6:raise ValueError('Invalid overlap')
    return dict(frobenius_norm=float(np.linalg.norm(g)),spectral_norm=float(s.max()),
      max_absolute_entry=float(np.abs(g).max()),rho=rho,residual_fraction=1-rho,
      singular_values=s.tolist(),principal_angles_deg=angles.tolist(),
      angle_statistics=dict(min=float(angles.min()),P10=float(np.percentile(angles,10)),
      median=float(np.median(angles)),mean=float(angles.mean()),P90=float(np.percentile(angles,90)),max=float(angles.max())),
      max_canonical_correlation=float(s.max()),mean_squared_canonical_correlation=float(np.mean(s*s)))
def random_null(u32,u64,u128,n=N_NULL,seed=SEED,conditional=True):
    basis_guard(u32);basis_guard(u64);basis_guard(u128)
    u0=np.linalg.qr(u32,mode='reduced')[0]
    rng=np.random.default_rng(seed);values=[];max_orth=0.;max_cross=0.
    for i in range(n):
        gaussian=rng.standard_normal((u32.shape[0],32))
        if conditional:gaussian-=u0@(u0.T@gaussian)
        q=np.linalg.qr(gaussian,mode='reduced')[0]
        max_orth=max(max_orth,float(np.abs(q.T@q-np.eye(32)).max()))
        if conditional:max_cross=max(max_cross,float(np.abs(u32.T@q).max()))
        values.append([np.sum((u64.T@q)**2)/32,np.sum((u128.T@q)**2)/32])
    return np.asarray(values),dict(max_orthogonality_error=max_orth,max_U32_cross=max_cross if conditional else None,n=n,seed=seed,
         conditional=conditional,generator='NumPy PCG64 standard_normal, float64 project then reduced QR')
def null_summary(values,observed):
    return dict(mean=float(values.mean()),std=float(values.std(ddof=1)),
       P5=float(np.percentile(values,5)),P50=float(np.percentile(values,50)),
       P95=float(np.percentile(values,95)),P99=float(np.percentile(values,99)),
       observed=observed,percentile_rank=float(100*(np.sum(values<observed)+.5*np.sum(values==observed))/len(values)),
       empirical_two_sided_p=float(min(1,2*min((1+np.sum(values<=observed))/(len(values)+1),(1+np.sum(values>=observed))/(len(values)+1)))))
def residualize(u,r):
    basis_guard(u);basis_guard(r)
    raw=r-u@(u.T@r)
    q,upper=np.linalg.qr(raw,mode='reduced')
    sign=np.where(np.diag(upper)<0,-1.,1.);q=q*sign
    left,_,right=np.linalg.svd(q.T@r,full_matrices=False)
    aligned=q@(left@right)
    return raw,q,aligned
def official_targets(z,mean,u,r):
    # Invoke the production FP32 centering/projection/L2 method without a Student or training head.
    holder=SimpleNamespace(teacher_mean=mean.float(),top32_basis=torch.as_tensor(u,dtype=torch.float32),
                           random32_basis=torch.as_tensor(r,dtype=torch.float32))
    with torch.no_grad():
        top,rnd=DualSTSTSupervision.teacher_targets(holder,torch.as_tensor(z,dtype=torch.float32))
    return top[0].numpy(),rnd[0].numpy()
def geometry(targets,rows,ids):
    d=legacy.centroid(targets,rows,ids,'drone');s=legacy.centroid(targets,rows,ids,'satellite')
    return {'drone':d@d.T,'satellite':s@s.T,'D2S':d@s.T}
def compare_geometry(a,b):
    out={}
    for domain in ['drone','satellite','D2S']:
        m=legacy.geometry_metrics(a[domain],b[domain],within=domain!='D2S')
        m['unique_neighbor_fraction']={k:1-v for k,v in m['top_k_neighbor_overlap'].items()}
        if domain=='D2S':
            # Primary cross-view includes same-identity matches, also report mismatch-only geometry.
            mask=~np.eye(len(a[domain]),dtype=bool);x=a[domain][mask];y=b[domain][mask]
            aa=a[domain].copy();bb=b[domain].copy()
            np.fill_diagonal(aa,-np.inf);np.fill_diagonal(bb,-np.inf)
            ranks=[np.argsort(v,axis=1)[:,::-1] for v in [aa,bb]]
            m['exclude_same_identity_sensitivity']=dict(Pearson=float(np.corrcoef(x,y)[0,1]),
              Spearman=float(spearmanr(x,y).statistic),pairwise_cosine_RMSE=float(np.sqrt(np.mean((x-y)**2))),
              top_k_neighbor_overlap={str(k):float(np.mean([len(set(ranks[0][i,:k])&set(ranks[1][i,:k]))/k
                for i in range(len(aa))])) for k in [1,5,10]})
        out[domain]=m
    return out
def split_guard(rows,split):
    for row in rows:train_guard(row['path'])
    fit,held=set(split['fit']),set(split['heldout'])
    if fit&held or set(split['internal_fit'])&set(split['internal_val']):raise ValueError('Identity leakage')
    if set(split['internal_fit'])|set(split['internal_val'])!=fit:raise ValueError('Internal fit split invalid')
    if set(r['pid'] for r in rows)!=fit|held:raise ValueError('Rows do not cover the exact split')
    return fit,held
def score_prediction(pred,target):
    p=legacy.unit(pred);t=legacy.unit(target)
    cos=np.clip(np.sum(p*t,axis=1),-1,1)
    centered=float(np.sum((t-t.mean(0))**2))
    sq=float(np.sum((p-t)**2));rawsq=float(np.sum((pred-t)**2))
    return dict(n=len(t),cosine_similarity=float(cos.mean()),cosine_std=float(cos.std(ddof=1)),
       angular_error_deg=float(np.degrees(np.arccos(cos)).mean()),R2=1-sq/centered,
       NMSE=sq/float(np.sum(t*t)),per_coordinate_normalized_MSE=sq/t.size,
       raw_prediction_R2=1-rawsq/centered,raw_prediction_NMSE=rawsq/float(np.sum(t*t)))
def ridge_probe(x,y,rows,split):
    split_guard(rows,split)
    ids=np.array([r['pid'] for r in rows]);domains=np.array([r['domain'] for r in rows])
    masks={k:np.isin(ids,split[k]) for k in ['fit','heldout','internal_fit','internal_val']}
    candidates=[]
    for lam in LAMBDAS:
        m=masks['internal_fit'];coef,bias=legacy.ridge_fit(x[m],y[m],lam,legacy.view_weights(domains[m]))
        v=masks['internal_val'];pred=legacy.unit(x[v]@coef+bias);t=legacy.unit(y[v])
        candidates.append(dict(lambda_value=lam,internal_equal_view_cosine=float(legacy.view_weights(domains[v])@np.sum(pred*t,axis=1))))
    selected=max(range(len(candidates)),key=lambda i:candidates[i]['internal_equal_view_cosine'])
    lam=candidates[selected]['lambda_value'];m=masks['fit']
    coef,bias=legacy.ridge_fit(x[m],y[m],lam,legacy.view_weights(domains[m]))
    held=masks['heldout'];pred=np.asarray(x[held],dtype=np.float64)@coef+bias;t=np.asarray(y[held],dtype=np.float64)
    dom=domains[held]
    metrics={d:score_prediction(pred[dom==d],t[dom==d]) for d in ['drone','satellite']}
    metrics['combined']=score_prediction(pred,t)
    metrics['equal_view_cosine']=float(np.mean([metrics[d]['cosine_similarity'] for d in ['drone','satellite']]))
    return dict(selected_lambda=lam,internal_selection=candidates,heldout=metrics,
        fit_ids=len(split['fit']),heldout_ids=len(split['heldout']),identity_overlap=0)
def spectrum(x):
    x=np.asarray(x,dtype=np.float64);x=x-x.mean(0)
    vals=np.maximum(np.linalg.eigvalsh(x.T@x/(len(x)-1)),0)[::-1]
    return dict(eigenvalues=vals.tolist(),effective_rank_PR=float(vals.sum()**2/(vals@vals)),n=len(x))
def prepare():
    if (OUT/'AUDIT_CONFIG.json').exists():raise FileExistsError('Audit already prepared')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'_CACHE').mkdir(exist_ok=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    cache=PREVIOUS/'_CACHE/teacher_train_descriptors.npy'
    meta=read(PREVIOUS/'_CACHE/teacher_train_descriptors.meta.json')
    index=read(PREVIOUS/'_CACHE/TRAIN_IMAGE_INDEX.json');rows=index['rows']
    split=read(PREVIOUS/'TRAIN_ID_SPLIT.json')
    assert meta['TRAIN_ONLY'] and meta['model_state_unchanged'] and meta['strict_load']['sha256']==sha(TEACHER)
    assert sha(cache)==meta['descriptor_sha256'] and legacy.digest(rows)==meta['rows_sha256']==index['rows_sha256']
    assert len(rows)==38555 and all(train_guard(row['path']).is_file() for row in rows)
    assert {row['domain'] for row in rows}=={'drone','satellite'}
    split_guard(rows,split)
    assert len(split['fit'])==560 and len(split['heldout'])==141 and len(split['internal_fit'])==448 and len(split['internal_val'])==112
    assert split==legacy.identity_split(sorted(set(row['pid'] for row in rows)))
    z=np.load(cache,mmap_mode='r')
    assert z.shape==(38555,768) and z.dtype==np.float32 and np.isfinite(z).all()
    assert np.max(np.abs(np.linalg.norm(z,axis=1)-1))<1e-5
    original=load_stst_asset(ORIGINAL,sha(TEACHER));asset=load_extended_asset(EXTENDED,ORIGINAL,sha(TEACHER))
    checks=check_extended_tensors(asset,original)
    assert torch.equal(asset['top64_basis'],asset['top128_basis'][:,:64])
    # Reproduce the exact original random construction: it does NOT project away Top32.
    gen=torch.Generator(device='cpu').manual_seed(20260808)
    recreated=torch.linalg.qr(torch.randn(768,32,generator=gen,dtype=torch.float64),mode='reduced').Q.float().contiguous()
    assert torch.equal(recreated,asset['random32_A'])
    protected=[TEACHER,TEACHER.parent/'run_config.json',ORIGINAL,EXTENDED,cache,
      PREVIOUS/'_CACHE/teacher_train_descriptors.meta.json',PREVIOUS/'_CACHE/TRAIN_IMAGE_INDEX.json',
      PREVIOUS/'TRAIN_ID_SPLIT.json',PREVIOUS/'AUDIT_CONFIG.json',PREVIOUS/'RESULT_MANIFEST.txt',
      STUDENT/'TOP_BANDWIDTH_3SEED_BOARD.json']
    for pattern in ['D0-DUAL-STST-S*','P1-T128-R32-S*']:
        for run in STUDENT.glob(pattern):
            if run.is_dir():protected.extend(p for p in run.rglob('*') if p.is_file())
    thresholds=dict(null_near_percentile=[5,95],geometry_pearson_low_below=.4,geometry_pearson_high_at_least=.8,
      predictability_low_cos_below=.4,predictability_high_cos_at_least=.8,predictability_high_R2_at_least=.5,
      unique_top5_min_fraction=.2,close_target_cos_at_least=.9,close_geometry_pearson_at_least=.9,
      close_top5_overlap_at_least=.8,clean_subspace_overlap_at_most=1e-6)
    config=dict(TRAIN_ONLY=True,ZERO_TRAINING=True,cache_reused=True,cache_path=str(cache),cache_meta=meta,
      split_path=str(PREVIOUS/'TRAIN_ID_SPLIT.json'),TRAIN_SPLIT_REUSED=True,FIT_IDS=560,HELDOUT_IDS=141,
      identity_overlap=0,N_NULL=N_NULL,null_seed=SEED,lambda_candidates=LAMBDAS,thresholds=thresholds,
      original_random_generation='Torch CPU seed20260808 Gaussian(768,32) -> float64 QR -> FP32; no U32 removal; bit-exact reproduction PASS',
      conditional_null_note='User conditional-U32 null is a counterfactual control; primary inference uses matched unconditional ambient null because rho32 is nonzero.',
      target_semantics='Production DualSTSTSupervision.teacher_targets on FP32 descriptors/bases/mean; no trained model or head instantiated.',
      geometry='L2-normalized identity mean of official per-image normalized targets;141heldout TRAIN identities. Within-view upper triangle excludes self;D2S all cells, same-ID exclusion sensitivity also reported.',
      probe='Official per-image targets; FP64 closed-form ridge with unregularized intercept; equal total fit weight per view. lambda chosen only internal448/112 IDs by equal-view cosine, first tie. combined heldout is image-pooled, per-view and equal-view separately.',
      metric_definitions='Prediction L2-normalized before primary cosine/R2/NMSE;R2=1-SSE/SST;NMSE=SSE/sum(target^2), per-coordinate MSE also supplied;raw prediction R2/NMSE supplementary.',
      residual_coordinates='float64 reduced QR with positive diagonal; target cosine is coordinate dependent, report raw QR and Procrustes aligned-to-original basis; geometry invariant to orthogonal rotations.',
      variance='Total energy centered using certified teacher_mean;all images primary plus equal-identity/domain centroid sensitivity. Explained energy is not retrieval importance.',
      limitations=['All701identities already informed Teacher/bank; heldout is only disjoint for probe fitting, not unseen identity generalization.',
       'No TEST/SUES/GTA descriptors or performance values enter mechanism analysis. Existing bandwidth decision used only as historical fact.',
       'Decision thresholds are descriptive audit conventions, not universal statistical thresholds.'],
      initial_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      protected_files=snapshot(protected),basis_checks=checks)
    write('AUDIT_CONFIG.json',config)
    write('_CACHE/TRAIN_ID_SPLIT.json',split)
    print('CACHE_REUSED=True; TRAIN_SPLIT_REUSED=True; ORIGINAL_RANDOM_REPRODUCED_EXACT=True',flush=True)
def analyze():
    config=read(OUT/'AUDIT_CONFIG.json');unchanged(config['protected_files'])
    z=np.load(config['cache_path']);rows=read(PREVIOUS/'_CACHE/TRAIN_IMAGE_INDEX.json')['rows']
    split=read(config['split_path']);held=split['heldout']
    original=load_stst_asset(ORIGINAL,sha(TEACHER));asset=load_extended_asset(EXTENDED,ORIGINAL,sha(TEACHER))
    mean=asset['teacher_mean'];u=asset['top128_basis'].double().numpy();r=asset['random32_A'].double().numpy()
    bases={f'T{k}':u[:,:k] for k in [32,64,128]}
    ortho={key:basis_guard(v,(768,int(key[1:]))) for key,v in bases.items()};ortho['R32']=basis_guard(r,(768,32))
    observed={key:overlap(v,r) for key,v in bases.items()}
    nulls={}
    for conditional,label in [(True,'conditional_U32_complement'),(False,'unconditional_matched_primary')]:
        values,audit=random_null(bases['T32'],bases['T64'],bases['T128'],conditional=conditional)
        np.save(OUT/'_CACHE'/('null_'+label+'.npy'),values)
        nulls[label]=dict(audit=audit,**{f'T{k}':null_summary(values[:,i],observed[f'T{k}']['rho']) for i,k in enumerate([64,128])})
        print('NULL_COMPLETE='+label,flush=True)
    sub=dict(TOP32_NESTED_EXACT=torch.equal(original['top32_basis'],asset['top128_basis'][:,:32]),
      TOP64_NESTED_EXACT=torch.equal(asset['top64_basis'],asset['top128_basis'][:,:64]),
      RANDOM32_ORIGINAL_EXACT=torch.equal(original['random32_basis'],asset['random32_A']),
      orthogonality_max_errors=ortho,overlap=observed,null=nulls,
      analytical_conditional=dict(EXPECTED_OVERLAP_T64=32/736,EXPECTED_OVERLAP_T128=96/736,
        OBSERVED_MINUS_EXPECTED_T64=observed['T64']['rho']-32/736,OBSERVED_MINUS_EXPECTED_T128=observed['T128']['rho']-96/736,
        valid_as_primary_for_official_R32=False),
      analytical_unconditional=dict(T32=32/768,T64=64/768,T128=128/768,valid_as_primary=True))
    pair('SUBSPACE_OVERLAP',sub)
    targets={};rnd=None
    for key,basis in bases.items():
        targets[key],rnd=official_targets(z,mean,basis,r)
    targets['R32']=rnd
    matrices={key:geometry(t,rows,held) for key,t in targets.items()}
    redundancy={key+'_vs_R32':compare_geometry(matrices[key],matrices['R32']) for key in bases}
    pair('TARGET_REDUNDANCY',dict(heldout_ids=held,comparisons=redundancy,
         geometry_construction=config['geometry']))
    probes={}
    for key in bases:
        for a,b in [(key,'R32'),('R32',key)]:
            probes[a+'_to_'+b]=ridge_probe(targets[a],targets[b],rows,split)
            print('PROBE_COMPLETE='+a+'_to_'+b,flush=True)
    pair('LINEAR_PREDICTABILITY',dict(probes=probes,protocol=config['probe'],metrics=config['metric_definitions']))
    raw,q,aligned=residualize(u,r)
    residual_checks=dict(raw_residual_frobenius_energy=float(np.sum(raw*raw)),
       raw_residual_energy_fraction=float(np.sum(raw*raw)/32),expected_fraction=1-observed['T128']['rho'],
       energy_identity_abs_error=abs(float(np.sum(raw*raw)/32)-(1-observed['T128']['rho'])),
       orthogonality_error=basis_guard(q,(768,32)),U128_cross_error=float(np.abs(u.T@q).max()))
    residual_checks['RESIDUAL_RANDOM_ORTHOGONAL_PASS']=residual_checks['U128_cross_error']<1e-6
    assert residual_checks['RESIDUAL_RANDOM_ORTHOGONAL_PASS'] and residual_checks['energy_identity_abs_error']<1e-6
    _,tqr=official_targets(z,mean,u,q);_,talign=official_targets(z,mean,u,aligned)
    ids=np.array([row['pid'] for row in rows]);doms=np.array([row['domain'] for row in rows]);mask=np.isin(ids,held)
    cos={}
    for label,t in [('positive_diagonal_QR',tqr),('Procrustes_aligned',talign)]:
        cos[label]={d:score_prediction(t[mask if d=='combined' else mask&(doms==d)],rnd[mask if d=='combined' else mask&(doms==d)])
                    for d in ['combined','drone','satellite']}
    resmat=geometry(talign,rows,held)
    residual=dict(checks=residual_checks,old_new_subspace=overlap(r,q),
        old_vs_residual_target_metrics=cos,coordinate_note=config['residual_coordinates'],
        original_vs_residual_geometry=compare_geometry(matrices['R32'],resmat),
        T128_vs_residual_geometry=compare_geometry(matrices['T128'],resmat),
        AUDIT_ONLY=True,OFFICIAL_RANDOM_REPLACED=False)
    np.savez(OUT/'_CACHE/residual_random_audit_only.npz',raw_residual=raw,qr_basis=q,aligned_basis=aligned)
    pair('RESIDUAL_RANDOM_ANALYSIS',residual)
    x=np.asarray(z,dtype=np.float64)-mean.numpy().astype(np.float64)
    total=float(np.sum(x*x));variance={k:float(np.sum((x@v)**2)/total) for k,v in {**bases,'R32':r,'R32_PERP':q}.items()}
    allids=sorted(set(ids));centroid_z=np.concatenate([legacy.centroid(z,rows,allids,d) for d in ['drone','satellite']])
    cx=centroid_z-mean.numpy();ctotal=float(np.sum(cx*cx))
    centroid_ratios={k:float(np.sum((cx@v)**2)/ctotal) for k,v in {**bases,'R32':r,'R32_PERP':q}.items()}
    spec={key:spectrum(targets[key]) for key in ['T128','R32']}
    cspec={key:spectrum(np.concatenate([legacy.centroid(targets[key],rows,allids,d) for d in ['drone','satellite']])) for key in ['T128','R32']}
    pair('TEACHER_VARIANCE_ALLOCATION',dict(V_total=total,mean_centered_squared_norm=total/len(z),
       ratios=variance,equal_identity_view_centroid_ratios=centroid_ratios,
       covariance_spectrum=spec,equal_identity_centroid_target_spectrum=cspec,
       note='Production targets are per-image L2 normalized; covariance separately centers normalized targets. Variance ratios use unnormalized projections about fixed certified mean. R32 overlaps Top128, so ratios are not additive.'))
    unchanged(config['protected_files'])
    write('_CACHE/ANALYSIS_DONE.json',dict(PASS=True,OPTIMIZER_STEP_CALLS=0,TEACHER_FORWARD_CALLS=0))
if __name__=='__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','analyze'])
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:analyze()
