"""Same-descriptor clean/reference certification and bounded cache persistence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .reference_metrics import u1652_official_reference_metrics,sues_official_reference_metrics,gta_official_reference_metrics
from .metrics import getdist_1652_val_and_get_recall,run_sues_val_and_get_metrics,run_gta_val_and_get_metrics
from src.utils.train_eval_utils import extract_features_dist

METRIC_ATOL=1e-4  # percentage points, or meters for DIS; FP32 accumulation only


def measure_clean(model,pair,features,dataset,device):
    if dataset=='u1652':
        r1,r5,_,ap=getdist_1652_val_and_get_recall(model,*pair,device,precomputed_features=features)
        return {'R@1':r1,'R@5':r5,'AP':ap}
    if dataset=='sues200':
        result=run_sues_val_and_get_metrics(model,*pair,device,horizontal_flip=False,precomputed_features=features)
        return {k:result[k] for k in ('R@1','AP')}
    result=run_gta_val_and_get_metrics(model,*pair,device,precomputed_features=features)
    return {k:result[k] for k in ('R@1','AP','DIS@1','SDM@3')}


def measure_reference(features,dataset,device):
    qf,ql,qc,gf,gl,gc=features
    if dataset=='u1652':return u1652_official_reference_metrics(qf,ql,gf,gl,device)
    if dataset=='sues200':return sues_official_reference_metrics(qf,ql,gf,gl,device)
    return gta_official_reference_metrics(qf,ql,qc,gf,gl,gc,device)


def cache_features(path,features,pair=None):
    data={k:v.numpy() for k,v in zip(('qf','ql','qc','gf','gl','gc'),features) if v is not None}
    if pair is not None and hasattr(pair[0].dataset,'images'):
        data['query_paths']=np.asarray(pair[0].dataset.images)
        data['gallery_paths']=np.asarray(pair[1].dataset.images)
    np.savez(path,**data)


def align_u1652_cache(path,features,pair,direction):
    """Reindex already-extracted features; never rerun a model for a sort fix."""
    with np.load(path,allow_pickle=False) as data:
        if 'query_paths' in data:
            old={'query':data['query_paths'].tolist(),'gallery':data['gallery_paths'].tolist()}
        else:
            old=json.loads((path.parent/'original_sample_order.json').read_text())[direction]
    target={'query':pair[0].dataset.images,'gallery':pair[1].dataset.images}
    if old==target:return features
    changed=list(features)
    for offset,view in ((0,'query'),(3,'gallery')):
        index={str(p):i for i,p in enumerate(old[view])}
        if len(index)!=len(old[view]) or set(index)!=set(target[view]):raise RuntimeError('Cache path set mismatch')
        permutation=torch.tensor([index[p] for p in target[view]],dtype=torch.long)
        for k in range(offset,offset+3):
            if changed[k] is not None:changed[k]=changed[k][permutation]
    backup=path.with_suffix('.presort.npz')
    if backup.exists():raise FileExistsError('Pre-sort backup already exists')
    path.rename(backup)
    cache_features(path,changed,pair)
    print(f'OFFICIAL_ORDER_REINDEX_ONLY={path}; original={backup}',flush=True)
    return tuple(changed)


def read_features(path):
    with np.load(path,allow_pickle=False) as data:
        return tuple(torch.from_numpy(data[k].copy()) if k in data else None for k in ('qf','ql','qc','gf','gl','gc'))


def feature_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


@torch.no_grad()
def certify_pair(model,pair,dataset,direction,device,cache_dir):
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    key=f'{dataset}_{direction}';path=cache_dir/(key+'.npz')
    if path.exists():
        features=read_features(path)
        if dataset=='u1652':features=align_u1652_cache(path,features,pair,direction)
        if len(features[0])!=len(pair[0].dataset) or len(features[3])!=len(pair[1].dataset):
            raise RuntimeError('Resume cache dataset-count mismatch')
        print(f'REUSING_EXTRACTED_FEATURES={path}',flush=True)
    else:
        features=(*extract_features_dist(model,pair[0],device,stage_name=key+':query',horizontal_flip=False),
                  *extract_features_dist(model,pair[1],device,stage_name=key+':gallery',horizontal_flip=False))
        cache_features(path,features,pair)
    clean=measure_clean(model,pair,features,dataset,device)
    reference=measure_reference(features,dataset,device)
    rows=[{'metric':k,'clean':v,'reference':reference[k],'abs_diff':abs(v-reference[k]),
           'tolerance':METRIC_ATOL,'pass':bool(abs(v-reference[k])<=METRIC_ATOL)} for k,v in clean.items()]
    record={'dataset':dataset,'direction':direction,'clean':clean,'reference':reference,'rows':rows,
            'certification_version':'OFFICIAL_NUMPY_ORDER_V2',
            'pass':all(r['pass'] for r in rows),'feature_cache':str(path),'feature_sha256':feature_sha(path),
            'query_count':len(pair[0].dataset),'gallery_count':len(pair[1].dataset),
            'query_norm_max_error':float((features[0].norm(dim=1)-1).abs().max()),
            'gallery_norm_max_error':float((features[3].norm(dim=1)-1).abs().max())}
    (cache_dir/(key+'.json')).write_text(json.dumps(record,indent=2))
    print('METRIC_CERTIFICATION='+json.dumps(record),flush=True)
    if not record['pass']:raise RuntimeError(f'Clean/reference disagreement: {key}; cache retained for diagnosis')
    return record


def reuse_pair(model,pair,dataset,direction,device,cache_dir):
    cache_dir=Path(cache_dir);key=f'{dataset}_{direction}'
    record=json.loads((cache_dir/(key+'.json')).read_text())
    path=cache_dir/(key+'.npz')
    if not record['pass'] or feature_sha(path)!=record['feature_sha256']:
        raise RuntimeError('Uncertified or changed feature cache')
    if record['query_count']!=len(pair[0].dataset) or record['gallery_count']!=len(pair[1].dataset):
        raise RuntimeError('Dataset counts changed after certification')
    metrics=measure_clean(model,pair,read_features(path),dataset,device)
    if any(abs(metrics[k]-record['clean'][k])>METRIC_ATOL for k in metrics):
        raise RuntimeError('Formal/cache metric regression failed')
    return metrics
