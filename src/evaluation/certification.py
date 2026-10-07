"""Same-descriptor clean/reference certification and bounded cache persistence."""
import hashlib
import json
from pathlib import Path
from .protocols import RETRIEVAL_PROTOCOL, GTA_PROTOCOL
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
        result=run_sues_val_and_get_metrics(model,*pair,device,precomputed_features=features)
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
    horizontal_flip = False
    if path.exists():
        if dataset == 'gta':
            previous_path=cache_dir/(key+'.json')
            if not previous_path.is_file() or json.loads(previous_path.read_text()).get('gta_protocol') != GTA_PROTOCOL:
                raise RuntimeError('GTA cache protocol missing or changed; use a fresh cache directory')
        if dataset == 'u1652':
            previous_path=cache_dir/(key+'.json')
            if (not previous_path.is_file() or json.loads(previous_path.read_text()).get('retrieval_protocol') != RETRIEVAL_PROTOCOL):
                raise RuntimeError('U1652 cache uses an incompatible gallery protocol; use a fresh cache directory')
        if dataset == 'sues200':
            previous_path=cache_dir/(key+'.json')
            if (not previous_path.is_file()
                    or json.loads(previous_path.read_text()).get('retrieval_protocol') != RETRIEVAL_PROTOCOL):
                raise RuntimeError('SUES cached features use an incompatible protocol; use a fresh cache directory')
        provenance_path=cache_dir/(key+'.json')
        provenance=json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
        if provenance.get('extraction_batch_sizes') != [pair[0].batch_size,pair[1].batch_size]:
            raise RuntimeError('Cached extraction batch sizes missing or changed; use a fresh cache directory')
        features=read_features(path)
        if len(features[0])!=len(pair[0].dataset) or len(features[3])!=len(pair[1].dataset):
            raise RuntimeError('Resume cache dataset-count mismatch')
        print(f'REUSING_EXTRACTED_FEATURES={path}',flush=True)
    else:
        features=(*extract_features_dist(model,pair[0],device,stage_name=key+':query'),
                  *extract_features_dist(model,pair[1],device,stage_name=key+':gallery'))
        cache_features(path,features,pair)
    clean=measure_clean(model,pair,features,dataset,device)
    reference=measure_reference(features,dataset,device)
    rows=[{'metric':k,'clean':v,'reference':reference[k],'abs_diff':abs(v-reference[k]),
           'tolerance':METRIC_ATOL,'pass':bool(abs(v-reference[k])<=METRIC_ATOL)} for k,v in clean.items()]
    record={'dataset':dataset,'direction':direction,'horizontal_flip':horizontal_flip,
            'extraction_batch_sizes':[pair[0].batch_size,pair[1].batch_size],
            'clean':clean,'reference':reference,'rows':rows,
            'ranking':'numpy_descending_sort_then_remove_junk',
            'pass':all(r['pass'] for r in rows),'feature_cache':str(path),'feature_sha256':feature_sha(path),
            'query_count':len(pair[0].dataset),'gallery_count':len(pair[1].dataset),
            'query_norm_max_error':float((features[0].norm(dim=1)-1).abs().max()),
            'gallery_norm_max_error':float((features[3].norm(dim=1)-1).abs().max())}
    if dataset == 'gta': record['gta_protocol']=GTA_PROTOCOL
    if dataset == 'u1652': record['retrieval_protocol']=RETRIEVAL_PROTOCOL
    if dataset == 'sues200': record['retrieval_protocol']=RETRIEVAL_PROTOCOL
    (cache_dir/(key+'.json')).write_text(json.dumps(record,indent=2))
    print('METRIC_CERTIFICATION='+json.dumps(record),flush=True)
    if dataset == 'gta' and record.get('gta_protocol') != GTA_PROTOCOL:
        raise RuntimeError('GTA certified cache protocol missing or changed')
    if not record['pass']:raise RuntimeError(f'Clean/reference disagreement: {key}; cache retained for diagnosis')
    return record


def reuse_pair(model,pair,dataset,direction,device,cache_dir):
    cache_dir=Path(cache_dir);key=f'{dataset}_{direction}'
    record=json.loads((cache_dir/(key+'.json')).read_text())
    path=cache_dir/(key+'.npz')
    if dataset == 'u1652' and record.get('retrieval_protocol') != RETRIEVAL_PROTOCOL:
        raise RuntimeError('U1652 certified cache uses an incompatible gallery protocol')
    if dataset == 'sues200' and record.get('retrieval_protocol') != RETRIEVAL_PROTOCOL:
        raise RuntimeError('SUES certified features use an incompatible protocol')
    if dataset == 'gta' and record.get('gta_protocol') != GTA_PROTOCOL:
        raise RuntimeError('GTA certified cache protocol missing or changed')
    if record.get('extraction_batch_sizes') != [pair[0].batch_size,pair[1].batch_size]:
        raise RuntimeError('Certified extraction batch sizes missing or changed')
    if not record['pass'] or feature_sha(path)!=record['feature_sha256']:
        raise RuntimeError('Uncertified or changed feature cache')
    if record['query_count']!=len(pair[0].dataset) or record['gallery_count']!=len(pair[1].dataset):
        raise RuntimeError('Dataset counts changed after certification')
    metrics=measure_clean(model,pair,read_features(path),dataset,device)
    if any(abs(metrics[k]-record['clean'][k])>METRIC_ATOL for k in metrics):
        raise RuntimeError('Formal/cache metric regression failed')
    return metrics
