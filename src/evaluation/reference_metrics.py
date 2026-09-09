"""Independent certification metrics. Never used to fit a model.

Reference protocols: layumi/University1652-Baseline evaluate_gpu.py;
Reza-Zhu/SUES-200-Benchmark evaluate.py/test_and_evaluate.py;
Yux1angJi/GTA-UAV Game4Loc/game4loc/evaluate/gta.py.
Source revisions/hashes are saved with the certification report.
"""
import numpy as np
import torch


def _validate(features):
    if features.dtype != torch.float32 or features.ndim != 2:
        raise ValueError('Certification requires FP32 descriptors')
    if not torch.isfinite(features).all() or not torch.allclose(features.norm(dim=1),torch.ones(len(features)),atol=1e-4):
        raise ValueError('Certification requires finite L2 normalized descriptors')


def _rankings(q,g,device):
    _validate(q);_validate(g)
    gallery=g.to(device)
    for start in range(0,len(q),1000):
        # Use the same FP32 dot-product definition, independent NumPy ranking.
        scores=(q[start:start+1000].to(device) @ gallery.T).cpu().numpy()
        for score in scores:yield np.argsort(score)[::-1]


def _triangle(qf,ql,gf,gl,device):
    labels=gl.numpy().reshape(-1);query_labels=ql.numpy().reshape(-1)
    hits1=hits5=0;ap_sum=0.0
    for label,order in zip(query_labels,_rankings(qf,gf,device)):
        order=order[labels[order]!=-1]
        ranks=np.flatnonzero(labels[order]==label)
        if len(ranks)==0:continue
        hits1+=int(ranks[0]<1);hits5+=int(ranks[0]<5)
        ap=0.0
        for i,rank in enumerate(ranks):
            precision=(i+1)/(rank+1)
            old_precision=i/rank if rank!=0 else 1.0
            ap+=(old_precision+precision)/(2*len(ranks))
        ap_sum+=ap
    return {'R@1':float(100*hits1/len(qf)),'R@5':float(100*hits5/len(qf)),'AP':float(100*ap_sum/len(qf))}


def u1652_official_reference_metrics(qf,ql,gf,gl,device='cpu'):
    return _triangle(qf,ql,gf,gl,device)


def sues_official_reference_metrics(qf,ql,gf,gl,device='cpu'):
    return _triangle(qf,ql,gf,gl,device)


def gta_official_reference_metrics(qf,ql,qc,gf,gl,gc,device='cpu'):
    hits=0;ap_sum=dis_sum=sdm_sum=0.0
    labels=gl.numpy().reshape(-1);coords=gc.numpy().astype(np.float64)
    query_coords=qc.numpy().astype(np.float64)
    for i,order in enumerate(_rankings(qf,gf,device)):
        positives=ql[i].numpy().reshape(-1);positives=positives[positives!=-1]
        ranks=np.flatnonzero(np.isin(labels[order],positives))
        if len(ranks)==0:raise ValueError('Formal GTA query has no positive in gallery')
        hits+=int(ranks[0]==0)
        ap_sum+=float(np.mean(np.arange(1,len(ranks)+1)/(ranks+1)))
        distances=np.linalg.norm(coords[order[:3]]-query_coords[i],axis=1)
        weights=np.arange(len(distances),0,-1,dtype=np.float64)
        dis_sum+=float(distances[0])
        sdm_sum+=float(np.sum(weights*np.exp(-.001*distances))/weights.sum())
    return {'R@1':100*hits/len(qf),'AP':100*ap_sum/len(qf),
            'DIS@1':dis_sum/len(qf),'SDM@3':100*sdm_sum/len(qf)}
