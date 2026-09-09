"""TRAIN-only canonical Dual-STST subspace mathematics.

Recovered from build_stst_teacher_subspace.py: identity centroids, shared
1402-row float64 SVD, and a fixed-seed float64 random QR. No fitting on tests.
This function does not run extraction or save a bank implicitly.
"""
import torch
import torch.nn.functional as F


def identity_centroids(features, labels):
    features=F.normalize(features.float(),dim=1)
    labels=labels.view(-1)
    identities=sorted(set(map(int,labels.tolist())))
    rows=[]
    for identity in identities:
        rows.append(F.normalize(features[labels==identity].float().mean(dim=0),dim=0))
    return torch.stack(rows),identities


def construct_train_subspace(drone_features,drone_labels,satellite_features,satellite_labels,*,middle_sha256,split):
    if split!='train':raise ValueError('Dual-STST fits University-1652 TRAIN only')
    if len(middle_sha256)!=64:raise ValueError('Middle SHA256 must be recorded')
    drone,drone_ids=identity_centroids(drone_features.cpu(),drone_labels.cpu())
    satellite,satellite_ids=identity_centroids(satellite_features.cpu(),satellite_labels.cpu())
    if drone_ids!=satellite_ids or len(drone_ids)!=701:
        raise ValueError('Expected exactly 701 matching TRAIN identities')
    if any(int((satellite_labels==identity).sum())!=1 for identity in satellite_ids):
        raise ValueError('Expected one TRAIN satellite per identity')
    x=torch.cat((drone,satellite),dim=0).double()
    if x.shape!=(1402,768) or not torch.isfinite(x).all():raise ValueError('Invalid TRAIN descriptors')
    mean=x.mean(dim=0)
    _,_,vh=torch.linalg.svd(x-mean,full_matrices=False)
    top=vh[:32].T.float().contiguous()
    generator=torch.Generator(device='cpu').manual_seed(20260808)
    random=torch.linalg.qr(torch.randn(768,32,generator=generator,dtype=torch.float64),mode='reduced').Q.float().contiguous()
    return {'teacher_mean':mean.float().contiguous(),'top32_basis':top,'random32_basis':random,
            'metadata':{'teacher_sha256':middle_sha256,'dataset':'University-1652','split':'train',
                        'train_ids':701,'teacher_dim':768,'subspace_dim':32,'random_seed':20260808,
                        'train_only':True,'shared_drone_satellite_basis':True,'train_rows':1402}}
