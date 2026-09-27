"""TRAIN identity centroids for canonical Top128 fitting."""
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
