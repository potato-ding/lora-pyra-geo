"""Tensor identity and orthogonality checks shared by formal subspaces."""
import hashlib
import torch


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def orthogonality_error(matrix):
    x=matrix.double()
    return float((x.T@x-torch.eye(x.shape[1],dtype=x.dtype,device=x.device)).abs().max())
