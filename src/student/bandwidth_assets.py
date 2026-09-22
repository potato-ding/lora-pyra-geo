"""Strict FP32 nested bandwidth assets, anchored to immutable formal V128/R32."""
import hashlib
import json
from pathlib import Path
import torch
from .dual_stst import file_sha256

SCHEMA = 'NESTED_BANDWIDTH_V1'
RANDOM_EXTENSION_SEED = 20260923
ORTHO_ATOL = 1e-5
SHAPES = {'teacher_mean': (768,), 'top128_basis': (768,128),
          'top256_basis': (768,256), 'random32_A': (768,32),
          'random64_basis': (768,64), 'random128_basis': (768,128)}


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def orthogonality_error(matrix):
    x = matrix.double()
    return float((x.T @ x - torch.eye(x.shape[1], dtype=x.dtype, device=x.device)).abs().max())


def orthogonal_extension(candidates, anchor):
    # QR only the working copy of the anchor; never replace its FP32 values.
    q = torch.linalg.qr(anchor.double(), mode='reduced').Q
    residual = candidates.double() - q @ (q.T @ candidates.double())
    extra = torch.linalg.qr(residual, mode='reduced').Q
    pivots = extra.abs().argmax(dim=0)
    extra *= torch.where(extra[pivots, torch.arange(extra.shape[1])] < 0, -1., 1.)
    return extra.float().contiguous()


def validate_tensors(asset, old):
    for key, shape in SHAPES.items():
        t = asset[key]
        if t.shape != shape or t.dtype != torch.float32 or not torch.isfinite(t).all():
            raise ValueError('Invalid bandwidth tensor: ' + key)
    checks = dict(
        OLD_MU_PRESERVED=torch.equal(asset['teacher_mean'], old['teacher_mean']),
        OLD_V128_PRESERVED=torch.equal(asset['top128_basis'], old['top128_basis']),
        OLD_R32_PRESERVED=torch.equal(asset['random32_A'], old['random32_A']),
        V256_PREFIX_EQUALS_V128_BITWISE=torch.equal(asset['top256_basis'][:,:128], old['top128_basis']),
        R64_PREFIX_EQUALS_R32_BITWISE=torch.equal(asset['random64_basis'][:,:32], old['random32_A']),
        R128_PREFIX_EQUALS_R64_BITWISE=torch.equal(asset['random128_basis'][:,:64], asset['random64_basis']))
    for key in SHAPES:
        if key == 'teacher_mean': continue
        error = orthogonality_error(asset[key])
        checks[key + '_orthogonality_max_error'] = error
        checks[key + '_orthogonality_pass'] = error <= ORTHO_ATOL
    if not all(v for v in checks.values() if isinstance(v, bool)):
        raise ValueError(checks)
    return checks


def build_tensors(rows, old):
    if rows.shape != (1402,768) or not torch.isfinite(rows).all():
        raise ValueError('Exactly 1402 finite TRAIN identity representatives required')
    mean, top, random = (old[k].cpu().clone() for k in ('teacher_mean','top128_basis','random32_A'))
    centered = rows.double().cpu() - mean.double()
    residual = centered - (centered @ top.double()) @ top.double().T
    _, singular, vh = torch.linalg.svd(residual, full_matrices=False)
    top256 = torch.cat((top, orthogonal_extension(vh[:128].T, top)), dim=1)
    generator = torch.Generator(device='cpu').manual_seed(RANDOM_EXTENSION_SEED)
    gaussian = torch.randn(768,96,generator=generator,dtype=torch.float64)
    # No PCA projection: Random is an internally orthogonal random sketch.
    random128 = torch.cat((random, orthogonal_extension(gaussian, random)), dim=1)
    asset = dict(teacher_mean=mean, top128_basis=top, top256_basis=top256,
                 random32_A=random, random64_basis=random128[:,:64].clone(),
                 random128_basis=random128, random32_B=random128[:,32:64].clone())
    checks = validate_tensors(asset, old)
    checks['residual_singular_values'] = singular[:128].tolist()
    return asset, checks


def validate_asset(asset, original_path, teacher_sha):
    from .part1 import load_extended_asset
    m = asset['metadata']
    expected = dict(schema=SCHEMA, teacher_sha256=teacher_sha,
        original_stst_asset_sha256=file_sha256(original_path), image_size=224,
        dataset='University-1652', split='train', train_only=True,
        train_ids=701, bank_rows=1402, teacher_dim=768,
        available_top_dims=[128,256], available_random_dims=[32,64,128])
    if any(m.get(k) != v for k,v in expected.items()):
        raise ValueError('Bandwidth asset identity mismatch')
    old_path = Path(m['anchor_asset_path'])
    if file_sha256(old_path) != m['anchor_asset_sha256']:
        raise ValueError('Immutable anchor SHA mismatch')
    old = load_extended_asset(old_path, original_path, teacher_sha)
    validate_tensors(asset, old)
    for key in SHAPES:
        if tensor_sha256(asset[key]) != m['tensor_sha256'][key]:
            raise ValueError('Bandwidth tensor SHA mismatch: ' + key)
    return asset


def validate_manifest(cfg, asset):
    path = Path(cfg['asset_manifest'])
    if file_sha256(path) != cfg['asset_manifest_sha256']:
        raise ValueError('Bandwidth manifest SHA mismatch')
    manifest = json.loads(path.read_text())
    if manifest['asset_sha256'] != file_sha256(cfg['stst_asset']):
        raise ValueError('Manifest asset binding mismatch')
    if manifest['metadata'] != asset['metadata']:
        raise ValueError('Manifest metadata mismatch')
