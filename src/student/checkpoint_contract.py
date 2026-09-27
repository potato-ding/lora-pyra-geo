"""Formal Student best-only checkpoint validation and reload boundaries."""
import hashlib
import json
from pathlib import Path

import torch

from src.evaluation.precision_contract import VERSION, flat_selection_metrics
from src.student.subspace_utils import tensor_sha256

SCHEMA = 'STUDENT_BEST_MODEL_V1'
ARTIFACT_CONTRACT = 'STUDENT_BEST_ONLY_V1'


def config_fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_student_best(payload, *, verify_assets=False):
    if not isinstance(payload, dict) or payload.get('artifact_schema') != SCHEMA:
        raise ValueError('Formal Student checkpoint schema mismatch')
    state, meta, config = payload.get('model'), payload.get('metadata'), payload.get('config')
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(x) for x in state.values()):
        raise ValueError('Student deployment state missing')
    if any(any(token in name.lower() for token in ('stst', 'projector', 'teacher', 'random32_basis', 'allocation_gate')) for name in state):
        raise ValueError('Training auxiliary leaked into Student deployment state')
    if not isinstance(meta, dict) or not isinstance(config, dict):
        raise ValueError('Student checkpoint identity missing')
    size = meta.get('image_size')
    if size not in (224, 256) or meta.get('experiment_id') != f'S3-ADUAL-LEARNABLE-R{size}':
        raise ValueError('Student experiment/resolution identity mismatch')
    if meta.get('artifact_contract') != ARTIFACT_CONTRACT or meta.get('student_architecture') != 'RepViT-M1.5':
        raise ValueError('Student model/artifact identity mismatch')
    if config.get('img_size') != size or config.get('experiment_name') != meta['experiment_id']:
        raise ValueError('Student config/resolution identity mismatch')
    if meta.get('config_sha256') != config_fingerprint(config):
        raise ValueError('Student config SHA mismatch')
    from .core_config import validate_config
    validate_config(config)
    signature = payload.get('precision_signature')
    if not isinstance(signature, dict) or any((signature.get('model_type') != 'student',
        signature.get('architecture') != 'StudentModel',
        signature.get('precision_contract_version') != VERSION,
        signature.get('image_size') != size,
        signature.get('parameter_dtype') != 'bfloat16',
        signature.get('descriptor_dtype') != 'float32')):
        raise ValueError('Student precision signature mismatch')
    if set(signature.get('state_shapes', {})) != set(state) or any(
            signature['state_shapes'][name] != list(value.shape) for name, value in state.items()):
        raise ValueError('Student state/signature mismatch')
    if payload.get('protocol_id') != f'STU-1G-B32-R{size}-v1':
        raise ValueError('Student protocol mismatch')
    from .artifacts import selection_metadata
    if payload.get('selection_protocol') != selection_metadata():
        raise ValueError('Student selection protocol mismatch')
    if payload.get('best_epoch') != meta.get('best_epoch') or payload.get('best_score') != meta.get('best_score'):
        raise ValueError('Student best-selection identity mismatch')
    refs = flat_selection_metrics(payload.get('selection_metrics', {}))
    selected = flat_selection_metrics(meta.get('selection_metrics', {}))
    if refs != selected or refs['D2S']['R@1'] + refs['S2D']['R@1'] != meta['best_score']:
        raise ValueError('Student selection metrics mismatch')
    for key in ('middle_checkpoint', 'middle_checkpoint_sha256', 'middle_config',
                'middle_config_sha256', 'stst_asset', 'extended_stst_asset_sha256',
                'random_basis_seed', 'random_basis_sha256'):
        if not meta.get(key):
            raise ValueError('Student source identity missing: ' + key)
    aux = payload.get('training_auxiliary')
    if not isinstance(aux, dict) or aux.get('schema') != 'RANDOM_BASIS_PROJECTOR_V1':
        raise ValueError('Formal Student training auxiliary missing')
    if not isinstance(aux.get('supervision'), dict) or not isinstance(aux.get('allocation_gate'), dict):
        raise ValueError('Student heads/gate missing')
    if 'random32_basis' not in aux['supervision'] or 'd' not in aux['allocation_gate']:
        raise ValueError('Student Random32/gate state missing')
    if not any(k.startswith('projector_top.') for k in aux['supervision']) or not any(
            k.startswith('projector_random.') for k in aux['supervision']):
        raise ValueError('Student Top/Random head state missing')
    identity = aux.get('basis_identity', {})
    if identity.get('random_basis_sha256') != meta['random_basis_sha256'] or tensor_sha256(aux['supervision']['random32_basis']) != meta['random_basis_sha256']:
        raise ValueError('Student Random32 SHA mismatch')
    if identity.get('random_basis_seed') != meta['random_basis_seed']:
        raise ValueError('Student Random32 seed mismatch')
    if verify_assets:
        for path_key, sha_key in (('middle_checkpoint', 'middle_checkpoint_sha256'),
                                  ('middle_config', 'middle_config_sha256'),
                                  ('stst_asset', 'extended_stst_asset_sha256')):
            path = Path(meta[path_key])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != meta[sha_key]:
                raise ValueError('Student source asset SHA mismatch: ' + path_key)
    return meta


def full_training_resume(*args, **kwargs):
    raise NotImplementedError('Formal Student full training resume is unsupported; best_model.pth restores deployment and training auxiliary only')


def verify_checkpoint_file(path, *, verify_assets=True):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    return verify_student_best(payload, verify_assets=verify_assets)
