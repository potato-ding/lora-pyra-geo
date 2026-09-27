"""Exact legacy config identity and E3 KD-only runtime identity."""
import copy
import hashlib
import json

_E3_SAM_KEYS = frozenset({
    'framework', 'enabled', 'sharpness_mode', 'search_direction',
    'perturb_scope', 'rho', 'adaptive', 'same_batch', 'rng_replay',
    'second_pass_objective', 'norm_epsilon',
    'balanced_task_weight', 'balanced_kd_weight',
})
_NONRUNTIME_KD_KEYS = ('balanced_task_weight', 'balanced_kd_weight')


def canonical_runtime_config(config):
    normalized = copy.deepcopy(config)
    name = normalized.get('experiment', {}).get('name', '')
    sam = normalized.get('sam', {})
    if name.startswith('M2-SAM-E3-KD-R') and sam.get('search_direction') == 'kd':
        unknown = set(sam) - _E3_SAM_KEYS
        if unknown:
            raise ValueError('Unknown E3 SAM fields: ' + repr(sorted(unknown)))
        for key in _NONRUNTIME_KD_KEYS:
            if key in sam and sam[key] != .5:
                raise ValueError('Legacy E3 nonruntime field changed: ' + key)
            sam.pop(key, None)
    return normalized


def runtime_fingerprint(config):
    value = canonical_runtime_config(config)
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
