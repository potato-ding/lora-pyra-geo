"""Versioned Teacher selection signature for actual batch-size-16 validation."""
from src.evaluation.precision_contract import selection_signature


def formal_selection_signature(model, model_type, image_size):
    if model_type != 'teacher' or image_size not in (224,256):
        raise ValueError('Only formal Teacher R224/R256 is supported')
    signature = dict(selection_signature(model,model_type,image_size))
    if signature.get('selection_batch_size') != 8:
        raise ValueError('Unexpected parent precision contract')
    signature['selection_batch_size'] = 16
    return signature
