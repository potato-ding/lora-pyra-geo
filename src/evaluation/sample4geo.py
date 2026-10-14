"""Sample4Geo model adapter for shared retrieval evaluation.

External checkpoints have their own provenance, never a fabricated internal
train-selection/precision certification. Run one dataset per visible GPU.
"""
from contextlib import nullcontext
import hashlib
import importlib.util
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'src/checkpoint/others/Sample4geo'
MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k_384'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Sample4GeoEncoder(nn.Module):
    descriptor_dim = 1024

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, images):
        # Shared extraction applies FP32 L2 to this single-forward descriptor.
        context = (torch.autocast('cuda', dtype=torch.float16)
                   if images.device.type == 'cuda' else nullcontext())
        with context:
            features = self.model(images)
        features = features.float()
        if features.ndim != 2 or features.shape[1] != self.descriptor_dim:
            raise RuntimeError(f'Invalid Sample4Geo descriptor shape: {features.shape}')
        if not torch.isfinite(features).all():
            raise RuntimeError('Nonfinite Sample4Geo descriptor')
        return features

    @torch.no_grad()
    def encode(self, images):
        # Adapter smoke checks use normalized FP32 descriptors.
        return F.normalize(self(images), dim=-1)


def load_encoder(checkpoint, source, device):
    model_file = source / 'sample4geo/model.py'
    spec = importlib.util.spec_from_file_location('_official_sample4geo_model', model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TimmModel(MODEL_NAME, pretrained=False, img_size=384)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Deliberately strict: no silently ignored keys or fallback backbone weights.
    loaded = model.load_state_dict(state, strict=True)
    config = model.get_config()
    if tuple(config['mean']) != (0.485, 0.456, 0.406) or tuple(config['std']) != (0.229, 0.224, 0.225):
        raise RuntimeError('Dataset transforms do not match model normalization')
    encoder = Sample4GeoEncoder(model.to(device)).eval()
    encoder.requires_grad_(False)
    commit_file = source / 'UPSTREAM_COMMIT.txt'
    commit = commit_file.read_text().strip() if commit_file.exists() else subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    import timm
    audit = dict(model='Sample4Geo', backbone=MODEL_NAME, training_dataset='University-1652',
                 checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                 upstream='https://github.com/Skyy93/Sample4Geo', upstream_commit=commit,
                 model_source_sha256=sha256(model_file), evaluator_sha256=sha256(__file__),
                 image_size=384, state_keys=len(state), missing=loaded.missing_keys,
                 unexpected=loaded.unexpected_keys, parameters=sum(p.numel() for p in model.parameters()),
                 torch_version=torch.__version__, timm_version=timm.__version__,
                 parameters_dtype='float32', inference_autocast='cuda_float16',
                 descriptor_dtype='float32', similarity_dtype='float32',
                 normalization='single-forward L2 FP32', batch_size=16,
                 train_selection_certification='not_applicable_external_checkpoint',
                 shared_sources={str(p.relative_to(ROOT)): sha256(p) for p in [
                     ROOT/'src/utils/train_eval_utils.py',
                     ROOT/'src/dataset/teacher/val_dataloaders.py']})
    return encoder, audit
