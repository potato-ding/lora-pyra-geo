"""Strict adapters for the uploaded LPN and MuSe-Net University checkpoints."""
import importlib
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import yaml

class LegacyEncoder(nn.Module):
    def __init__(self, model, kind):
        super().__init__()
        self.model = model
        self.kind = kind
        self.descriptor_dim = 2048 if kind == 'lpn' else 512

    @torch.no_grad()
    def forward_unnormalized(self, images):
        self.model.eval()
        # Both uploads have identical satellite/drone encoder tensors; use the
        # satellite entry of the full network, retaining MuSe's style adapter.
        with torch.autocast(images.device.type, enabled=False):
            features = self.model(images.float(), None, None)[0].float()
        if self.kind == 'lpn':
            if features.ndim != 3 or features.shape[1:] != (512, 4):
                raise RuntimeError('Invalid LPN part descriptor')
            features = F.normalize(features, dim=1).flatten(1)
        if features.ndim != 2 or features.shape[1] != self.descriptor_dim or not torch.isfinite(features).all():
            raise RuntimeError('Invalid legacy descriptor')
        return features

    def encode(self, images):
        return F.normalize(self.forward_unnormalized(images), dim=1)

    def forward(self, images):
        return self.encode(images)

def load_legacy(kind, base, device):
    import torchvision.models as tv
    from src.middle_teacher.checkpoint import sha256
    source = base/'source'
    if 'model' in sys.modules or 'resnet_adaibn' in sys.modules:
        raise RuntimeError('Run each legacy baseline in its own process')
    original_resnet = tv.resnet50
    tv.resnet50 = lambda *args, **kwargs: original_resnet(weights=None)
    configs = {}
    restore_ibn = None
    try:
        if kind == 'lpn':
            code = source
            weight = base/'weights/net_119.pth'
        else:
            candidates = sorted((base/'weights').glob('three_view*'))
            if len(candidates) != 1:
                raise RuntimeError('Expected exactly one uploaded MuSe-Net bundle')
            code = candidates[0]
            weight = code/'net_209.pth'
            configs = yaml.safe_load((code/'opts.yaml').read_text())
            sys.path.insert(0, str(source))
        sys.path.insert(0, str(code))
        if kind == 'muse':
            ibn = importlib.import_module('resnet_adaibn')
            restore_ibn = ibn.resnet50_adaibn_a
            ibn.resnet50_adaibn_a = lambda *a, **kw: restore_ibn(pretrained=False, adain=kw.get('adain', 'a'))
        module = importlib.import_module('model')
        if Path(module.__file__).resolve() != (code/'model.py').resolve():
            raise RuntimeError('Unexpected model module')
        if kind == 'lpn':
            arguments = dict(stride=1, pool='avg', share_weight=True, LPN=True, block=4)
            model = module.three_view_net(701, .75, **arguments)
        else:
            arguments = dict(stride=configs['stride'], pool=configs['pool'], share_weight=configs['share'],
                LPN=configs['LPN'], block=configs['block'], norm=configs['norm'], adain=configs['adain'],
                circle=configs['circle'], btnk=configs['btnk'], conv_norm=configs['conv_norm'])
            model = module.three_view_net(configs['nclasses'], configs['droprate'], **arguments)
    finally:
        tv.resnet50 = original_resnet
        if restore_ibn is not None:
            ibn.resnet50_adaibn_a = restore_ibn
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    paired = [(k, k.replace('model_1.', 'model_3.', 1)) for k in state if k.startswith('model_1.')]
    if not paired or not all(other in state and torch.equal(state[k], state[other]) for k, other in paired):
        raise RuntimeError('Satellite and drone checkpoint encoders differ')
    if kind == 'lpn':
        for index in range(4):
            getattr(model, 'classifier'+str(index)).classifier = nn.Identity()
    else:
        model.classifier.classifier = nn.Identity()
    encoder = LegacyEncoder(model.to(device).eval().requires_grad_(False), kind).eval()
    warnings = (['Training opts.yaml absent; stride=1, 256 input and shared four-part variant follow upstream defaults; extra Google training data unverified']
                if kind == 'lpn' else ['Training configuration includes extra Google street-view data'])
    files = [code/'model.py']
    if kind == 'muse': files += [code/'resnet_adaibn.py', source/'modules.py', code/'opts.yaml']
    audit = dict(model=kind, checkpoint=str(weight), checkpoint_sha256=sha256(weight),
        upstream_commit=(source/'UPSTREAM_COMMIT.txt').read_text().strip(),
        code_sha256={str(path):sha256(path) for path in files}, strict_checkpoint_load=True,
        satellite_drone_tensors_identical=True, descriptor_dim=encoder.descriptor_dim,
        precision='FP32 parameters and forward; FP32 normalization and similarity',
        training_dataset='University-1652', extra_Google=(configs.get('extra_Google') if kind == 'muse' else None),
        input_size=256, constructor=arguments, training_configuration=configs, warnings=warnings,
        checkpoint_selection=('uploaded net_119' if kind == 'lpn' else 'upstream latest-epoch default net_209; no target-dataset checkpoint selection'),
        adaptations=['Remove classification output head after strict loading',
            'Use full shared satellite/drone network; retain learned style conditioning for MuSe-Net',
            'LPN part-wise normalization before flattening' if kind == 'lpn' else 'MuSe-Net global bottleneck descriptor'])
    return encoder, audit, 256
