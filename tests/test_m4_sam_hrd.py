"""M4 ablation and real two-pass objective integration without semantic features."""
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/middle_teacher'


@pytest.mark.parametrize("size", [224, 256])
def test_m4_is_m1_plus_exact_m3_sam(size):
    m1 = json.loads((CONFIGS / f'm1_hrd_{size}.json').read_text())
    m3 = json.loads((CONFIGS / f'm3_sam_hrd_sem_{size}.json').read_text())
    m4 = json.loads((CONFIGS / f'm4_sam_hrd_{size}.json').read_text())
    assert m4['sam'] == m3['sam']
    assert 'semantic' not in m4
    expected = dict(m1, sam=m3['sam'], experiment_id=f'M4-SAM-HRD-R{size}',
                    output_dir=str(ROOT / f'src/checkpoint/middle_teacher/R{size}/M4-{size}'))
    assert m4 == expected


def test_m4_rejects_accidental_semantic():
    from src.middle_teacher.formal_config import load_formal_config
    path = CONFIGS / 'm4_sam_hrd_256.json'
    public = json.loads(path.read_text())
    public['semantic'] = json.loads((CONFIGS / 'm2_hrd_sem_256.json').read_text())['semantic']
    with patch.object(Path, 'read_text', return_value=json.dumps(public)):
        with pytest.raises(ValueError, match='config keys mismatch'):
            load_formal_config(path)


@pytest.mark.parametrize('semantic', [False, True])
def test_real_sam_objective_two_passes_optional_semantic(monkeypatch, semantic):
    import src.middle_teacher.distill_sam as sam
    import src.middle_teacher.formal_distillation as kd_module
    import src.utils.gather_features_and_labels_and_views as gather

    torch.manual_seed(41)
    calls = []
    class Middle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 768)
            self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1/.07)))
            if semantic:
                self.layer_semantic_projectors = torch.nn.Linear(768, 768)
        def forward(self, images, return_layer_features=False):
            calls.append(return_layer_features)
            descriptor = F.normalize(self.backbone(images).float(), dim=-1)
            if return_layer_features:
                assert semantic
                return {'final_descriptor': descriptor, 'middle_features': [descriptor]}
            return descriptor

    model = Middle()
    teacher = torch.nn.Linear(3, 768).requires_grad_(False).eval()
    runtime = kd_module.FormalDistillationRuntime.__new__(kd_module.FormalDistillationRuntime)
    runtime.teacher = teacher
    runtime.local_pair_batch = runtime.chunk_size = 2
    runtime.config = {'base_loss': 'pair_infonce', 'margin': {'enabled': True, 'weight': .1}}
    if semantic:
        runtime.config['adaptive_bridge_v2'] = {'weight': .05, 'teacher_layers': [28, 36]}
    features = {'final_cls': F.normalize(torch.randn(4, 768), dim=-1),
                'timing': {'teacher_forward_time': 0.}}
    for layer in [28, 36]:
        features[f'layer{layer}_cls'] = torch.zeros(4, 768)
        features[f'layer{layer}_patch'] = torch.zeros(4, 2, 768)
    def teacher_forward(*args, **kwargs):
        assert kwargs['return_patch_tokens'] is semantic
        return features
    monkeypatch.setattr(kd_module, 'adaptive_teacher_fused_forward', teacher_forward)
    monkeypatch.setattr(kd_module, 'concat_all_gather', lambda x: x)
    monkeypatch.setattr(gather, 'concat_all_gather', lambda x: x)
    monkeypatch.setattr(gather.GatherLayer, 'apply', lambda x: (x,))
    monkeypatch.setattr(sam.dist, 'get_world_size', lambda: 1)
    if semantic:
        monkeypatch.setattr(kd_module, 'adaptive_bridge_v2_loss',
                            lambda cls, patches, target, bank, component:
                            ((bank(target)-cls[0]).square().mean(), {}))

    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    class Engine:
        def __init__(self):
            self.module = model
            self.backward_count = 0
        def __call__(self, *args, **kwargs):
            return model(*args, **kwargs)
        def zero_grad(self):
            optimizer.zero_grad(set_to_none=True)
        def backward(self, loss):
            self.backward_count += 1
            loss.backward()
    engine = Engine()
    original = {n: p.detach().clone() for n,p in model.named_parameters()}
    teacher_original = {n: p.detach().clone() for n,p in teacher.named_parameters()}
    options = dict(rho=.1, norm_epsilon=1e-12)
    result = sam.sam_backward(engine, runtime, torch.randn(4,3), torch.arange(2), 0, options)
    assert calls == [semantic, semantic]
    assert engine.backward_count == 1
    assert result[-1]['kd_grad_norm'] > 0
    assert result[-1]['perturb_restore_exact']
    assert result[-1]['second_pass_objective'] == 'full'
    assert result[-2]['margin_active']
    assert ('adaptive_bridge_v2_active' in result[-2]) is semantic
    assert all(torch.equal(p,original[n]) for n,p in model.named_parameters())
    assert all(p.grad is None and torch.equal(p,teacher_original[n]) for n,p in teacher.named_parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert all(optimizer.state[p]['step'] == 1 for p in model.parameters() if p.grad is not None)
    assert any(not torch.equal(p,original[n]) for n,p in model.named_parameters())


def test_m4_checkpoint_requires_sam_metadata(monkeypatch):
    from src.middle_teacher.formal_config import to_runtime_config
    from src.middle_teacher.artifacts import SCHEMA, checkpoint_metadata
    from src.middle_teacher.selection import selection_metadata
    import hashlib
    monkeypatch.setattr('src.middle_teacher.checkpoint.sha256', lambda _: 'foundation-sha')
    path = CONFIGS / 'm4_sam_hrd_256.json'
    public = json.loads(path.read_text())
    internal = to_runtime_config(path)
    metrics = {'D2S': {'R@1':20., 'R@5':30., 'AP':.2},
               'S2D': {'R@1':25., 'R@5':35., 'AP':.3}, 'R1_sum':45.}
    protocol = selection_metadata(256,16)
    encoded = json.dumps(public, sort_keys=True, separators=(',',':'),ensure_ascii=False).encode()
    metadata = dict(protocol, experiment_id=public['experiment_id'], best_epoch=6,
                    best_score=45., training_world_size=2, selection_metrics=metrics,
                    public_config_sha256=hashlib.sha256(encoded).hexdigest(),
                    sam=True, sharpness=internal['sam'], teacher={'checkpoint_metadata':
                    dict(experiment_id='T0-INFONCE-R256',image_size=256,
                         selection_mode='SINGLE_GPU_CANONICAL',selection_world_size=1,selection_rank=0)})
    payload = dict(artifact_schema=SCHEMA,metadata=metadata,config=internal,
                   public_config=public,selection_metrics=metrics,selection_protocol=protocol,
                   precision_signature=dict(image_size=256,selection_batch_size=16))
    assert checkpoint_metadata(payload) == metadata
    invalid = copy.deepcopy(payload)
    invalid['metadata']['sam'] = False
    with pytest.raises(ValueError,match='SAM metadata'):
        checkpoint_metadata(invalid)
