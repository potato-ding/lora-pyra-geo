"""Deployment fusion must preserve trained BN statistics and standalone neck."""
import torch
from torch import nn
from src.models.repvit_module import Conv2d_BN, RepVGGDW, Residual
from src.evaluation.efficiency import prepare_deployment

torch.manual_seed(91)
model = nn.Sequential(Conv2d_BN(3, 8, 3, pad=1), RepVGGDW(8),
                      Residual(Conv2d_BN(8, 8, 3, pad=1, groups=8)),
                      nn.Conv2d(8, 8, 1), nn.BatchNorm2d(8),
                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(8)).eval()
for m in model.modules():
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.running_mean.copy_(torch.randn_like(m.running_mean))
        m.running_var.copy_(torch.rand_like(m.running_var)+.1)
        m.weight.data.copy_(torch.randn_like(m.weight))
        m.bias.data.copy_(torch.randn_like(m.bias))
x = torch.randn(4, 3, 16, 16)
with torch.inference_mode():
    expected = model(x)
    neck = model[-1]
    audit = prepare_deployment(model)
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert model[-1] is neck and audit['batchnorm_modules_after'] == 1
    second = prepare_deployment(model)
    assert not second['changes']
    torch.testing.assert_close(model(x), actual, atol=0, rtol=0)
print('PASS: trained BN/branch fusion, standalone neck preservation, idempotence')

# A conversion rejected for the quantized spike path must preserve its BN.
from types import SimpleNamespace
from src.evaluation.efficiency import prepare_deployment
spike = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4)).eval()
spike.kind = 'spikevimformer'
probe = torch.randn(2, 3, 4, 4)
with torch.inference_mode():
    before_spike = spike(probe)
    audit = prepare_deployment(spike)
    assert torch.equal(before_spike, spike(probe))
assert isinstance(spike[1], nn.BatchNorm2d) and audit['changes'] == []
assert audit['batchnorm_modules_before'] == audit['batchnorm_modules_after'] == 1
print('PASS: rejected spike folding retains original BN and exact output')
