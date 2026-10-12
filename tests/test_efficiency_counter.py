"""Analytical operator-count and functional-parameter coverage checks."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from src.evaluation.efficiency import ParameterUse, flops, executed_flops


class Functional(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3,2))
        self.alias = self.weight
        self.bias = nn.Parameter(torch.ones(3))
        self.unused = nn.Parameter(torch.ones(100))
    def forward(self, x):
        return F.relu(F.linear(x, self.weight.view(3,2), self.bias))


model = Functional().eval()
x = torch.ones(1,2)
watch = ParameterUse(model)
with torch.inference_mode(), watch:
    model(x)
assert watch.used == {('parameter','weight'), ('parameter','bias')}, watch.used
result = flops(model, x)
assert result['status'] == 'operator_coverage_complete', result
assert math.isclose(result['gflops']*1e9, 15), result  # 6 MACs*2 plus three ReLUs.


class Attention(nn.Module):
    def forward(self, q):
        return F.scaled_dot_product_attention(q, q, q, dropout_p=0.)


result = flops(Attention(), torch.ones(1,2,4,3))
assert result['status'] == 'operator_coverage_complete', result
assert math.isclose(result['gflops']*1e9, 576), result
for test_model, inputs, expected in [(model,x,15),(Attention(),torch.ones(1,2,4,3),576)]:
    result = executed_flops(test_model, inputs)
    assert result['status']=='operator_coverage_complete', result
    assert math.isclose(result['gflops']*1e9,expected), result
with torch.inference_mode():
    result = executed_flops(nn.Conv2d(4,4,3,groups=2).eval(), torch.ones(1,4,5,5))
    assert result['status']=='operator_coverage_complete', result
    assert math.isclose(result['gflops']*1e9,1296), result
print('PASS: MAC convention, SDPA shape formula, parameter views/aliases, unused exclusion')

class Nearest(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')
with torch.inference_mode():
    result = executed_flops(Nearest(), torch.randn(1, 3, 4, 4))
    assert result['gflops'] == 0 and not result['unsupported_operators'], result
print('PASS: nearest-neighbor sampling has zero floating arithmetic')

class BilinearPooling(nn.Module):
    def forward(self, x):
        attention = x[:, :2]
        first = torch.einsum('imjk,injk->imn', attention, x)
        torch.einsum('imjk,injk->imn', attention, x)  # executed, discarded
        return first

for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
    inputs = torch.randn(2, 3, 4, 5, device=device)
    attention = inputs[:, :2]
    actual = torch.einsum('imjk,injk->imn', attention, inputs)
    expected = torch.bmm(attention.flatten(2), inputs.flatten(2).transpose(1, 2))
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    with torch.inference_mode():
        result = executed_flops(BilinearPooling(), inputs)
    assert result['status'] == 'operator_coverage_complete', result
    assert math.isclose(result['gflops'] * 1e9, 2 * (2 * 2 * 2 * 3 * 4 * 5)), result
print('PASS: bilinear attention contraction matches bmm; discarded calls counted on CPU/CUDA')

class BroadcastMatmul(nn.Module):
    def __init__(self, right):
        super().__init__(); self.register_buffer('right', right)
    def forward(self, x):
        return torch.matmul(x, self.right)

for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
    for left_shape, right_shape in [((2,3,4),(1,4,5)), ((4,),(4,)), ((3,4),(4,)), ((4,),(2,4,5))]:
        left = torch.randn(left_shape, device=device)
        right = torch.randn(right_shape, device=device)
        expected = 2 * torch.matmul(left, right).numel() * left.shape[-1]
        with torch.inference_mode():
            result = executed_flops(BroadcastMatmul(right), left)
        assert result['status'] == 'operator_coverage_complete', result
        assert math.isclose(result['gflops'] * 1e9, expected), result
    conv = nn.Conv1d(4, 6, 3, groups=2, bias=False).to(device).eval()
    left = torch.randn(2,4,8,device=device)
    with torch.inference_mode():
        result = executed_flops(conv, left)
    assert result['status'] == 'operator_coverage_complete', result
    assert math.isclose(result['gflops'] * 1e9, 2 * 2 * 6 * 6 * 2 * 3), result
print('PASS: broadcast/vector matmul and grouped conv1d MAC counts on CPU/CUDA')
