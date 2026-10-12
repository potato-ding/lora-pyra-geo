"""Single-process, single-GPU retrieval encoder efficiency measurement.

Run each model in a fresh process. FP32 measurements are separate from the
checkpoint's accuracy-evaluation precision contract; accuracy files are untouched.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from collections import Counter
import subprocess
import time
from datetime import datetime, timezone
from unittest.mock import patch
from contextlib import ExitStack

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

ROOT = Path(__file__).resolve().parents[2]


class RetrievalForward(nn.Module):
    """Descriptor operations identical to evaluator; validation is outside timing."""
    def __init__(self, encoder, kind):
        super().__init__()
        self.model = encoder.model
        self.kind = kind

    def forward(self, images):
        k, m = self.kind, self.model
        if k == 'middle':
            # Same numerical path as DINOv3ViTB16Backbone.forward, without its
            # per-call nonfinite assertion (which forces GPU synchronization).
            from src.models.dinov3_hierarchical import forward_frozen_prefix
            backbone = m.backbone
            if backbone.hierarchical_config is None:
                _, cls = backbone.model.get_intermediate_layers(
                    images, n=backbone.target_layers, return_class_token=True)[0]
            else:
                cfg = backbone.hierarchical_config
                cls = forward_frozen_prefix(backbone.model, images,
                    cfg.frozen_prefix_length, cfg.preserve_nonblock_trainability)
            return F.normalize(cls.float(), dim=-1, eps=1e-6)
        elif k == 'student':
            return m(images).float()
        elif k == 'sample4geo':
            features = m(images).float()
        elif k in ('mean', 'dac', 'camp', 'dmnil'):
            features = m(images)[-2].float()
        elif k in ('mccg', 'ccr'):
            features = F.normalize(m(images, None)[0].float(), dim=1).flatten(1)
        elif k in ('lpn', 'muse'):
            features = m(images, None, None)[0].float()
            if k == 'lpn':
                features = F.normalize(features, dim=1).flatten(1)
        elif k == 'spikevimformer':
            features = m(images)[0].mean((-2, -1)).float()
        elif k == 'infogeo_rd_trial':
            features = m.slot_mixvpr(m.model(images).flatten(2)).float()
        else:
            raise ValueError(k)
        return F.normalize(features, dim=-1)


class ParameterUse(TorchDispatchMode):
    """Observe actual operator inputs, including functional parameter use/views.

    This audits the exercised retrieval path, not arbitrary data-dependent paths.
    Unused tensors are retained on CPU, never deleted from the checkpoint model.
    """
    def __init__(self, model):
        super().__init__()
        self.lookup = {}
        for label, entries in [('parameter', model.named_parameters()),
                               ('buffer', model.named_buffers())]:
            for name, tensor in entries:
                if tensor.numel():
                    key = (str(tensor.device), tensor.untyped_storage().data_ptr())
                    self.lookup.setdefault(key, set()).add((label, name))
        self.used = set()
        self.floating_dtypes = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        for item in tree_flatten((args, kwargs or {}))[0]:
            if isinstance(item, torch.Tensor) and item.numel():
                if item.device.type == 'cuda' and item.is_floating_point():
                    self.floating_dtypes.add(str(item.dtype))
                key = (str(item.device), item.untyped_storage().data_ptr())
                self.used.update(self.lookup.get(key, ()))
        return func(*args, **(kwargs or {}))


def fp32_context():
    # Existing adapters explicitly open autocast contexts internally. Override
    # only inside this isolated benchmark process, using the public context API.
    original = torch.autocast
    def disabled(device_type, *args, **kwargs):
        return original(device_type=device_type, enabled=False)
    return patch.object(torch, 'autocast', disabled)


def validate_output(output):
    if output.ndim != 2 or output.dtype != torch.float32:
        raise RuntimeError('Expected a two-dimensional FP32 descriptor')
    if not torch.isfinite(output).all():
        raise RuntimeError('Nonfinite descriptor')
    if not torch.allclose(output.norm(dim=1), torch.ones(len(output), device=output.device), atol=1e-4):
        raise RuntimeError('Descriptor is not normalized')


def parity(left, right):
    validate_output(left)
    validate_output(right)
    if left.shape != right.shape or not torch.allclose(left, right, atol=1e-6, rtol=1e-5):
        raise RuntimeError('Efficiency forward differs from evaluator')
    return {'max_abs_difference': float((left-right).abs().max()),
            'min_cosine': float(F.cosine_similarity(left, right).min())}


def sample_inputs(size, batch):
    # Actual University-1652 images; fixed, deterministic input and preprocessing.
    import cv2
    import numpy as np
    root = ROOT/'data/U1652/test/query_drone'
    files = sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.jpg', '.png', '.jpeg'))[:batch]
    if len(files) != batch:
        raise RuntimeError(f'Need {batch} real images under {root}')
    values = []
    for p in files:
        x = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR_EXACT)
        x = (x.astype(np.float32)/255 - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        values.append(torch.from_numpy(x.transpose(2, 0, 1).copy()))
    return torch.stack(values).cuda(), [str(p.relative_to(ROOT)) for p in files]


def flops(model, x):
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape, elementwise_flop_counter
    import fvcore
    analysis = FlopCountAnalysis(model, (x,))
    # fvcore treats fused multiply-add as one; scale only MAC-based handlers.
    mac_ops = ('aten::addmm', 'aten::bmm', 'aten::_convolution', 'aten::einsum',
               'aten::matmul', 'aten::mm', 'aten::linear')
    for op in mac_ops:
        original = analysis._op_handles[op]
        def twice(inputs, outputs, handle=original):
            result = handle(inputs, outputs)
            return {k: 2*v for k, v in result.items()} if isinstance(result, dict) else 2*result
        analysis.set_op_handle(op, twice)
    def sdpa(inputs, outputs):
        q, k, v = [get_shape(a) for a in inputs[:3]]
        batches = math.prod(q[:-2])
        nq, nk, dq, dv = q[-2], k[-2], q[-1], v[-1]
        # Dense attention MACs plus 5 operations per softmax element and scale.
        return {'sdpa': batches*nq*nk*(2*(dq+dv)+6)}
    analysis.set_op_handle('aten::scaled_dot_product_attention', sdpa)
    # Official Quant.forward = clamp to [0,4] (two comparisons) then round;
    # the subsequent division is a separate aten::div node, counted below.
    analysis.set_op_handle('prim::PythonOp.Quant', elementwise_flop_counter(0, 3))
    def maxpool(inputs, outputs):
        kernel = inputs[1].toIValue()
        if kernel is None:
            kernel = [v.toIValue() for v in inputs[1].node().inputs()]
        if isinstance(kernel, int):
            kernel = [kernel, kernel]
        return {'max_pool_comparisons': math.prod(get_shape(outputs[0]))*(math.prod(kernel)-1)}
    for op in ('max_pool2d', 'max_pool2d_with_indices'):
        analysis.set_op_handle('aten::'+op, maxpool)
    for op in ('add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'rsqrt', 'abs', 'neg', 'exp', 'log', 'sign', 'cos', 'sin'):
        analysis.set_op_handle('aten::'+op, elementwise_flop_counter(0, 1))
    for op in ('add_', 'sub_', 'mul_', 'div_', 'relu_'):
        analysis.set_op_handle('aten::'+op, elementwise_flop_counter(0, 1))
    for op, factor in [('gelu', 8), ('silu', 5), ('sigmoid', 4), ('relu', 1), ('clamp', 2),
                       ('clamp_min', 1), ('round', 1), ('softmax', 5), ('sigmoid_', 4)]:
        analysis.set_op_handle('aten::'+op, elementwise_flop_counter(0, factor))
    for op in ('sum', 'mean', 'linalg_vector_norm', 'norm'):
        factor = 2 if 'norm' in op else 1
        analysis.set_op_handle('aten::'+op, elementwise_flop_counter(factor, 0))
    analysis.set_op_handle('aten::std', elementwise_flop_counter(4, 0))
    analysis.set_op_handle('aten::upsample_bicubic2d', elementwise_flop_counter(0, 32))
    analysis.set_op_handle('aten::upsample_bilinear2d', elementwise_flop_counter(0, 8))
    # Structural/dtype/shape operators are not floating-point arithmetic.
    ignored = ('aten::to', 'aten::detach', 'aten::clone', 'aten::contiguous', 'aten::expand',
               'aten::expand_as', 'aten::repeat', 'aten::repeat_interleave', 'aten::tile', 'aten::unbind',
               'aten::split', 'aten::split_with_sizes', 'aten::chunk', 'aten::index',
               'aten::index_select', 'aten::gather', 'aten::roll', 'aten::meshgrid',
               'aten::floor_divide', 'aten::Int', 'aten::size', 'aten::len',
               'aten::eq', 'aten::ne', 'aten::gt', 'aten::lt', 'aten::ge', 'aten::le',
               'aten::isfinite', 'aten::all', 'aten::any', 'aten::ones_like',
               'aten::zeros_like', 'aten::full_like', 'aten::new_zeros', 'aten::new_ones', 'aten::pad')
    for op in ignored:
        analysis.set_op_handle(op, None)
    analysis.tracer_warnings('none').unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    trace_adaptations = []
    with ExitStack() as stack:
        if getattr(model, 'kind', None) == 'infogeo_rd_trial':
            dino = model.model.model.backbone.dino_model
            original = dino.interpolate_pos_encoding
            # Fixed-resolution FLOPs trace only: PyTorch tracing turns w,h
            # into scalar tensors, but interpolate requires Python scale floats.
            def static_resolution(tokens, width, height):
                return original(tokens, int(width), int(height))
            stack.enter_context(patch.object(dino, 'interpolate_pos_encoding', static_resolution))
            trace_adaptations.append('Convert image w,h to Python ints during fixed-resolution FLOPs tracing only; original method restored before timing')
        count = analysis.total()
    missing = dict(analysis.unsupported_ops())
    return {'gflops': count/1e9 if not missing else None,
            'counted_gflops_lower_bound': count/1e9,
            'status': 'operator_coverage_complete' if not missing else 'incomplete_do_not_publish',
            'unsupported_operators': missing, 'by_operator': dict(analysis.by_operator()),
            'trace_adaptations': trace_adaptations,
            'convention': '1 MAC = 2 FLOPs; documented estimates for non-MAC arithmetic; dense GPU-equivalent for spiking model',
            'fvcore_version': fvcore.__version__, 'explicit_ignored_operators': list(ignored)}


def executed_flops(model, x):
    """Count executed operators, retaining work whose result Python discards.

    JIT output-graph tracing may prune such work. TorchDispatch also exposes the
    actual clamp/round within Spike's custom autograd Function without a handler
    for an opaque PythonOp. Core convolution/matmul formulas are PyTorch's own.
    """
    from torch.utils.flop_counter import flop_registry
    arithmetic = {'add':1, 'add_':1, 'sub':1, 'sub_':1, 'mul':1, 'mul_':1,
                  'div':1, 'div_':1, 'pow':1, 'sqrt':1, 'rsqrt':1, 'abs':1,
                  'neg':1, 'exp':1, 'log':1, 'cos':1, 'sin':1, 'sign':1,
                  'relu':1, 'relu_':1, 'clamp':2, 'clamp_':2, 'clamp_min':1,
                  'round':1, 'sigmoid':4, 'sigmoid_':4, 'gelu':8, 'silu':5,
                  '_softmax':5, '_safe_softmax':5, 'softmax':5,
                  'upsample_bilinear2d':8, 'upsample_bicubic2d':32}
    structural = set('detach alias view view_as reshape _unsafe_view _reshape_alias '
        'permute transpose t unsqueeze squeeze expand expand_as repeat tile '
        'repeat_interleave split split_with_sizes chunk unbind cat stack slice '
        'select narrow index index_select gather roll clone contiguous copy_ '
        '_to_copy to lift_fresh empty empty_like empty_strided new_empty '
        'flatten meshgrid dropout zeros zeros_like new_zeros ones ones_like new_ones full full_like '
        'arange linspace scalar_tensor fill_ zero_ constant_pad_nd reflection_pad2d '
        'replication_pad2d pad upsample_nearest2d isfinite isnan isinf eq ne lt le gt ge all any '
        '_local_scalar_dense floor_divide sym_size size '.split())
    class CounterMode(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = Counter()
            self.missing = Counter()
            self.ignored = Counter()
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            packet = func._overloadpacket
            name = str(packet).split('.')[-1]
            tensors = [v for v in tree_flatten((args, kwargs))[0] if isinstance(v, torch.Tensor)]
            outputs = [v for v in tree_flatten(out)[0] if isinstance(v, torch.Tensor)]
            floating = any(t.is_floating_point() for t in tensors)
            if not floating or name in structural:
                self.ignored[str(packet)] += 1
                return out
            count = None
            if packet in flop_registry:
                count = flop_registry[packet](*args, **kwargs, out_val=out)
                if 'scaled_dot_product' in name:
                    q, k = args[:2]
                    count += math.prod(q.shape[:-2])*q.shape[-2]*k.shape[-2]*6
            elif name == 'einsum' and args[0].replace(' ', '') == 'imjk,injk->imn':
                # CCR bilinear attention pooling: B x M x H x W contracted
                # with B x C x H x W. Count every call, even discarded output.
                left, right = args[1]
                batch, parts, height, width = left.shape
                if right.ndim != 4 or tuple(right.shape[0:1] + right.shape[2:]) != (batch, height, width):
                    raise RuntimeError('Unexpected bilinear attention einsum shapes')
                count = 2 * batch * parts * right.shape[1] * height * width
            elif name in ('conv1d', 'conv2d', 'conv3d'):
                count = 2*outputs[0].numel()*math.prod(args[1].shape[1:])
            elif name == 'matmul':
                # Broadcasting and vector promotion change output size, not K.
                count = 2*outputs[0].numel()*args[0].shape[-1]
            elif name in ('scaled_dot_product_attention', '_scaled_dot_product_flash_attention_for_cpu'):
                q, k, v = args[:3]
                count = math.prod(q.shape[:-2])*q.shape[-2]*k.shape[-2]*(2*(q.shape[-1]+v.shape[-1])+6)
            elif name in arithmetic:
                count = sum(v.numel() for v in outputs if v.is_floating_point())*arithmetic[name]
            elif name in ('sum', 'mean', 'amax', 'amin', 'max', 'min'):
                count = args[0].numel()
            elif name in ('linalg_vector_norm', 'norm'):
                count = 2*args[0].numel()+outputs[0].numel()
            elif name in ('var', 'std', 'var_mean', 'std_mean'):
                count = 4*args[0].numel()
            elif name in ('native_layer_norm', 'layer_norm', 'native_group_norm', 'group_norm', 'instance_norm'):
                count = 7*args[0].numel()
            elif name in ('native_batch_norm', '_native_batch_norm_legit',
                          '_native_batch_norm_legit_no_training', 'batch_norm'):
                count = 2*args[0].numel()
            elif name in ('adaptive_avg_pool2d', '_adaptive_avg_pool2d'):
                count = args[0].numel()
            elif name in ('avg_pool2d', 'max_pool2d', 'max_pool2d_with_indices'):
                kernel = args[1]
                if isinstance(kernel, int): kernel = [kernel, kernel]
                count = outputs[0].numel()*math.prod(kernel)
                if 'max_pool' in name: count -= outputs[0].numel()
            elif name in ('addmm', 'linear'):
                # CPU linear is normally decomposed into addmm/mm before dispatch.
                count = 2*args[0].numel()*args[1].shape[0]
            if count is None:
                self.missing[str(packet)] += 1
            else:
                self.counts[str(packet)] += int(count)
            return out
    counter = CounterMode()
    with counter:
        model(x)
    total = sum(counter.counts.values())/1e9
    return {'gflops': total if not counter.missing else None,
            'counted_gflops_lower_bound': total,
            'status': 'operator_coverage_complete' if not counter.missing else 'incomplete_do_not_publish',
            'unsupported_operators': dict(counter.missing), 'by_operator': dict(counter.counts),
            'ignored_executed_operators': dict(counter.ignored),
            'convention': 'Executed dense GPU-equivalent arithmetic, 1 MAC=2 FLOPs; estimated non-MAC counts; includes discarded computations',
            'non_mac_output_factors': arithmetic,
            'normalization_and_reduction_estimates': 'LN/GN/IN=7N, inference BN=2N, norm=2N+output size, sum/mean=N, std/var=4N'}


def measure(model, x, warmup, iterations, repeats):
    for _ in range(warmup):
        y = model(x)
    del y
    torch.cuda.synchronize()
    records = []
    for _ in range(repeats):
        times = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        wall = time.perf_counter()
        if len(x) == 1:
            # Request latency includes CPU dispatch and completion of GPU work.
            # Synchronize each request; GPU-event/kernel sums are not this metric.
            for _ in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                y = model(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                del y
        else:
            # Throughput: continuous forwards, one synchronization at each end.
            # No per-iteration events or synchronization to perturb throughput.
            for _ in range(iterations):
                y = model(x)
                del y
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter()-wall
        if not times:
            times = [wall_seconds*1000/iterations] * iterations
        average = statistics.mean(times)
        records.append({'iteration_ms': times, 'mean_ms': average,
                        'iteration_samples_measured': len(x) == 1,
                        'wall_seconds': wall_seconds,
                        'median_ms': statistics.median(times),
                        'p95_ms': sorted(times)[math.ceil(.95*len(times))-1],
                        'images_per_second_from_mean_batch_time': len(x)*1000/average,
                        'synchronized_wall_images_per_second': len(x)*iterations/wall_seconds,
                        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                        'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20})
    means = [r['mean_ms'] for r in records]
    return {'batch_size': len(x), 'rounds': records,
            'mean_latency_ms_per_batch': statistics.mean(means),
            'round_mean_std_ms': statistics.stdev(means) if len(means)>1 else 0,
            'wall_fps': statistics.mean(r['synchronized_wall_images_per_second'] for r in records),
            'peak_allocated_mib': max(r['peak_allocated_mib'] for r in records)}


def prepare_deployment(model):
    """Validated structural fusion only; never drop a standalone learned BN.

    RepViT's existing fuse implementations encode branch/identity equivalence.
    Sequential Conv-BN folding is general; other architecture-specific deploy
    transformations must be added explicitly and verified before publication.
    """
    from src.models.repvit_module import Conv2d_BN, RepVGGDW, Residual
    changes = []
    before = sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if getattr(model, 'kind', None) == 'spikevimformer':
        # Quantized spike thresholds amplify FP32 roundoff from BN folding.
        # The candidate conversion failed descriptor parity at B=1,2,16;
        # retain the official backbone and the unchanged strict tolerance.
        return {'policy': 'existing RepViT structural fuse plus adjacent Sequential Conv-BN folding; standalone BN retained',
                'changes': [], 'batchnorm_modules_before': before,
                'batchnorm_modules_after': before, 'lora_merged': False,
                'remaining_batchnorm_names': [n for n,m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)],
                'skipped_fusion': 'SpikeViMFormer generic Conv-BN candidate failed descriptor parity; official unmodified backbone retained'}
    def visit(parent, prefix=''):
        for name, child in list(parent.named_children()):
            path = prefix + name
            replacement = None
            if isinstance(child, (Conv2d_BN, RepVGGDW)):
                replacement = child.fuse()
            elif isinstance(child, Residual) and isinstance(child.m, Conv2d_BN):
                replacement = child.fuse()
            if replacement is not None:
                setattr(parent, name, replacement)
                changes.append({'path': path, 'from': type(child).__name__, 'to': type(replacement).__name__})
                child = replacement
            visit(child, path+'.')
        if isinstance(parent, nn.Sequential):
            children = list(parent.named_children())
            for (cn, conv), (bn_name, bn) in zip(children, children[1:]):
                if isinstance(conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    setattr(parent, cn, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
                    setattr(parent, bn_name, nn.Identity())
                    changes.append({'path': prefix+cn+' + '+bn_name, 'from': 'sequential Conv-BN', 'to': 'fused Conv + Identity'})
    visit(model)
    model.eval().requires_grad_(False)
    after = sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    return {'policy': 'existing RepViT structural fuse plus adjacent Sequential Conv-BN folding; standalone BN retained',
            'changes': changes, 'batchnorm_modules_before': before,
            'batchnorm_modules_after': after, 'lora_merged': False,
            'remaining_batchnorm_names': [n for n,m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    from .comparison_models import SPECS, load_comparison
    p.add_argument('--model', choices=list(SPECS)+['middle', 'student'], required=True)
    p.add_argument('--label')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--image-size', type=int)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--throughput-batch', type=int, default=16)
    a = p.parse_args()
    if min(a.warmup, a.iterations, a.repeats, a.throughput_batch) < 1:
        p.error('Counts must be positive')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one otherwise idle GPU with CUDA_VISIBLE_DEVICES')
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    torch.manual_seed(20261011)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible or ',' in visible:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to one physical GPU index/UUID')
    gpu_pids = subprocess.check_output(['nvidia-smi', '-i', visible,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    other_pids = [int(line.strip()) for line in gpu_pids.splitlines()
                  if line.strip().isdigit() and int(line.strip()) != os.getpid()]
    if other_pids:
        raise RuntimeError(f'Benchmark GPU already used by processes {other_pids}')
    if a.model in ('middle', 'student'):
        if a.checkpoint is None or a.image_size not in (224, 256):
            p.error('Own models need --checkpoint and --image-size 224|256')
        from .model_loader import load_encoder
        encoder, audit = load_encoder(a.model, a.checkpoint, device=device, image_size=a.image_size)
        size = a.image_size
    else:
        if a.checkpoint is not None or a.image_size is not None:
            p.error('Comparison checkpoints and native resolution come from audited adapters')
        encoder, audit, size = load_comparison(a.model, device)
    encoder.float().eval().requires_grad_(False)
    model = RetrievalForward(encoder, a.model).eval()
    loaded_count = sum(t.numel() for t in model.parameters())
    x, input_files = sample_inputs(size, a.throughput_batch)
    checks = {}
    references = {}
    with torch.inference_mode(), fp32_context():
        for b in sorted(set([1, 2, a.throughput_batch])):
            if b > len(x): continue
            # Warm one-off source logging and dtype audits before observing use.
            expected = encoder.encode(x[:b])
            actual = model(x[:b])
            checks[str(b)] = parity(expected, actual)
            references[b] = expected.cpu()
            del expected, actual
        deployment = prepare_deployment(model)
        all_parameters = dict(model.named_parameters())
        all_buffers = dict(model.named_buffers())
        deployment_count = sum(t.numel() for t in all_parameters.values())
        observer = ParameterUse(model)
        for b in references:
            with observer:
                actual = model(x[:b])
            checks['deployment_'+str(b)] = parity(references[b].cuda(), actual)
            del actual
        print(json.dumps({'deployment_validated': a.label or a.model,
                          'fusion_changes': len(deployment['changes']),
                          'batchnorm_before_after': [deployment['batchnorm_modules_before'], deployment['batchnorm_modules_after']],
                          'parity': {k:v for k,v in checks.items() if k.startswith('deployment_')}}), flush=True)
        active = {n for kind, n in observer.used if kind == 'parameter'}
        inactive = {n: list(t.shape) for n, t in all_parameters.items() if n not in active}
        inactive_buffers = {n: list(t.shape) for n, t in all_buffers.items() if ('buffer', n) not in observer.used}
        # Retain unused tensors on CPU: full source schema and strict load audit
        # remain intact while GPU memory reflects the exercised deployment path.
        for n, tensor in list(all_parameters.items()) + list(all_buffers.items()):
            label = 'parameter' if n in all_parameters else 'buffer'
            if (label, n) not in observer.used:
                tensor.data = tensor.data.cpu()
        for b in sorted(set([1, 2, a.throughput_batch])):
            if b > len(x): continue
            checks['cpu_offload_'+str(b)] = parity(references[b].cuda(), model(x[:b]))
        active_count = sum(all_parameters[n].numel() for n in active)
        observed_dtypes = sorted(observer.floating_dtypes)
        if any(d in observed_dtypes for d in ('torch.float16', 'torch.bfloat16')):
            raise RuntimeError('FP32 forward contains low-precision arithmetic: '+str(observed_dtypes))
        del all_parameters, all_buffers, observer, encoder
        flop_result = executed_flops(model, x[:1])
        # Keep an independent output-graph count to expose pruning differences.
        flop_result['output_graph_reference'] = flops(model, x[:1])
        # Separate B=1 input allocation from B=16 memory measurement.
        x1 = x[:1].clone()
        del x
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        one = measure(model, x1, a.warmup, a.iterations, a.repeats)
        del x1
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        xb, _ = sample_inputs(size, a.throughput_batch)
        throughput = measure(model, xb, a.warmup, a.iterations, a.repeats)
    payload = {'label': a.label or a.model, 'model': a.model, 'resolution': size,
               'params_m': active_count/1e6, 'loaded_params_m': loaded_count/1e6,
               'deployment_loaded_params_m': deployment_count/1e6, 'deployment_audit': deployment,
               'inactive_parameter_shapes_cpu': inactive, 'inactive_buffer_shapes_cpu': inactive_buffers,
               'parameter_scope': 'unique parameters consumed by tensor operators in exercised retrieval forward (B=1,2,16)',
               'gflops': flop_result['gflops'], 'flop_audit': flop_result,
               'latency_b1_ms': one['mean_latency_ms_per_batch'],
               'fps_b16': throughput['wall_fps'], 'fps_batch_size': a.throughput_batch,
               'peak_allocated_b1_mib': one['peak_allocated_mib'],
               'b1_measurements': one, 'throughput_measurements': throughput,
               'parity_checks': checks, 'checkpoint_audit': audit, 'input_files': input_files,
               'observed_gpu_floating_dtypes': observed_dtypes,
               'protocol': {'version': 'retrieval-efficiency-v2-deployment-wall',
                            'precision': 'FP32 parameters/forward; TF32 disabled',
                            'accuracy_precision_changed': False,
                            'scope': 'GPU-resident image to normalized retrieval descriptor; excludes IO, transfers and gallery search',
                            'fps_timer': 'synchronized wall clock; includes CPU launch overhead of inference loop',
                            'latency_timer': 'perf_counter with CUDA synchronization before and after each B=1 forward',
                            'deployment_fusion': deployment['policy'],
                            'memory_scope': 'PyTorch peak allocated, deployment model plus input and intermediates, warm steady state',
                            'cudnn_benchmark': False, 'warmup': a.warmup,
                            'iterations_per_round': a.iterations, 'rounds': a.repeats,
                            'torch_compile': False, 'horizontal_flip': False, 'multiscale': False},
               'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                               'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(),
                               'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                               'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw', '--format=csv,noheader'], text=True)},
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'completed_utc': datetime.now(timezone.utc).isoformat()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
    print(json.dumps({k: payload[k] for k in ['label', 'params_m', 'gflops', 'latency_b1_ms', 'fps_b16', 'peak_allocated_b1_mib']}), flush=True)


if __name__ == '__main__':
    main()
