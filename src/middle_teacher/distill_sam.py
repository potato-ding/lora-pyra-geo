"""Distillation-aware SAM over the canonical M2 objective (ZeRO-2).

Search uses autograd.grad on the unwrapped, replicated model: leaf .grad and
ZeRO post-accumulation hooks are not populated. Each branch is explicitly
SUM/world synchronized in FP32. Only the second backward enters DeepSpeed.
"""
import hashlib
import torch
import torch.distributed as dist
from .sam_rng import capture_rng_state, restore_rng_state

FRAMEWORK = 'M2_DISTILL_SAM_V1'


def validate_sharpness(config, allow_blocked=False):
    """Accept only the paper E3 KD-guided Standard SAM contract."""
    sam = config['sam']
    expected = dict(enabled=True, framework=FRAMEWORK, sharpness_mode='sam',
                    search_direction='kd', perturb_scope='all_trainable',
                    adaptive=False, rho=0.10, second_pass_objective='full',
                    same_batch=True, rng_replay=True, norm_epsilon=1e-12)
    for key, value in expected.items():
        if sam.get(key) != value:
            raise ValueError('E3 SAM contract changed: ' + key)
    for key in ('balanced_task_weight', 'balanced_kd_weight'):
        if key in sam and sam[key] != .5:
            raise ValueError('Legacy E3 metadata changed: ' + key)
    if config['seed'] != 0 or config['checkpoint'].get('save_last') is not False:
        raise ValueError('E3 requires seed0 and best-only artifacts')
    return True

def parameter_spaces(model):
    """Explicit deployment classification for MiddleTeacherModel.forward.

    Deployment calls backbone only, final CLS then FP32 L2. The unmasked
    forward multiplies mask_token by zero: it is retained state but has no
    retrieval function. logit_scale and the ABV2 bank are training-only.
    Unknown trainable modules fail closed; classification is audit-only.
    """
    all_params = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
    recipient, excluded = [], []
    for name, param in all_params:
        if name.startswith('backbone.') and name != 'backbone.model.mask_token':
            recipient.append((name, param))
        elif name == 'logit_scale' or name == 'backbone.model.mask_token' or name.startswith('layer_semantic_projectors.'):
            excluded.append((name, param))
        else:
            raise ValueError('Unaudited trainable parameter: ' + name)
    return all_params, recipient, excluded


def vector_norm(values):
    return torch.stack([v.float().square().sum() for v in values]).sum().sqrt()


def synchronized_gradients(loss, named, retain_graph=False):
    grads = torch.autograd.grad(loss, [p for _, p in named], retain_graph=retain_graph, allow_unused=True)
    values = []
    for (_, p), g in zip(named, grads):
        v = torch.zeros_like(p, dtype=torch.float32) if g is None else g.detach().float().clone()
        if dist.is_initialized():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            v.div_(dist.get_world_size())
        values.append(v)
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError('Nonfinite search branch gradient')
    return values


def make_direction(task, kd, options, named=None):
    """E3 search is the synchronized KD gradient, normalized in FP32."""
    eps = options['norm_epsilon']
    nt, nk = vector_norm(task), vector_norm(kd)
    dot = torch.stack([(a*b).sum() for a, b in zip(task, kd)]).sum()
    cosine = dot/(nt*nk+eps)
    search = kd
    norm = vector_norm(search)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite KD search direction')
    perturbations = [options['rho']*g/(norm+eps) for g in search]
    pn = vector_norm(perturbations)
    expected = options['rho']*norm/(norm+eps)
    if not torch.isclose(pn, expected, rtol=2e-5, atol=1e-7):
        raise RuntimeError('SAM radius normalization failed')
    stats = dict(task_grad_norm=float(nt), kd_grad_norm=float(nk),
                 task_kd_cosine=float(cosine),
                 search_grad_norm_before_radius_normalization=float(norm),
                 perturb_norm=float(pn), euclidean_perturb_norm=float(pn),
                 rho=options['rho'], search_mode='kd',
                 perturb_scope='all_trainable', adaptive=False)
    return search, perturbations, stats

def rank_vector_audit(named, values):
    """Full-vector hash AND parameterwise rank-0 maximum absolute difference."""
    digest = hashlib.sha256()
    maximum = torch.zeros((), device=values[0].device, dtype=torch.float32)
    for (name, _), value in zip(named, values):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        if dist.is_initialized():
            ref = value.clone()
            dist.broadcast(ref, src=0)
            maximum = torch.maximum(maximum, (ref-value).abs().max())
    record = dict(rank=dist.get_rank() if dist.is_initialized() else 0,
                  sha256=digest.hexdigest(), norm=float(vector_norm(values)), max_abs_diff=float(maximum))
    records = [record]
    if dist.is_initialized():
        records = [None]*dist.get_world_size()
        dist.all_gather_object(records, record)
    if any(x['max_abs_diff'] > 1e-7 for x in records):
        raise RuntimeError('Rank search/perturbation mismatch')
    return records


class PerturbedParameters:
    def __init__(self, named, perturbations):
        self.named, self.perturbations = named, perturbations
        self.original = []
        self.restore_exact = False

    def __enter__(self):
        actual = []
        with torch.no_grad():
            for (_, p), epsilon in zip(self.named, self.perturbations):
                original = p.detach().clone()
                self.original.append(original)
                # Sum in FP32, round once to the canonical parameter dtype.
                p.copy_((original.float()+epsilon).to(p.dtype))
                actual.append((p.float()-original.float()).square().sum())
        self.applied_parameter_delta_norm = float(torch.stack(actual).sum().sqrt())
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            for (_, p), original in zip(self.named, self.original):
                p.copy_(original)
            self.restore_exact = all(torch.equal(p, old) for (_, p), old in zip(self.named, self.original))
        if not self.restore_exact:
            raise RuntimeError('Parameter restore drift')
        self.original.clear()


class GradientSummary:
    def __init__(self):
        self.rows = []

    def add(self, stats):
        self.rows.append(tuple(stats[k] for k in ('task_grad_norm', 'kd_grad_norm', 'task_kd_cosine')))

    def result(self):
        if not self.rows:
            raise RuntimeError('No gradient diagnostic steps')
        return dict(mean_task_grad_norm=sum(x[0] for x in self.rows)/len(self.rows),
                    mean_kd_grad_norm=sum(x[1] for x in self.rows)/len(self.rows),
                    mean_cosine=sum(x[2] for x in self.rows)/len(self.rows),
                    negative_cosine_fraction=sum(x[2]<0 for x in self.rows)/len(self.rows),
                    steps=len(self.rows))


def canonical_objective(forward, model, kd, images, ids, step):
    from .fchain_train import r0_pair_loss
    from src.utils.gather_features_and_labels_and_views import GatherLayer, concat_all_gather
    hidden = forward(images, return_layer_features=True)
    descriptor = hidden['final_descriptor']
    pairs = kd.local_pair_batch
    global_pairs = pairs * dist.get_world_size()
    assert descriptor.dtype == torch.float32 and tuple(descriptor.shape) == (2 * pairs, 768)
    md = torch.cat(GatherLayer.apply(descriptor[:pairs]), 0)
    ms = torch.cat(GatherLayer.apply(descriptor[pairs:]), 0)
    global_ids = concat_all_gather(ids)
    assert md.shape == ms.shape == (global_pairs, 768) and global_ids.unique().numel() == global_pairs
    task, d2s, s2d = r0_pair_loss(md, ms, model.logit_scale)
    full, stats, kd_loss = kd.compose_all(task, md, ms, images, global_ids, model, step, hidden,
                                        return_kd_objective=True)
    if not bool(torch.isfinite(full)):
        raise FloatingPointError('Nonfinite full objective')
    return full, task, kd_loss, stats, d2s, s2d


def sam_backward(engine, kd, images, ids, step, options, audit_ranks=False):
    """Exactly two forwards; only second full backward reaches the optimizer.

    Returns after exact restore, before the caller's canonical engine.step().
    Diagnostics use two autograd traversals of the same first-pass graph.
    """
    model = engine.module
    all_named, _, _ = parameter_spaces(model)
    named = all_named
    if any(p.grad is not None for _, p in all_named):
        raise RuntimeError('Unexpected gradients at start of SAM batch')
    if any(p.requires_grad or p.grad is not None for p in kd.teacher.parameters()):
        raise RuntimeError('Teacher must be frozen')
    state = capture_rng_state()
    batch_identity = (images.data_ptr(), images._version, ids.data_ptr(), ids._version)
    full, task, kd_loss, _, _, _ = canonical_objective(model, model, kd, images, ids, step)
    first_loss = float(full.detach())
    task_grad = synchronized_gradients(task, named, retain_graph=True)
    kd_grad = synchronized_gradients(kd_loss, named)
    if any(p.grad is not None for _, p in all_named):
        raise RuntimeError('First-pass autograd unexpectedly accumulated parameter gradients')
    search, perturbations, stats = make_direction(task_grad, kd_grad, options, named=named)
    if audit_ranks:
        stats['task_rank_audit'] = rank_vector_audit(named, task_grad)
        stats['kd_rank_audit'] = rank_vector_audit(named, kd_grad)
        stats['search_rank_audit'] = rank_vector_audit(named, search)
        stats['epsilon_rank_audit'] = rank_vector_audit(named, perturbations)
    del full, task, kd_loss, task_grad, kd_grad, search
    engine.zero_grad()
    restore_rng_state(state)
    with PerturbedParameters(named, perturbations) as perturbed:
        second, base, kd2, kd_stats, d2s, s2d = canonical_objective(engine, model, kd, images, ids, step)
        engine.backward(second)
    assert batch_identity == (images.data_ptr(), images._version, ids.data_ptr(), ids._version)
    assert all(not p.requires_grad and p.grad is None for p in kd.teacher.parameters())
    stats.update(first_loss=first_loss, second_loss=float(second.detach()),
                 perturb_restore_exact=perturbed.restore_exact,
                 applied_parameter_delta_norm=perturbed.applied_parameter_delta_norm,
                 teacher_logical_forward_count=2, same_batch=True, rng_replay=True,
                 first_pass_gradient_leak=False, second_pass_objective='full')
    return second, base, d2s, s2d, kd_stats, stats
