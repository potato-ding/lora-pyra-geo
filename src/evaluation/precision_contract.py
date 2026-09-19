"""Train-time selection and checkpoint reload share this runtime contract.

Selection inspects the live model; it never casts it. Reload restores the
recorded dtypes, including integer buffers, then requires an identical signature.
Legacy checkpoints without a signature can be loaded for supervision, but cannot
be certified as a new precision-consistent run.
"""
from contextlib import nullcontext
import torch
import torch.nn.functional as F

VERSION = "TRAIN_TEST_PRECISION_CONSISTENCY_V1"
PROFILES = {kind: {"parameter_dtype": "bfloat16", "autocast": "cuda_bfloat16",
                   "descriptor_dtype": "float32", "normalization_dtype": "float32",
                   "similarity_dtype": "float32"} for kind in ("teacher", "middle", "student")}

def unwrap(model):
    model = getattr(model, "module", model)
    return getattr(model, "student", model)

def dtype_name(tensor):
    return str(tensor.dtype).removeprefix("torch.")

def descriptor_postprocess(descriptor, eps=1e-12):
    return F.normalize(descriptor.float(), dim=-1, eps=eps)

def forward_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()

def inspect_precision_signature(model, model_type, image_size=224):
    model = unwrap(model)
    if model_type not in PROFILES:
        raise ValueError(model_type)
    params = {n: dtype_name(p) for n, p in model.named_parameters()}
    buffers = {n: dtype_name(b) for n, b in model.named_buffers()}
    bn = {n: dtype_name(b) for n, b in model.named_buffers()
          if n.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    lora = {n: d for n, d in params.items() if "lora_" in n}
    return dict(model_type=model_type, precision_contract_version=VERSION,
                image_size=int(image_size), architecture=type(model).__name__,
                parameter_dtypes=params, buffer_dtypes=buffers, bn_buffer_dtypes=bn,
                lora_dtypes=lora, lora_merge_state="unmerged" if lora else "not_applicable",
                state_shapes={n: list(t.shape) for n,t in model.state_dict().items()},
                descriptor_before_normalization="float32", descriptor_after_normalization="float32",
                final_cls_dtype="bfloat16" if model_type in ("teacher", "middle") else "not_applicable",
                evaluation_preprocessing="canonical_deterministic", horizontal_flip=False,
                selection_batch_size=8 if model_type=="teacher" else 32,
                **PROFILES[model_type])

def assert_precision_signature(actual, expected):
    if actual != expected:
        keys=sorted(k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k))
        raise RuntimeError("PRECISION_OR_RELOAD_CONTRACT_FAILURE: signature fields " + str(keys))

def selection_signature(model, model_type, image_size=224):
    signature=inspect_precision_signature(model, model_type, image_size)
    # Canonical training paths use DeepSpeed BF16 for all inference parameters.
    bad={n:d for n,d in signature['parameter_dtypes'].items() if d != 'bfloat16'}
    if bad:
        raise RuntimeError("PRECISION_OR_RELOAD_CONTRACT_FAILURE: live inference parameters " + str(bad))
    return signature

def apply_runtime_precision(model, model_type, expected=None, image_size=224):
    if model_type not in PROFILES:
        raise ValueError(model_type)
    if expected is None:
        model.bfloat16()
    if expected is not None:
        if expected.get('model_type')!=model_type or expected.get('precision_contract_version')!=VERSION:
            raise RuntimeError('PRECISION_OR_RELOAD_CONTRACT_FAILURE: unsupported profile')
        # Restore per-tensor runtime dtypes, never infer them from checkpoint storage.
        for field, tensors in [('parameter_dtypes',dict(model.named_parameters())),
                               ('buffer_dtypes',dict(model.named_buffers()))]:
            if set(tensors)!=set(expected[field]):
                raise RuntimeError('PRECISION_OR_RELOAD_CONTRACT_FAILURE: tensor schema')
            for name,tensor in tensors.items():
                target=getattr(torch,expected[field][name])
                if tensor.dtype != target:
                    tensor.data=tensor.data.to(dtype=target)
        assert_precision_signature(inspect_precision_signature(model,model_type,image_size),expected)
    return dict(PROFILES[model_type], precision_contract_version=VERSION,
                train_selection_signature_verified=expected is not None)

def assert_best_reload_metrics(actual, expected):
    for direction in ('D2S','S2D'):
        for metric in ('R@1','R@5','AP'):
            if float(actual[direction][metric]) != float(expected[direction][metric]):
                raise RuntimeError('PRECISION_OR_RELOAD_CONTRACT_FAILURE: '+direction+' '+metric)
    return 'PASS'

def flat_selection_metrics(metrics):
    if 'D2S' in metrics:
        return {d:{k:float(metrics[d]['mAP'] if k=='AP' and 'AP' not in metrics[d] else metrics[d][k]) for k in ('R@1','R@5','AP')} for d in ('D2S','S2D')}
    result={}
    for d in ('D2S','S2D'):
        result[d]={}
        for metric,aliases in [('R@1',('R1','R@1')),('R@5',('R5','R@5')),('AP',('AP','mAP'))]:
            key=next((d+'_'+a for a in aliases if d+'_'+a in metrics),None)
            if key is None:raise RuntimeError('Missing selection metric '+d+' '+metric)
            result[d][metric]=float(metrics[key])
    return result
