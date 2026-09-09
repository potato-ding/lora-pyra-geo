"""Single-traversal Teacher feature extraction for Adaptive Bridge v1.

The formal T0 wrapper computes its canonical descriptor with a complete
``get_intermediate_layers`` traversal.  Temporary block hooks capture the
selected normalized CLS tokens during that same traversal, so the canonical
descriptor and candidate hidden features always come from one logical pass.
"""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F


ADAPTIVE_FUSED_KEYS = (
    "layer32_cls",
    "layer34_cls",
    "layer36_cls",
    "layer38_cls",
    "final_cls",
)


def _normalized_cls(core, block_output):
    if not torch.is_tensor(block_output) or block_output.ndim != 3:
        raise RuntimeError(
            "teacher block hook expected a [batch,tokens,dim] tensor, got "
            f"{type(block_output).__name__}"
        )
    raw_cls = block_output[:, 0]
    if getattr(core, "untie_cls_and_patch_norms", False):
        return core.cls_norm(raw_cls)
    return core.norm(raw_cls)


def _normalized_patches(core, block_output):
    storage_tokens = int(getattr(core, "n_storage_tokens", 0))
    patches = block_output[:, storage_tokens + 1 :]
    if getattr(core, "untie_cls_and_patch_norms", False):
        return core.patch_norm(patches)
    return core.norm(patches)


def _sync_if_cuda(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


@torch.no_grad()
def adaptive_teacher_fused_forward(
    teacher,
    source_images,
    *,
    chunk_size=4,
    teacher_layers=(32, 34, 36, 38),
    return_patch_tokens=False,
    extra_patch_layers=(),
    collect_timing=None,
):
    """Return selected normalized CLS features and FINAL_CLS in one traversal.

    ``teacher(chunk)`` remains the only forward entry.  Hooks are registered on
    the existing formal T0 blocks and removed in ``finally``.  Consequently the
    returned ``final_cls`` is the exact formal wrapper descriptor, never the
    Block38 hidden state.
    """
    selected = tuple(int(layer) for layer in teacher_layers)
    extra_patches = tuple(int(layer) for layer in extra_patch_layers)
    hook_layers = tuple(dict.fromkeys(selected + extra_patches))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Adaptive fused forward requires unique non-empty layers")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    if collect_timing is None:
        collect_timing = os.environ.get("ADAPTIVE_FUSED_DEBUG_TIMING", "0") == "1"

    core = teacher.backbone.model
    if not hasattr(core, "blocks"):
        raise AttributeError("formal T0 backbone does not expose Transformer blocks")
    if max(hook_layers) >= len(core.blocks):
        raise ValueError(
            f"requested Block{max(selected)} but Teacher has {len(core.blocks)} blocks"
        )

    per_layer = {layer: [] for layer in selected}
    per_layer_patches = {layer: [] for layer in hook_layers}
    final_chunks = []
    active_capture = {}
    hidden_capture_seconds = 0.0
    forward_seconds = 0.0
    final_descriptor_seconds = 0.0

    def make_hook(layer):
        def capture(_module, _inputs, output):
            nonlocal hidden_capture_seconds
            started = time.perf_counter()
            if layer in active_capture:
                raise RuntimeError(f"Block{layer} executed more than once in one Teacher pass")
            capture = {"cls": _normalized_cls(core, output).detach()}
            if return_patch_tokens or layer in extra_patches:
                capture["patch"] = _normalized_patches(core, output).detach()
            active_capture[layer] = capture
            hidden_capture_seconds += time.perf_counter() - started
        return capture

    handles = [
        core.blocks[layer].register_forward_hook(make_hook(layer))
        for layer in hook_layers
    ]
    if collect_timing:
        _sync_if_cuda(source_images)
    total_started = time.perf_counter()
    all_forwards_started = total_started
    try:
        for start in range(0, source_images.size(0), int(chunk_size)):
            chunk = source_images[start : start + int(chunk_size)].to(
                dtype=torch.bfloat16
            )
            active_capture.clear()
            descriptor = teacher(chunk)
            descriptor_started = time.perf_counter()
            if isinstance(descriptor, dict):
                descriptor = descriptor.get(
                    "final_descriptor", descriptor.get("descriptor")
                )
            if not torch.is_tensor(descriptor):
                raise RuntimeError("formal T0 teacher did not return a descriptor tensor")
            missing = [layer for layer in hook_layers if layer not in active_capture]
            if missing:
                raise RuntimeError(f"teacher fused hook missed blocks: {missing}")
            for layer in selected:
                per_layer[layer].append(active_capture[layer]["cls"])
                if return_patch_tokens:
                    per_layer_patches[layer].append(active_capture[layer]["patch"])
            for layer in extra_patches:
                if layer not in selected:
                    per_layer_patches[layer].append(active_capture[layer]["patch"])
            final_chunks.append(descriptor.detach().float())
            final_descriptor_seconds += time.perf_counter() - descriptor_started
    finally:
        for handle in handles:
            handle.remove()

    if collect_timing:
        _sync_if_cuda(source_images)
        forward_seconds = time.perf_counter() - all_forwards_started

    features = {
        f"layer{layer}_cls": torch.cat(per_layer[layer], dim=0)
        for layer in selected
    }
    if return_patch_tokens:
        features.update(
            {
                f"layer{layer}_patch": torch.cat(per_layer_patches[layer], dim=0)
                for layer in selected
            }
        )
    for layer in extra_patches:
        features[f"layer{layer}_patch"] = torch.cat(per_layer_patches[layer], dim=0)
    features["final_cls"] = F.normalize(
        torch.cat(final_chunks, dim=0).float(), dim=1, eps=1e-6
    )
    expected = (source_images.size(0), 4096)
    required_cls_keys = tuple(f"layer{layer}_cls" for layer in selected) + ("final_cls",)
    if any(tuple(features[key].shape) != expected for key in required_cls_keys):
        raise RuntimeError(
            "adaptive fused feature shape mismatch: "
            f"{ {key: list(features[key].shape) for key in required_cls_keys} }"
        )
    if return_patch_tokens:
        for layer in selected:
            patch = features[f"layer{layer}_patch"]
            if patch.ndim != 3 or patch.shape[0] != source_images.size(0) or patch.shape[-1] != 4096:
                raise RuntimeError(f"Block{layer} patch feature shape mismatch: {list(patch.shape)}")
    finite_keys = required_cls_keys + tuple(
        f"layer{layer}_patch" for layer in hook_layers
        if return_patch_tokens or layer in extra_patches
    )
    if any(not torch.isfinite(features[key]).all() for key in finite_keys):
        raise FloatingPointError("adaptive fused Teacher features contain NaN/Inf")

    features["timing"] = {
        "teacher_forward_time": forward_seconds if collect_timing else None,
        "hidden_capture_time": hidden_capture_seconds if collect_timing else None,
        "final_descriptor_time": final_descriptor_seconds if collect_timing else None,
        "total_teacher_time": (
            time.perf_counter() - total_started if collect_timing else None
        ),
        "teacher_logical_forward_calls": 1,
        "teacher_physical_chunk_forwards": int(
            math.ceil(source_images.size(0) / int(chunk_size))
        ),
        "debug_timing_enabled": bool(collect_timing),
        "patch_tokens_captured": bool(return_patch_tokens),
    }
    return features
