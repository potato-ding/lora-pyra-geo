"""AnyVisLoc Scene_01/02 using the pinned official retrieval implementation."""
import traceback
import tempfile
from types import SimpleNamespace
import torch
from torch import nn
from . import anyvisloc_official as official
from .anyvisloc_data import AnyVisLocNPZDataset

SCENE_COUNTS = {"Scene_01": 1846, "Scene_02": 1833}
RETRIEVAL_KS = (1, 3, 5)
RETRIEVAL_COVER = 50
EVALUATION_BATCH_SIZE = 16
PATCH_SCALE = 1.0
PDM_LAMBDA = 6.0
PDM_ALPHA = 0.9


def official_protocol_metadata():
    return dict(batch_size=EVALUATION_BATCH_SIZE, upstream_batch_size=128, batch_size_override="user requested 16 for GPU memory", upstream_repository="https://github.com/UAV-AVL/Benchmark",
        upstream_commit=official.UPSTREAM_COMMIT,
        implementation="pinned official dataset, reference loaders, retrieval entry and aggregation",
        sample_loader="official AnyVisLocNPZDataset",
        reference_loader="official build_anyvisloc_reference_view including DSM",
        failure_policy="official retrieval failures recorded, contribute zero and remain in denominator",
        query_channel_path="RGB uint8 -> RGB float tensor -> BGR uint8 -> center crop -> RGB PIL",
        gallery_channel_path="BGR uint8 -> yaw rotation -> sliding crop -> OpenCV resize -> BGR to RGB -> PIL",
        query_channels="RGB", gallery_channels="RGB",
        normalization="torchvision ToTensor then ImageNet Normalize",
        inference="model.eval; preserve checkpoint evaluation precision",
        feature_normalization="official feature_fusion_all then F.normalize inside autocast",
        similarity="official full-gallery gf @ qf; NumPy argsort reversed")


def validate_anyvisloc_root(root):
    """Fix the released scene subset; sample discovery uses the official loader."""
    dataset = AnyVisLocNPZDataset(str(root), scenes=list(SCENE_COUNTS),
                                load_reference=False, cache_reference=True, recursive=True)
    scenes = {scene: [] for scene in SCENE_COUNTS}
    for path, scene in zip(dataset.samples, dataset.sample_scene):
        scenes[scene].append(path)
    for scene, count in SCENE_COUNTS.items():
        if len(scenes[scene]) != count:
            raise ValueError(f"AnyVisLoc {scene} has {len(scenes[scene])} samples; current official release has {count}")
    return scenes


class _OfficialRetrievalModel(nn.Module):
    """Adapt model outputs while preserving the checkpoint evaluation precision."""
    def __init__(self, encoder, device):
        super().__init__()
        from .model_loader import EvaluationEncoder
        self.native_encoder = isinstance(encoder, EvaluationEncoder)
        raw = encoder.model
        # Preserve restored parameter/buffer dtypes. Native encode() retains
        # its original BF16 autocast, even inside the upstream FP16 context.
        self.legacy_encoder = getattr(encoder, 'kind', None) in ('lpn', 'muse', 'infogeo_rd_trial')
        self.model = (encoder if self.native_encoder or self.legacy_encoder else raw).to(device).eval()
        self.kind = getattr(encoder, "kind", None)
        self.fusion = "descriptor" if self.legacy_encoder else ("MCCG" if self.kind == "mccg" else ("CAMP" if self.kind is not None else "descriptor"))
        parameter = next(raw.parameters(), torch.empty(0))
        self.runtime = dict(parameters_dtype=str(parameter.dtype).removeprefix("torch."),
            inference_autocast=("cuda_bfloat16" if self.native_encoder else ("disabled" if self.kind == "mccg" or (self.legacy_encoder and self.kind != "infogeo_rd_trial") else "cuda_float16")) if device.type == "cuda" else "disabled",
            source="checkpoint evaluation precision; no parameter dtype conversion")

    def forward(self, images):
        if self.legacy_encoder:
            return self.model.forward_unnormalized(images)
        if self.native_encoder:
            return self.model.encode(images)
        if self.kind == "mccg":
            # Override the benchmark outer FP16 context to preserve MCCG test.py FP32.
            with torch.autocast(images.device.type, enabled=False):
                return self.model(images, None)[0]
        return self.model(images)


def retrieval_metrics(ratios):
    metrics, rank = official.retrieval_metrics_for_ks(ratios, RETRIEVAL_KS, PDM_LAMBDA, PDM_ALPHA)
    metrics["retrieval_gt_rank"] = rank
    return metrics


def _evaluation_options(root):
    return SimpleNamespace(dataset_root=str(root), scenes=list(SCENE_COUNTS),
        reference_mode="aerial", pose_priori="yp", patch_scale=PATCH_SCALE,
        min_patch_size_m=0.0, max_patch_size_m=0.0, visualize=False,
        disable_ref_feature_cache=True, skip_existing=False,
        retrieval_ks=list(RETRIEVAL_KS), retrieval_k=max(RETRIEVAL_KS),
        pdm_lambda=PDM_LAMBDA, pdm_alpha=PDM_ALPHA)


@torch.no_grad()
def evaluate_anyvisloc(model, root, image_size, device, batch_size=EVALUATION_BATCH_SIZE, query_callback=None):
    """Official sample/reference loading, retrieval execution and failure accounting."""
    if batch_size != EVALUATION_BATCH_SIZE:
        raise ValueError("AnyVisLoc uses the fixed evaluation batch size 16")
    scenes = validate_anyvisloc_root(root)
    dataset = AnyVisLocNPZDataset(str(root), scenes=list(SCENE_COUNTS),
                                load_reference=False, cache_reference=True, recursive=True)
    expected_paths = [path for paths in scenes.values() for path in paths]
    if dataset.samples != expected_paths:
        raise RuntimeError("AnyVisLoc official dataset order changed after validation")
    retrieval_model = _OfficialRetrievalModel(model, device).eval()
    model.anyvisloc_runtime = dict(retrieval_model.runtime)
    opt = _evaluation_options(root)
    config = {"RETRIEVAL_SIZE": image_size, "BATCH_SIZE": batch_size,
              "DEVICE": str(device), "RETRIEVAL_FEATURE_NORM": True}
    method = {"retrieval_method": retrieval_model.fusion,
              "retrieval_cover": RETRIEVAL_COVER, "retrieval_model": retrieval_model,
              "img_transform": official.get_transforms_new()}
    method_name = retrieval_model.kind or ("native" if retrieval_model.native_encoder else "Sample4Geo")
    rows, failed_rows = [], []
    scene_rows = {scene: [] for scene in scenes}
    scene_failed = {scene: [] for scene in scenes}
    scene_processed = {scene: 0 for scene in scenes}
    with tempfile.TemporaryDirectory(prefix="anyvisloc_retrieval_") as work:
        for index in range(len(dataset)):
            # Upstream main also loads dataset[idx] outside the retrieval try:
            # malformed NPZ/decode errors are not counted as retrieval failures.
            sample = dataset[index]
            scene = sample["scene_name"]
            try:
                row = official.process_one_sample(sample, opt, config, method, method_name, work)
            except Exception as exc:
                row = dict(scene_name=scene, sample_id=sample["sample_id"],
                    npz_path=sample["npz_path"], reference_mode=opt.reference_mode,
                    retrieval_method=method_name, error=repr(exc), traceback=traceback.format_exc())
                failed_rows.append(row)
                scene_failed[scene].append(row)
                print(f"[Error] {scene}/{sample['sample_id']} {method_name}: {exc}", flush=True)
                callback_row = dict(row, scene=scene, status="failed")
            else:
                # Keep existing result aliases; all metrics and counts above
                # originate in the unmodified official single-query function.
                row.pop("result_dir", None)  # per-query JSON is temporary; results are retained below
                row.update(scene=scene, gallery_blocks=row["num_gallery_blocks"],
                    patch_height=row["patch_h"], patch_width=row["patch_w"],
                    retrieval_time_per_feature_ms=1000 * row["retrieval_time_s"])
                rows.append(row)
                scene_rows[scene].append(row)
                callback_row = dict(row, status="success")
            if query_callback is not None:
                query_callback(callback_row)
            scene_processed[scene] += 1
            count = scene_processed[scene]
            if count % 20 == 0 or count == len(scenes[scene]):
                print(f"AnyVisLoc {scene}: {count}/{len(scenes[scene])}", flush=True)
    results = {scene: _summarize(scene_rows[scene], scene_failed[scene], opt) for scene in scenes}
    for scene in scenes:
        results[scene]["scenes"] = [scene]
    results["overall"] = _summarize(rows, failed_rows, opt)
    results["per_query"] = rows
    results["failed"] = failed_rows
    return results


def _summarize(rows, failed_rows=None, opt=None):
    failed_rows = [] if failed_rows is None else failed_rows
    opt = _evaluation_options("") if opt is None else opt
    compatible_rows = [dict(row, retrieval_time_s=row.get("retrieval_time_s",
                       row.get("retrieval_time_per_feature_ms", 0.0) / 1000)) for row in rows]
    summary = official.summarize_retrieval(compatible_rows, failed_rows, opt)
    summary["queries"] = summary["num_attempted"]
    seconds = summary["mean_retrieval_time_s"]
    summary["mean_retrieval_time_per_feature_ms"] = None if seconds is None else 1000 * seconds
    return summary
