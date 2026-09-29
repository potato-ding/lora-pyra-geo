"""AnyVisLoc Scene_01/02 retrieval protocol for the unified evaluator.

Geometry, gallery filtering, and metrics follow UAV-AVL/Benchmark's
run_avl_retrieval_only.py and avl_utils.py. Model preprocessing belongs to the
evaluated checkpoint, as it does for the other datasets in this repository.
"""

import json
import math
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image


SCENE_COUNTS = {"Scene_01": 1846, "Scene_02": 1833}
RETRIEVAL_KS = (1, 3, 5)
RETRIEVAL_COVER = 50
PATCH_SCALE = 1.0
PDM_LAMBDA = 6.0
PDM_ALPHA = 0.9


def validate_anyvisloc_root(root):
    """Require the current released aerial-only Scene_01/02 subset."""
    root = Path(root)
    scenes = {}
    for scene, count in SCENE_COUNTS.items():
        directory = root / scene
        scene_id = int(scene.rsplit("_", 1)[1])
        reference = directory / f"L{scene_id:02d}_reference.json"
        if not reference.is_file():
            raise FileNotFoundError(reference)
        with reference.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("scene_name") != scene or metadata.get("scene_id") != scene_id:
            raise ValueError(f"AnyVisLoc reference identity mismatch: {reference}")
        aerial = metadata.get("modes", {}).get("aerial", {})
        map_path = directory / str(aerial.get("map_path", ""))
        if not map_path.is_file() or map_path.suffix.lower() != ".png":
            raise FileNotFoundError(f"AnyVisLoc aerial map missing: {map_path}")
        resolution = np.asarray(aerial.get("map_resolution"), dtype=np.float64).reshape(-1)
        origin = np.asarray(aerial.get("map_origin_local"), dtype=np.float64).reshape(-1)
        if resolution.size != 2 or origin.size != 2 or not np.isfinite(resolution).all() or not np.isfinite(origin).all() or (resolution <= 0).any():
            raise ValueError(f"AnyVisLoc map geometry invalid: {reference}")
        samples = sorted(directory.glob(f"L{scene_id:02d}_????.npz"))
        if len(samples) != count:
            raise ValueError(f"AnyVisLoc {scene} has {len(samples)} samples; current official release has {count}")
        scenes[scene] = (map_path, resolution, origin, samples)
    return scenes


def _rotate_map(image, yaw):
    """Official dumpRotateImage: yaw rotation about the map center."""
    radians = float(yaw) / 180.0 * np.pi
    height, width = image.shape[:2]
    height_new = int(width * abs(np.sin(radians)) + height * abs(np.cos(radians)))
    width_new = int(height * abs(np.sin(radians)) + width * abs(np.cos(radians)))
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), float(yaw), 1)
    matrix[0, 2] += (width_new - width) // 2
    matrix[1, 2] += (height_new - height) // 2
    return cv2.warpAffine(image, matrix, (width_new, height_new), borderValue=(0, 0, 0)), matrix


def _gallery_centers(image, patch_h, patch_w):
    """Official compute_block_mid_wo_black, including its 1/10 mask grid."""
    height, width = image.shape[:2]
    step_h = max(1, int(patch_h * (100 - RETRIEVAL_COVER) / 100))
    step_w = max(1, int(patch_w * (100 - RETRIEVAL_COVER) / 100))
    small = cv2.resize(image, (int(width / 10), int(height / 10)), interpolation=cv2.INTER_NEAREST)
    total = patch_h * patch_w / 100
    mids = []
    for i in range(len(range(0, height, step_h))):
        for j in range(len(range(0, width, step_w))):
            row0 = min(i * step_h, height - patch_h - 1)
            col0 = min(j * step_w, width - patch_w - 1)
            block = small[int(row0 / 10):int((row0 + patch_h) / 10), int(col0 / 10):int((col0 + patch_w) / 10)]
            if np.sum(block[:, :, 0] > 0) >= total / 5 * 2:
                mids.append(((row0 + row0 + patch_h) / 2, (col0 + col0 + patch_w) / 2))
    if not mids:
        raise RuntimeError("AnyVisLoc: no valid aerial gallery blocks")
    return np.asarray(mids, dtype=np.float64)


def _even_ceil(value):
    result = max(2, int(math.ceil(float(value))))
    return result + result % 2


def _geometry(image, metadata, map_resolution, map_origin):
    """Official yp-prior view center, footprint, and clipped even patch size."""
    xyz = metadata["xyz"].astype(np.float64).reshape(3)
    euler = metadata["euler_deg"].astype(np.float64).reshape(3)
    intrinsics = metadata["K"].astype(np.float64).reshape(3, 3)
    image_size = metadata["image_size"].astype(np.int32).reshape(2)
    pitch, yaw = float(euler[1]), float(euler[2])
    x, y, altitude = map(float, xyz)
    tangent = np.tan(pitch / 180.0 * np.pi)
    if np.isfinite(tangent) and abs(float(tangent)) >= 1e-6:
        x += altitude * float(tangent) * np.sin(yaw / 180.0 * np.pi)
        y -= altitude * float(tangent) * np.cos(yaw / 180.0 * np.pi)
    col = (x - float(map_origin[0])) / max(float(map_resolution[0]), 1e-12)
    row = (y - float(map_origin[1])) / max(float(map_resolution[1]), 1e-12)
    rotated, matrix = _rotate_map(image, yaw)
    center_col, center_row = matrix @ np.asarray([col, row, 1.0], dtype=np.float32)
    focal = max(float(intrinsics[0, 0]), 1e-6)
    denominator = max(abs(float(np.sin(np.pi * (90.0 + pitch) / 180.0))), 1e-6)
    drone_resolution = max(altitude, 1e-6) / denominator / focal
    footprint = float(min(image_size)) * drone_resolution * PATCH_SCALE
    patch_h = _even_ceil(footprint / max(float(map_resolution[1]), 1e-12))
    patch_w = _even_ceil(footprint / max(float(map_resolution[0]), 1e-12))
    patch_h = min(patch_h, max(2, rotated.shape[0] - 2))
    patch_w = min(patch_w, max(2, rotated.shape[1] - 2))
    if patch_h % 2:
        patch_h = max(2, patch_h - 1)
    if patch_w % 2:
        patch_w = max(2, patch_w - 1)
    return rotated, (float(center_col), float(center_row)), patch_h, patch_w


def retrieval_metrics(ratios):
    """Official per-query Recall@K and PDM@K from ranked normalized distances."""
    ratios = np.asarray(ratios, dtype=np.float64).reshape(-1)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        raise RuntimeError("AnyVisLoc: no finite gallery distances")
    rank = int(np.argmin(ratios)) + 1
    result = {}
    for k in RETRIEVAL_KS:
        effective_k = min(k, len(ratios))
        weights = np.arange(effective_k, 0, -1, dtype=np.float64)
        logits = np.clip(PDM_LAMBDA * (ratios[:effective_k] - PDM_ALPHA), -60.0, 60.0)
        pdm = float(np.sum(weights / (1.0 + np.exp(logits))) / np.sum(weights))
        result[f"Recall@{k}"] = float(rank <= effective_k)
        result[f"PDM@{k}"] = pdm
    result["retrieval_gt_rank"] = rank
    return result


def _query_tensor(image_rgb, image_size, normalization):
    # The official query path resizes an RGB PIL image without an explicit
    # filter; Pillow uses BICUBIC for RGB. Keep the evaluated model's size.
    resized = Image.fromarray(image_rgb).resize(
        (image_size, image_size), resample=Image.Resampling.BICUBIC
    )
    return normalization(image=np.asarray(resized))["image"]


def _gallery_tensor(image_bgr, image_size, normalization):
    # The official gallery path uses cv2.resize's default INTER_LINEAR.
    # Convert BGR to RGB for this repository's RGB-trained encoders.
    resized = cv2.resize(
        image_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    )
    return normalization(image=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))["image"]


@torch.no_grad()
def evaluate_anyvisloc(model, root, image_size, device, batch_size=16):
    """Evaluate all released queries from Scene_01/02 against aerial tiles."""
    scenes = validate_anyvisloc_root(root)
    normalization = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    results = {}
    all_rows = []
    for scene, (map_path, resolution, origin, samples) in scenes.items():
        aerial = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if aerial is None:
            raise RuntimeError(f"Cannot decode AnyVisLoc aerial map: {map_path}")
        rows = []
        for index, sample_path in enumerate(samples, start=1):
            with np.load(sample_path, allow_pickle=False) as record:
                required = ("image", "K", "image_size", "xyz", "euler_deg", "sample_id", "scene_id")
                missing = set(required) - set(record.files)
                if missing:
                    raise ValueError(f"AnyVisLoc sample missing {sorted(missing)}: {sample_path}")
                sample = {key: record[key] for key in required}
            if int(sample["scene_id"]) != int(scene.rsplit("_", 1)[1]) or str(sample["sample_id"].item()) != sample_path.stem:
                raise ValueError(f"AnyVisLoc sample identity mismatch: {sample_path}")
            query_rgb = sample["image"]
            if query_rgb.ndim != 3 or query_rgb.shape[2] != 3 or query_rgb.dtype != np.uint8:
                raise ValueError(f"AnyVisLoc unsupported UAV image: {sample_path}")
            rotated, center, patch_h, patch_w = _geometry(aerial, sample, resolution, origin)
            mids = _gallery_centers(rotated, patch_h, patch_w)
            side = min(query_rgb.shape[:2])
            cy, cx = query_rgb.shape[0] // 2, query_rgb.shape[1] // 2
            query_crop = query_rgb[cy - side // 2:cy + side // 2, cx - side // 2:cx + side // 2]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            query = _query_tensor(query_crop, image_size, normalization).unsqueeze(0).to(device)
            q_feature = model.encode(query).squeeze(0)
            scores = []
            for start in range(0, len(mids), batch_size):
                images = []
                for mid_row, mid_col in mids[start:start + batch_size]:
                    row0 = int(mid_row - patch_h / 2)
                    col0 = int(mid_col - patch_w / 2)
                    crop_bgr = rotated[row0:row0 + patch_h, col0:col0 + patch_w]
                    if crop_bgr.shape[:2] != (patch_h, patch_w):
                        raise RuntimeError(f"AnyVisLoc invalid gallery crop: {sample_path}")
                    images.append(_gallery_tensor(crop_bgr, image_size, normalization))
                gallery = model.encode(torch.stack(images).to(device))
                scores.extend((gallery @ q_feature).cpu().numpy().tolist())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            order = np.argsort(np.asarray(scores))[::-1]
            ranked = mids[order]
            distances = np.hypot((center[0] - ranked[:, 1]) * float(resolution[0]),
                                 (center[1] - ranked[:, 0]) * float(resolution[1]))
            ratios = distances / max(float(resolution[0]) * patch_w, 1e-12)
            metrics = retrieval_metrics(ratios)
            rows.append({"sample_id": sample_path.stem, "scene": scene,
                         "gallery_blocks": len(mids), "patch_height": patch_h, "patch_width": patch_w,
                         "retrieval_time_per_feature_ms": 1000 * elapsed / (len(mids) + 1), **metrics})
            if index % 20 == 0 or index == len(samples):
                print(f"AnyVisLoc {scene}: {index}/{len(samples)}", flush=True)
            del rotated
        all_rows.extend(rows)
        results[scene] = _summarize(rows)
        del aerial
    results["overall"] = _summarize(all_rows)
    results["per_query"] = all_rows
    return results


def _summarize(rows):
    if not rows:
        raise RuntimeError("AnyVisLoc: no successful queries")
    metrics = (f"{name}@{k}" for name in ("Recall", "PDM") for k in RETRIEVAL_KS)
    return {"queries": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in metrics},
            "mean_retrieval_time_per_feature_ms": float(np.mean([row["retrieval_time_per_feature_ms"] for row in rows]))}
