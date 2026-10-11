"""Unmodified retrieval functions from UAV-AVL/Benchmark.

Pinned upstream commit: 6c8f7980f35defd0676d82e71b19fcd214da4033
Source: https://github.com/UAV-AVL/Benchmark
Only the retrieval dependency closure is included; matching/PnP are not used.
"""
from pathlib import Path
import math,time,json,hashlib,re,os,pickle
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision import transforms

UPSTREAM_COMMIT = "6c8f7980f35defd0676d82e71b19fcd214da4033"


def dumpRotateImage(img, degree):
    """Rotate an image around its center and return the rotated image plus affine matrix."""
    radians = degree / 180 * np.pi
    height, width = img.shape[:2]
    height_new = int(width * abs(np.sin(radians)) + height * abs(np.cos(radians)))
    width_new = int(height * abs(np.sin(radians)) + width * abs(np.cos(radians)))
    mat_rotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    mat_rotation[0, 2] += (width_new - width) // 2
    mat_rotation[1, 2] += (height_new - height) // 2
    img_rotation = cv2.warpAffine(
        img,
        mat_rotation,
        (width_new, height_new),
        borderValue=(0, 0, 0),
    )
    return img_rotation, mat_rotation

def compute_block_mid_wo_black(image, block_size, step_size):
    """Return valid reference block centers, filtering blocks with too much black area."""
    x_size, y_size, _ = image.shape
    small_img = cv2.resize(
        image,
        (int(y_size / 10), int(x_size / 10)),
        interpolation=cv2.INTER_NEAREST,
    )

    num_blocks_x = len(range(0, x_size, step_size[0]))
    num_blocks_y = len(range(0, y_size, step_size[1]))
    mids = []
    total_pixels = block_size[0] * block_size[1] / 100

    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            start_x = i * step_size[0]
            start_y = j * step_size[1]
            start_x = min(start_x, x_size - block_size[0] - 1)
            start_y = min(start_y, y_size - block_size[1] - 1)
            end_x = start_x + block_size[0]
            end_y = start_y + block_size[1]

            block = small_img[
                int(start_x / 10): int(end_x / 10),
                int(start_y / 10): int(end_y / 10),
            ]
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2

            count_above_threshold = np.sum(block[:, :, 0] > 0)
            if count_above_threshold >= total_pixels / 5 * 2:
                mids.append([mid_x, mid_y])

    return np.array(mids)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def to_python_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(x)

def normalize_uint8(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    lo, hi = np.percentile(x[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    y = np.clip((x.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)

def ensure_bgr_uint8(img):
    """Reference maps are stored/read in BGR order for OpenCV."""
    img = np.asarray(img)
    if img.ndim == 2:
        return cv2.cvtColor(normalize_uint8(img), cv2.COLOR_GRAY2BGR)
    if img.ndim != 3:
        raise ValueError(f"map image must be [H,W,3] or [H,W], got {img.shape}")
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return normalize_uint8(img)

def as_vec2(x, name, default=None):
    if x is None:
        if default is None:
            raise KeyError(f"missing {name}")
        x = default
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} must have at least 2 values, got {arr}")
    return arr[:2].astype(np.float32)

def tensor_rgb_to_bgr_uint8(x):
    """AnyVisLocNPZDataset returns RGB tensor [3,H,W] in [0,1]."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 1)
        x = (x * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def crop_center_from_bgr(img_bgr, width, height):
    """Return center crop as RGB PIL image for retrieval transforms."""
    h0, w0 = img_bgr.shape[:2]
    width = int(min(width, w0))
    height = int(min(height, h0))
    cx, cy = w0 // 2, h0 // 2
    left = max(0, int(cx - width // 2))
    top = max(0, int(cy - height // 2))
    right = min(w0, int(cx + width // 2))
    bottom = min(h0, int(cy + height // 2))
    crop_bgr = img_bgr[top:bottom, left:right]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)

def anyvisloc_sample_to_truepos(sample):
    """Build a small metadata dict compatible with the baseline naming style."""
    image_size = to_numpy(sample.get("image_size", np.array([0, 0]))).astype(np.int32).reshape(-1)
    if image_size.size >= 2 and image_size[0] > 0:
        h, w = int(image_size[0]), int(image_size[1])
    else:
        img = sample["image"]
        h, w = int(img.shape[-2]), int(img.shape[-1])
    xyz = to_numpy(sample["xyz"]).astype(np.float32).reshape(3)
    euler = to_numpy(sample["euler_deg"]).astype(np.float32).reshape(3)
    K = to_numpy(sample["K"]).astype(np.float32)
    return {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
        "rel_alt": float(xyz[2]),
        "roll": float(euler[0]),
        "pitch": float(euler[1]),
        "yaw": float(euler[2]),
        "width": int(w),
        "height": int(h),
        "K": K,
        "sample_id": to_python_str(sample.get("sample_id", "")),
        "scene_name": to_python_str(sample.get("scene_name", "")),
        "npz_path": to_python_str(sample.get("npz_path", "")),
    }

def _even_ceil(v, min_value=2):
    out = int(math.ceil(float(v)))
    out = max(int(min_value), out)
    if out % 2:
        out += 1
    return out

def resolution_size_anyvisloc(data, opt):
    """Estimate UAV ground sampling distance from K, altitude, and pitch.

    This follows the same geometry used by the previous working
    AnyVisLoc_open_npz_new.py implementation:

        denom = abs(sin(90 deg + pitch))
        ground_dist = z / denom
        drone_resolution = ground_dist / fx

    Patch scaling is intentionally NOT folded into this return value, because
    ``drone_resolution`` is also used as the RoMa fine-scale factor. Patch
    scaling is applied later when computing the retrieval patch size.
    """
    w = int(data["width"])
    h = int(data["height"])
    K = np.asarray(data["K"], dtype=np.float32)
    fx = max(float(K[0, 0]) if K.size >= 9 else 1.0, 1e-6)
    z = max(float(data["rel_alt"]), 1e-6)

    if getattr(opt, "pose_priori", "yp") == "unknown":
        denom = 1.0
    else:
        pitch = float(data.get("pitch", 0.0))
        denom = abs(float(np.sin(np.pi * (90.0 + pitch) / 180.0)))
        denom = max(denom, 1e-6)

    ground_dist = z / denom
    drone_resolution = ground_dist / fx
    min_size = min(w, h)
    return float(drone_resolution), np.array([min_size, min_size], dtype=np.int32)

def _apply_affine_xy(x, y, mat):
    """Apply an OpenCV-style affine/homography matrix to one x/y pixel point."""
    M = np.asarray(mat, dtype=np.float32)
    if M.size == 6 and M.shape != (2, 3):
        M = M.reshape(2, 3)
    p = np.array([float(x), float(y), 1.0], dtype=np.float32)
    if M.shape == (2, 3):
        out = M @ p
        return float(out[0]), float(out[1])
    if M.shape == (3, 3):
        out = M @ p
        denom = float(out[2]) if abs(float(out[2])) > 1e-12 else 1.0
        return float(out[0] / denom), float(out[1] / denom)
    raise ValueError(f"matRotation must be 2x3 or 3x3, got {M.shape}")

def view_center_anyvisloc(data, map_resolution, map_origin_local, matRotation, use_shift=True):
    """Estimate UAV view center in the current reference-map pixel frame.

    This intentionally matches the previous working
    AnyVisLoc_open_npz_new.py::compute_view_center_xy() formula:

        dx = z * tan(pitch) * sin(yaw)
        dy = -z * tan(pitch) * cos(yaw)

    The shifted dataset-local point is then converted to selected-map col/row
    pixels and, if the map has already been yaw-rotated globally, transformed
    by the same affine matrix.
    """
    x = float(data["x"])
    y = float(data["y"])
    z = float(data["rel_alt"])

    if use_shift:
        pitch = float(data.get("pitch", 0.0))
        yaw = float(data.get("yaw", 0.0))
        tan_v = np.tan(pitch / 180.0 * np.pi)
        if np.isfinite(tan_v) and abs(float(tan_v)) >= 1e-6:
            yaw_rad = yaw / 180.0 * np.pi
            dx = z * float(tan_v) * np.sin(yaw_rad)
            dy = -z * float(tan_v) * np.cos(yaw_rad)
            if np.isfinite(dx) and np.isfinite(dy):
                x += float(dx)
                y += float(dy)

    res = as_vec2(map_resolution, "map_resolution")
    origin = as_vec2(map_origin_local, "map_origin_local", [0.0, 0.0])
    col = (x - float(origin[0])) / max(float(res[0]), 1e-12)
    row = (y - float(origin[1])) / max(float(res[1]), 1e-12)

    return _apply_affine_xy(col, row, matRotation)

def _sanitize_filename_token(text):
    text = to_python_str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "unknown"

def _circled_number(n):
    """Return circled number labels for visualization titles."""
    circled = {
        1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤",
        6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
        11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮",
        16: "⑯", 17: "⑰", 18: "⑱", 19: "⑲", 20: "⑳",
    }
    return circled.get(int(n), str(n))

def _draw_red_cross_if_inside(img_bgr, center_col, center_row, col0, row0):
    """Draw a red cross at the true view-center if it falls inside this patch.

    ``center_col``/``center_row`` and ``col0``/``row0`` must be expressed in
    the same reference-map pixel frame. In yaw-prior mode this is the rotated
    map frame; otherwise it is the original selected reference-map frame.
    """
    img_bgr = np.asarray(img_bgr)
    if img_bgr.size == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    x_f = float(center_col) - float(col0)
    y_f = float(center_row) - float(row0)
    if not (np.isfinite(x_f) and np.isfinite(y_f)):
        return img_bgr

    x = int(round(x_f))
    y = int(round(y_f))
    if not (0 <= x < w and 0 <= y < h):
        return img_bgr

    out = img_bgr.copy()
    marker_size = max(12, int(round(min(w, h) * 0.070)))
    thickness = max(2, int(round(min(w, h) * 0.008)))
    cv2.drawMarker(
        out,
        (x, y),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )
    return out

def _draw_cross_at_image_center(img, color, marker_ratio=0.070, thickness_ratio=0.008):
    """Draw a crosshair at the geometric center of an image.

    The color tuple is interpreted in the image's native channel order. For
    BGR images use red=(0,0,255), green=(0,255,0). For RGB images use
    red=(255,0,0), green=(0,255,0).
    """
    img = np.asarray(img)
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    x = int(round((w - 1) * 0.5))
    y = int(round((h - 1) * 0.5))
    marker_size = max(12, int(round(min(w, h) * float(marker_ratio))))
    thickness = max(2, int(round(min(w, h) * float(thickness_ratio))))
    cv2.drawMarker(
        out,
        (x, y),
        tuple(int(c) for c in color),
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )
    return out

def _make_ref_feature_cache_meta(
    *,
    opt,
    method_dict,
    config,
    ref_image,
    block_size,
    step_size,
    mids,
):
    """Build strict metadata so cached gallery features are reused only safely."""
    mids_arr = np.ascontiguousarray(np.asarray(mids, dtype=np.float32).reshape(-1, 2))
    mids_hash = hashlib.sha1(mids_arr.tobytes()).hexdigest()[:16]
    return {
        "cache_version": "anyvisloc_npz_v5_ref_features",
        "scene_name": _sanitize_filename_token(getattr(opt, "_current_scene_name", "")),
        "reference_mode": _sanitize_filename_token(getattr(opt, "reference_mode", "")),
        "retrieval_method": _sanitize_filename_token(method_dict.get("retrieval_method", "")),
        "retrieval_size": int(config.get("RETRIEVAL_SIZE", 384)),
        "retrieval_cover": int(method_dict.get("retrieval_cover", -1)),
        "ref_shape": [int(x) for x in np.asarray(ref_image).shape[:2]],
        "block_size": [int(block_size[0]), int(block_size[1])],
        "step_size": [int(step_size[0]), int(step_size[1])],
        "num_blocks": int(mids_arr.shape[0]),
        "mids_hash": mids_hash,
    }

def _default_ref_feature_cache_root(save_path, opt):
    user_cache_dir = getattr(opt, "ref_feature_cache_dir", None)
    if user_cache_dir:
        return Path(user_cache_dir)
    save_path = Path(save_path)
    # save_path is normally combo_dir / scene_name / sample_id.
    # Put cache under combo_dir so different UAV samples in the same combo can share it.
    try:
        return save_path.parents[1] / "_ref_feature_cache"
    except Exception:
        return save_path.parent / "_ref_feature_cache"

def _make_ref_feature_cache_path(save_path, opt, meta):
    cache_root = _default_ref_feature_cache_root(save_path, opt)
    scene = _sanitize_filename_token(meta.get("scene_name", "scene"))
    reference_mode = _sanitize_filename_token(meta.get("reference_mode", "ref"))
    retrieval_method = _sanitize_filename_token(meta.get("retrieval_method", "retrieval"))
    num_blocks = int(meta.get("num_blocks", 0))
    block_h, block_w = meta.get("block_size", [0, 0])
    suffix = _sanitize_filename_token(meta.get("mids_hash", "hash"))
    filename = f"{scene}__{reference_mode}__{retrieval_method}__blocks{num_blocks}__{block_h}x{block_w}__{suffix}.pt"
    return cache_root / filename

def _load_cached_ref_features(cache_path, expected_meta):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(str(cache_path), map_location="cpu")
        if not isinstance(payload, dict):
            return None
        if payload.get("meta", None) != expected_meta:
            return None
        gf = payload.get("gf", None)
        if gf is None:
            return None
        if not torch.is_tensor(gf):
            gf = torch.as_tensor(gf)
        return gf.detach().cpu()
    except Exception as exc:
        print(f"[v6 patched cache warning] failed to load reference features: {cache_path} ({exc})")
        return None

def _save_cached_ref_features(cache_path, meta, gf):
    cache_path = Path(cache_path)
    ensure_dir(cache_path.parent)
    try:
        payload = {
            "meta": meta,
            "gf": gf.detach().cpu() if torch.is_tensor(gf) else torch.as_tensor(gf),
            "saved_at_unix": float(time.time()),
        }
        torch.save(payload, str(cache_path))
        meta_path = cache_path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[v6 patched cache save] saved reference features: {cache_path}")
    except Exception as exc:
        print(f"[v6 patched cache warning] failed to save reference features: {cache_path} ({exc})")

def _extract_query_feature_with_dummy_gallery(
    retrieval_size,
    config,
    method_dict,
    MID,
    block_size,
    ref_image,
    UAV_img,
    img_transform,
    retrieval_model,
):
    """Reuse the original extractor but pass only one dummy gallery block to get qf."""
    dummy_mid = np.asarray(MID, dtype=np.float32).reshape(-1, 2)[:1]
    _, qf = extract_features(
        retrieval_size,
        config,
        method_dict,
        dummy_mid,
        block_size,
        ref_image,
        UAV_img,
        img_transform,
        retrieval_model,
    )
    return qf

def retrieval_all_anyvisloc(
    ref_image,
    uav_image_bgr,
    uav_data,
    map_resolution,
    map_origin_local,
    matRotation,
    save_path,
    opt,
    config,
    method_dict,
):
    """Image-level retrieval for AnyVisLoc NPZ samples.

    v5 changes:
      - when opt.pose_priori == "p", reference/gallery features are cached by
        scene, reference mode, retrieval method, block size, step size, and block layout;
      - retrieval visualization separates UAV and reference results with a dashed line;
      - retrieval result titles use circled numbers;
      - each retrieved reference crop draws the true view-center as a red cross
        when that point lies inside the crop.
    """
    ensure_dir(save_path)
    retrieval_size = int(config.get("RETRIEVAL_SIZE", 384))
    map_res = as_vec2(map_resolution, "map_resolution")

    # Default behavior: use pitch/yaw/altitude to estimate the UAV view center
    # unless the whole pose prior is explicitly disabled.
    center_col, center_row = view_center_anyvisloc(
        uav_data,
        map_res,
        map_origin_local,
        matRotation,
        use_shift=(getattr(opt, "pose_priori", "yp") != "unknown"),
    )

    cover = method_dict["retrieval_cover"]
    drone_resolution, drone_size = resolution_size_anyvisloc(uav_data, opt)

    # RoMa fine scaling is scalar. AnyVisLoc maps are normally square-resolution;
    # use the mean as a safe fallback for slightly anisotropic maps.
    ref_res_scalar = float(np.mean(map_res))
    finescale = drone_resolution / max(ref_res_scalar, 1e-12)

    patch_scale = float(getattr(opt, "patch_scale", 1.5))
    view_m = float(drone_size[0]) * float(drone_resolution) * patch_scale
    min_patch_size_m = float(getattr(opt, "min_patch_size_m", 0.0) or 0.0)
    max_patch_size_m = float(getattr(opt, "max_patch_size_m", 0.0) or 0.0)
    if min_patch_size_m > 0:
        view_m = max(view_m, min_patch_size_m)
    if max_patch_size_m > 0:
        view_m = min(view_m, max_patch_size_m)
    patch_h = _even_ceil(view_m / max(float(map_res[1]), 1e-12))
    patch_w = _even_ceil(view_m / max(float(map_res[0]), 1e-12))

    # Avoid invalid negative starts in the original sliding-window helper.
    max_h = max(2, int(ref_image.shape[0]) - 2)
    max_w = max(2, int(ref_image.shape[1]) - 2)
    patch_h = min(patch_h, max_h)
    patch_w = min(patch_w, max_w)
    if patch_h % 2:
        patch_h = max(2, patch_h - 1)
    if patch_w % 2:
        patch_w = max(2, patch_w - 1)

    block_size = [patch_h, patch_w]
    step_size = [
        max(1, int(patch_h * (100 - cover) / 100)),
        max(1, int(patch_w * (100 - cover) / 100)),
    ]

    MID = compute_block_mid_wo_black(ref_image, block_size, step_size)
    if MID.size == 0:
        raise RuntimeError("No valid retrieval gallery blocks. Check map content, patch size, or retrieval cover.")
    mids = MID.reshape(-1, 2)

    UAV_image = crop_center_from_bgr(uav_image_bgr, int(drone_size[0]), int(drone_size[1]))
    retrieval_model = method_dict["retrieval_model"]
    img_transform = method_dict["img_transform"]

    retrieval_t0 = time.time()
    UAV_image = UAV_image.resize((retrieval_size, retrieval_size))
    UAV_img = img_transform(UAV_image)

    use_ref_cache = (
        getattr(opt, "pose_priori", "yp") == "p"
        and not bool(getattr(opt, "disable_ref_feature_cache", False))
    )
    cache_path = None
    cache_meta = None
    gf = None

    if use_ref_cache:
        cache_meta = _make_ref_feature_cache_meta(
            opt=opt,
            method_dict=method_dict,
            config=config,
            ref_image=ref_image,
            block_size=block_size,
            step_size=step_size,
            mids=mids,
        )
        cache_path = _make_ref_feature_cache_path(save_path, opt, cache_meta)
        gf = _load_cached_ref_features(cache_path, cache_meta)

    if gf is not None:
        print(f"[v6 patched cache hit] loaded reference features: {cache_path}")
        qf = _extract_query_feature_with_dummy_gallery(
            retrieval_size,
            config,
            method_dict,
            MID,
            block_size,
            ref_image,
            UAV_img,
            img_transform,
            retrieval_model,
        )
        gf = gf.to(qf.device)
    else:
        gf, qf = extract_features(
            retrieval_size,
            config,
            method_dict,
            MID,
            block_size,
            ref_image,
            UAV_img,
            img_transform,
            retrieval_model,
        )
        if use_ref_cache and cache_path is not None and cache_meta is not None:
            _save_cached_ref_features(cache_path, cache_meta, gf)

    retrieval_t1 = time.time()

    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().detach().cpu().numpy()
    order = np.argsort(score)[::-1]
    retrieval_time_cost = (retrieval_t1 - retrieval_t0) / (len(MID) + 1)

    row_starts, col_starts, PDE_list = [], [], []
    tile_physical_width = max(float(map_res[0]) * float(patch_w), 1e-12)
    for i in range(len(order)):
        mid_row, mid_col = mids[order[i], :]
        row0 = max(0, int(mid_row - patch_h / 2))
        col0 = max(0, int(mid_col - patch_w / 2))
        row_starts.append(row0)
        col_starts.append(col0)
        dx_m = (center_col - mid_col) * float(map_res[0])
        dy_m = (center_row - mid_row) * float(map_res[1])
        d_i = float(np.hypot(dx_m, dy_m))
        PDE_list.append(float(d_i / tile_physical_width))

    if bool(getattr(opt, "visualize", False)):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D

        show_n = int(config.get("RETRIEVAL_IMG_NUM", 5))
        retrieval_img_name = method_dict["retrieval_img_name"]
        show_n_eff = min(len(mids), show_n)

        fig = plt.figure(figsize=(10, 2.8))
        axes = []

        ax0 = plt.subplot(1, show_n + 1, 1)
        axes.append(ax0)
        ax0.axis("off")
        uav_vis = _draw_cross_at_image_center(np.asarray(UAV_image), color=(255, 0, 0))
        ax0.imshow(uav_vis)
        ax0.set_title("UAV")

        for i in range(show_n_eff):
            ax = plt.subplot(1, show_n + 1, i + 2)
            axes.append(ax)
            ax.axis("off")

            mid_row, mid_col = mids[order[i], :]
            row0 = max(0, int(mid_row - patch_h / 2))
            col0 = max(0, int(mid_col - patch_w / 2))
            img = ref_image[row0:row0 + int(patch_h), col0:col0 + int(patch_w)]
            # Red cross: estimated/true UAV view center if it falls inside this retrieved patch.
            img = _draw_red_cross_if_inside(img, center_col, center_row, col0, row0)
            # Green cross: geometric center of each retrieved patch, always visible.
            img = _draw_cross_at_image_center(img, color=(0, 255, 0))

            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{_circled_number(i + 1)} R:{PDE_list[i]:.3f}")

        plt.ioff()
        plt.tight_layout()

        # Dashed visual separator between UAV image and retrieval results.
        if len(axes) >= 2:
            fig.canvas.draw()
            b0 = axes[0].get_position()
            b1 = axes[1].get_position()
            x_sep = (b0.x1 + b1.x0) * 0.5
            y0 = min(ax.get_position().y0 for ax in axes)
            y1 = max(ax.get_position().y1 for ax in axes)
            fig.add_artist(
                Line2D(
                    [x_sep, x_sep],
                    [y0, y1],
                    transform=fig.transFigure,
                    linestyle="--",
                    linewidth=1.2,
                    color="0.35",
                    alpha=0.9,
                )
            )

        plt.savefig(str(Path(save_path) / retrieval_img_name.lstrip("/")), dpi=160)
        plt.close(fig)

    return order, row_starts, col_starts, PDE_list, patch_h, patch_w, finescale, retrieval_time_cost

def extract_features(retrieval_size, config, method_dict, MID, block_size, SATELLITE, UAV, val_transforms, model):
    model.eval()

    row, column = MID.shape
    process = list(range(row))
    img_features_list = []
    mids = MID.reshape(-1,2)
    # Process images in batches
    for batch_start in tqdm(range(0, len(process), config['BATCH_SIZE']), desc='Image Retrieval'):

        # Get the end index of the current batch
        batch_end = batch_start + config['BATCH_SIZE']
        # If the batch end index exceeds the total number, use the actual number
        if batch_end > len(process):
            batch_end = len(process)

        batch_indices = process[batch_start:batch_end]
        batch_images = []

        for item in batch_indices:
            mid_x, mid_y = mids[item, :]
            img = cv2.resize(SATELLITE[int(mid_x - block_size[0] / 2):int(mid_x + block_size[0] / 2),
                  int(mid_y - block_size[1] / 2):int(mid_y + block_size[1] / 2)], (retrieval_size,retrieval_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = val_transforms(img)
            batch_images.append(img)

        # Convert images to tensors and stack them
        if len(batch_images) > 0:  # Ensure there is at least one image
            batch_images = torch.stack(batch_images, dim=0).to(config['DEVICE'])
            with torch.no_grad():
                with torch.cuda.amp.autocast():

                    batch_features = model(batch_images)

                    batch_features = feature_fusion_all(method_dict, batch_features)

                    if config['RETRIEVAL_FEATURE_NORM']:
                        batch_features = F.normalize(batch_features, dim=-1)

            # Append features to the list
            img_features_list.append(batch_features.to(torch.float32))

    # Concatenate all features together
    if len(img_features_list) > 0:  # Ensure there is at least one feature
        img_features = torch.cat(img_features_list, dim=0)
    else:
        img_features = torch.tensor([], dtype=torch.float32, device=config['DEVICE'])

    with torch.no_grad():
        with autocast():
            UAV1 = UAV.to(config['DEVICE'])
            UAV1 = UAV1.unsqueeze(0)

            UAV_feature = model(UAV1)
            UAV_feature = feature_fusion_all(method_dict, UAV_feature)
            if config['RETRIEVAL_FEATURE_NORM']:
                UAV_feature = F.normalize(UAV_feature, dim=-1)

    return img_features, UAV_feature

def feature_fusion_all(method_dict, img_feature):
    if method_dict['retrieval_method'] == 'MCCG':
        if len(img_feature.shape) == 3:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(img_feature.size(-1))
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
    elif method_dict['retrieval_method'] == 'LPN':
        block = 6
        fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(block)
        img_feature = img_feature.div(fnorm.expand_as(img_feature))
        img_feature = img_feature.view(img_feature.size(0), -1)
    elif method_dict['retrieval_method'] == 'MIFT':
        if len(img_feature.shape)==3:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(img_feature.size(-1))
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
    elif method_dict['retrieval_method'] == 'FSRA':
        if len(img_feature.shape)==3:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(img_feature.size(-1))
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))

    elif method_dict['retrieval_method'] == 'CAMP':
            img_feature = img_feature[-2]
    elif method_dict['retrieval_method'] == 'DAC':
            img_feature = img_feature[-2]
    return img_feature

def get_transforms_new():
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return data_transforms

def safe_float(x):
    if x is None:
        return None
    try:
        value = float(x)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value

def retrieval_metrics_from_ratios(ratios, k=5, pdm_lambda=6.0, pdm_alpha=0.9):
    """Compute per-query PDM@K and Recall@K from retrieval ratios.

    ratios/PDE are ordered by retrieval rank. The nearest tile to the
    ground-truth view center has the smallest ratio, so its one-based index is
    the ground-truth rank.
    """
    ratios = np.asarray(ratios, dtype=np.float64).reshape(-1)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return None, None, None

    k_eff = min(max(1, int(k)), int(ratios.size))
    top_ratios = ratios[:k_eff]
    weights = np.arange(k_eff, 0, -1, dtype=np.float64)

    logits = np.clip(float(pdm_lambda) * (top_ratios - float(pdm_alpha)), -60.0, 60.0)
    retrieval_scores = 1.0 / (1.0 + np.exp(logits))
    pdm_at_k = float(np.sum(weights * retrieval_scores) / np.sum(weights))

    gt_rank = int(np.argmin(ratios)) + 1
    recall_at_k = float(gt_rank <= k_eff)
    return pdm_at_k, recall_at_k, gt_rank

def retrieval_metrics_for_ks(ratios, ks, pdm_lambda=6.0, pdm_alpha=0.9):
    metrics = {}
    retrieval_gt_rank = None

    for k in ks:
        k = int(k)
        pdm_at_k, recall_at_k, gt_rank = retrieval_metrics_from_ratios(
            ratios,
            k=k,
            pdm_lambda=pdm_lambda,
            pdm_alpha=pdm_alpha,
        )
        metrics[f"pdm_at_{k}"] = safe_float(pdm_at_k)
        metrics[f"recall_at_{k}"] = safe_float(recall_at_k)
        metrics[f"PDM@{k}"] = safe_float(pdm_at_k)
        metrics[f"Recall@{k}"] = safe_float(recall_at_k)
        if retrieval_gt_rank is None and gt_rank is not None:
            retrieval_gt_rank = int(gt_rank)

    return metrics, retrieval_gt_rank


# Official reference loaders and retrieval-only entry/aggregation.
_REF_JSON_CACHE = {}
_REF_ARRAY_CACHE = {}

def save_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def dsm_to_2d(dsm):
    dsm = np.asarray(dsm)
    if dsm.ndim == 3:
        dsm = dsm[:, :, 0]
    if dsm.ndim != 2:
        raise ValueError(f"DSM must be [H,W], got {dsm.shape}")
    return dsm.astype(np.float32, copy=False)

def _resolve_ref_json_path(sample, ref):
    json_path = None
    if isinstance(ref, dict):
        json_path = ref.get("ref_json_path", None) or ref.get("ref_path", None)

    sample_npz = sample.get("npz_path", None)
    if sample_npz is not None and json_path is None:
        p = Path(to_python_str(sample_npz))
        scene_id = int(sample["scene_id"])
        json_path = p.parent / f"L{scene_id:02d}_reference.json"

    return None if json_path is None else Path(to_python_str(json_path))

def _load_reference_json(sample, ref):
    json_path = _resolve_ref_json_path(sample, ref)
    if json_path is None:
        return None, ""
    if not json_path.exists():
        return None, str(json_path)

    key = str(json_path)
    if key not in _REF_JSON_CACHE:
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["__base_dir__"] = str(json_path.parent)
        meta["__ref_json_path__"] = str(json_path)
        _REF_JSON_CACHE[key] = meta
    return _REF_JSON_CACHE[key], str(json_path)

def _json_path_to_abs(meta, value):
    if value is None:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    return Path(meta.get("__base_dir__", ".")) / p

def _load_ref_array(path, kind):
    path = Path(path)
    key = (str(path), kind)
    if key in _REF_ARRAY_CACHE:
        return _REF_ARRAY_CACHE[key]
    if kind == "map":
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Cannot read reference map PNG: {path}")
    elif kind == "dsm":
        # mmap keeps large DSMs cheap to open. If later code needs a real ndarray,
        # NumPy will load pages on demand.
        arr = np.load(str(path), mmap_mode="r")
    elif kind == "mask":
        arr = np.load(str(path), mmap_mode="r")
    else:
        raise ValueError(f"Unknown reference array kind: {kind}")
    _REF_ARRAY_CACHE[key] = arr
    return arr

def normalize_reference_mode(mode="aerial"):
    """Normalize reference mode. The CLI only exposes aerial/satellite."""
    m = str(mode or "aerial").lower()
    if m in ("aerial", "ortho", "high"):
        return "aerial"
    if m in ("satellite", "sat", "low"):
        return "satellite"
    raise ValueError(f"Unsupported reference_mode: {mode!r}. Use 'aerial' or 'satellite'.")

def _reference_view_from_json(sample, ref, reference_mode):
    meta, ref_path = _load_reference_json(sample, ref)
    if meta is None:
        return None

    modes = meta.get("modes", {})
    if reference_mode not in modes:
        raise KeyError(f"Reference JSON {ref_path} has no mode {reference_mode!r}. Available={list(modes.keys())}")
    mode = modes[reference_mode]

    map_path = _json_path_to_abs(meta, mode.get("map_path"))
    dsm_path = _json_path_to_abs(meta, mode.get("dsm_path"))
    if map_path is None or dsm_path is None:
        raise KeyError(f"Reference JSON {ref_path} mode {reference_mode!r} missing map_path/dsm_path")

    map_img = _load_ref_array(map_path, "map")
    dsm = _load_ref_array(dsm_path, "dsm")

    map_origin = as_vec2(mode.get("map_origin_local", [0.0, 0.0]), f"{reference_mode}_map_origin_local", [0.0, 0.0])
    dsm_origin_rel = as_vec2(mode.get("dsm_origin_local", [0.0, 0.0]), f"{reference_mode}_dsm_origin_local", [0.0, 0.0])
    dsm_origin_dataset = (map_origin + dsm_origin_rel).astype(np.float32)

    valid_mask = None

    return {
        "reference_mode": reference_mode,
        "map": ensure_bgr_uint8(map_img),
        "dsm": dsm_to_2d(dsm),
        "dsm_valid_mask": valid_mask,
        "map_resolution": as_vec2(mode.get("map_resolution"), f"{reference_mode}_map_resolution"),
        "dsm_resolution": as_vec2(mode.get("dsm_resolution"), f"{reference_mode}_dsm_resolution"),
        "map_origin_local": map_origin,
        "dsm_origin_local": dsm_origin_dataset,
        "dsm_origin_relative_to_map": dsm_origin_rel,
        "map_field": mode.get("map_field", f"{reference_mode}_map"),
        "dsm_field": mode.get("dsm_field", f"{reference_mode}_dsm"),
        "ref_path": ref_path,
        "ref_json_path": ref_path,
    }

def build_anyvisloc_reference_view(sample, reference_mode="aerial"):
    reference_mode = normalize_reference_mode(reference_mode)
    ref = sample.get("reference", {})
    json_view = _reference_view_from_json(sample, ref, reference_mode)
    if json_view is None:
        json_path = _resolve_ref_json_path(sample, ref)
        raise FileNotFoundError(
            f"Reference JSON not found. Expected: {json_path}. "
            "This short-version runner requires Lxx_reference.json plus PNG/NPY reference files."
        )
    return json_view

def as_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(x)

def process_one_sample(sample, opt, config, method_dict, retrieval_method, combo_dir):
    sample_id = as_text(sample["sample_id"])
    scene_name = as_text(sample["scene_name"])
    save_path = Path(combo_dir) / scene_name / sample_id
    ensure_dir(save_path)

    result_json = save_path / "retrieval_result.json"
    if opt.skip_existing and result_json.exists():
        with result_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    opt._current_scene_name = scene_name
    opt._current_sample_id = sample_id

    true_pos = anyvisloc_sample_to_truepos(sample)
    uav_image = tensor_rgb_to_bgr_uint8(sample["image"])

    view = build_anyvisloc_reference_view(sample, opt.reference_mode)
    ref_map0 = view["map"]
    map_resolution = view["map_resolution"]
    map_origin_local = view["map_origin_local"]

    if opt.pose_priori == "yp":
        ref_map, mat_rotation = dumpRotateImage(ref_map0, true_pos["yaw"])
    else:
        ref_map = ref_map0
        mat_rotation = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    t0 = time.time()
    (
        ir_order,
        row_starts,
        col_starts,
        pde_list,
        patch_h,
        patch_w,
        fine_scale,
        retrieval_time,
    ) = retrieval_all_anyvisloc(
        ref_map,
        uav_image,
        true_pos,
        map_resolution,
        map_origin_local,
        mat_rotation,
        str(save_path),
        opt,
        config,
        method_dict,
    )
    total_time = time.time() - t0

    metrics_by_k, retrieval_gt_rank = retrieval_metrics_for_ks(
        pde_list,
        ks=opt.retrieval_ks,
        pdm_lambda=opt.pdm_lambda,
        pdm_alpha=opt.pdm_alpha,
    )
    primary_k = int(opt.retrieval_k)

    result = {
        "scene_name": scene_name,
        "sample_id": sample_id,
        "npz_path": as_text(sample.get("npz_path", "")),
        "reference_mode": opt.reference_mode,
        "retrieval_method": retrieval_method,
        "pose_priori": opt.pose_priori,
        "retrieval_ks": [int(k) for k in opt.retrieval_ks],
        "retrieval_k": primary_k,
        "pdm_lambda": float(opt.pdm_lambda),
        "pdm_alpha": float(opt.pdm_alpha),
        "pdm_at_k": metrics_by_k.get(f"PDM@{primary_k}"),
        "recall_at_k": metrics_by_k.get(f"Recall@{primary_k}"),
        "retrieval_gt_rank": None if retrieval_gt_rank is None else int(retrieval_gt_rank),
        "num_gallery_blocks": int(len(pde_list)),
        "patch_h": int(patch_h),
        "patch_w": int(patch_w),
        "fine_scale": float(fine_scale),
        "retrieval_time_s": float(retrieval_time),
        "total_time_s": float(total_time),
        "best_pde": safe_float(np.min(pde_list) if len(pde_list) else None),
        "top1_pde": safe_float(pde_list[0] if len(pde_list) else None),
        "result_dir": str(save_path),
    }
    result.update(metrics_by_k)

    save_json(result_json, result)
    return result

def summarize_retrieval(rows, failed_rows, opt, extra=None):
    attempted = len(rows) + len(failed_rows)
    denominator = max(attempted, 1)
    retrieval_times = [
        float(v)
        for v in (safe_float(r.get("retrieval_time_s")) for r in rows)
        if v is not None
    ]

    ks = [int(k) for k in opt.retrieval_ks]
    primary_k = int(opt.retrieval_k)

    summary = {
        "dataset_root": str(opt.dataset_root),
        "scenes": opt.scenes,
        "reference_mode": opt.reference_mode,
        "pose_priori": opt.pose_priori,
        "retrieval_ks": ks,
        "retrieval_k": primary_k,
        "num_attempted": int(attempted),
        "num_success": int(len(rows)),
        "num_failed": int(len(failed_rows)),
        "mean_retrieval_time_s": float(np.mean(retrieval_times)) if retrieval_times else None,
    }

    # Failed retrieval rows contribute 0 to aggregate metrics.
    for k in ks:
        pdm_values = [
            float(v)
            for v in (safe_float(r.get(f"PDM@{k}")) for r in rows)
            if v is not None
        ]
        recall_values = [
            float(v)
            for v in (safe_float(r.get(f"Recall@{k}")) for r in rows)
            if v is not None
        ]
        pdm = float(np.sum(pdm_values) / denominator) if attempted else None
        recall = float(np.sum(recall_values) / denominator) if attempted else None
        summary[f"metric_valid_rows@{k}"] = int(min(len(pdm_values), len(recall_values)))
        summary[f"pdm_at_{k}"] = pdm
        summary[f"recall_at_{k}"] = recall
        summary[f"PDM@{k}"] = pdm
        summary[f"Recall@{k}"] = recall

    summary["metric_valid_rows"] = summary.get(f"metric_valid_rows@{primary_k}", 0)
    summary["pdm_at_k"] = summary.get(f"PDM@{primary_k}")
    summary["recall_at_k"] = summary.get(f"Recall@{primary_k}")

    if extra:
        summary.update(extra)
    return summary

