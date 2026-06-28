import argparse
import inspect
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from src.models.student_model import StudentModel


def safe_torch_load(path, map_location):
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def unwrap_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt

    for key in ("model", "state_dict", "student", "net"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def strip_module_prefix(state_dict):
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_student_checkpoint(model, checkpoint_path, strict=True):
    ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(unwrap_state_dict(ckpt))
    training_only_keys = [
        key for key in state_dict
        if (
            key.startswith("kd_projector.")
            or key.startswith("local_attn_head.")
            or key.startswith("student_local_proj.")
            or key.startswith("teacher_local_proj.")
        )
    ]
    if training_only_keys:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if (
                not key.startswith("kd_projector.")
                and not key.startswith("local_attn_head.")
                and not key.startswith("student_local_proj.")
                and not key.startswith("teacher_local_proj.")
            )
        }
    msg = model.load_state_dict(state_dict, strict=strict)
    print(f"[Eval] loaded checkpoint: {checkpoint_path}")
    print(f"[Eval] strict load: {strict}")
    if training_only_keys:
        print(
            "[Eval] ignored training-only KD keys: "
            f"{len(training_only_keys)}"
        )
    if not strict:
        print(f"[Eval] missing keys: {len(msg.missing_keys)}")
        print(f"[Eval] unexpected keys: {len(msg.unexpected_keys)}")
    return ckpt


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        feat = F.normalize(model(images), p=2, dim=1)
        features.append(feat.cpu())
        labels.append(target.cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def compute_recall_map(q_feat, q_label, g_feat, g_label, topk=(1, 5, 10)):
    sim = q_feat @ g_feat.t()
    indices = sim.argsort(dim=1, descending=True)
    retrieved = g_label[indices]
    matches = retrieved.eq(q_label.unsqueeze(1))

    result = {}
    for k in topk:
        k = min(k, retrieved.size(1))
        result[f"R@{k}"] = matches[:, :k].any(dim=1).float().mean().item()

    ranks = torch.arange(1, retrieved.size(1) + 1, dtype=torch.float32).unsqueeze(0)
    precision_at_k = matches.float().cumsum(dim=1) / ranks
    positives = matches.float().sum(dim=1).clamp_min(1.0)
    result["mAP"] = ((precision_at_k * matches.float()).sum(dim=1) / positives).mean().item()
    return result


@torch.no_grad()
def evaluate_u1652(model, val_loaders):
    device = next(model.parameters()).device
    results = {}

    for task_name, (q_loader, g_loader) in val_loaders.items():
        print(f"[Eval] extracting {task_name} query features...")
        q_feat, q_label = extract_features(model, q_loader, device)
        print(f"[Eval] extracting {task_name} gallery features...")
        g_feat, g_label = extract_features(model, g_loader, device)

        metrics = compute_recall_map(q_feat, q_label, g_feat, g_label, topk=(1, 5, 10))
        results[f"{task_name}_R1"] = metrics["R@1"]
        results[f"{task_name}_R5"] = metrics["R@5"]
        results[f"{task_name}_R10"] = metrics["R@10"]
        results[f"{task_name}_mAP"] = metrics["mAP"]

    if "D2S_R1" in results and "S2D_R1" in results:
        results["avg_R1"] = (results["D2S_R1"] + results["S2D_R1"]) / 2.0
    if "D2S_mAP" in results and "S2D_mAP" in results:
        results["avg_mAP"] = (results["D2S_mAP"] + results["S2D_mAP"]) / 2.0
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RepViT student checkpoint on U1652.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth or last_model.pth.")
    parser.add_argument("--val_data_dir", type=str, default="data/U1652")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no_strict", dest="strict", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    from src.dataset.val_dataloaders import build_student_val_dataloaders

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Eval] CUDA is not available; falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    batch_size = args.val_batch_size if args.val_batch_size is not None else args.batch_size
    val_loaders = build_student_val_dataloaders(
        data_dir=args.val_data_dir,
        img_size=[args.img_size, args.img_size],
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    model = StudentModel(
        ckpt_path=None,
        temperature=args.temperature,
    ).to(device)
    load_student_checkpoint(model, args.checkpoint, strict=args.strict)
    model.eval()

    results = evaluate_u1652(model, val_loaders)
    print("[Eval] results:")
    print(json.dumps(results, indent=2, sort_keys=True))

    if args.output_json is not None:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"[Eval] saved results to: {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[Error] Exception occurred during evaluation:")
        import traceback

        traceback.print_exc()
        raise
