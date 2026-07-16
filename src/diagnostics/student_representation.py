"""Zero-training retrieval audit of real RepViT intermediate tensors."""

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diagnostics.runtime import (
    DATASET_CHOICES, apply_formal_protocol_range, build_formal_loaders,
    build_student, extract_pair, formal_pipeline_metrics, identity_fingerprint,
    iter_loader_pairs, parity_audit, raw_retrieval_metrics, write_json,
)


AUDIT_DESCRIPTORS = {
    "final_descriptor": "final_descriptor",
    "f2_gap_l2": "f2_gap",
    "f3_gap_l2": "f3_gap",
    "f4_gap_l2": "f4_gap",
}


class AuditDescriptor(nn.Module):
    def __init__(self, student, audit_key):
        super().__init__()
        self.student = student
        self.audit_key = audit_key
        self.tensor_audit = None

    @property
    def backbone(self):
        return self.student.backbone

    def forward(self, images):
        audit = self.student(images, return_audit_features=True)
        value = audit[self.audit_key]
        descriptor = value if self.audit_key == "final_descriptor" else F.normalize(value, dim=1)
        self.tensor_audit = {
            "shape": list(descriptor.shape),
            "dtype": str(descriptor.dtype).replace("torch.", ""),
            "finite": bool(torch.isfinite(descriptor).all()),
        }
        return descriptor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="src/checkpoint/student/B0-2GPU-3090/best_model.pth")
    parser.add_argument("--dataset", default="all", choices=DATASET_CHOICES + ("all",))
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--u1652_root", default=None)
    parser.add_argument("--sues_root", default=None)
    parser.add_argument("--gta_root", default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--sues_height", choices=("150", "200", "250", "300", "all"), default="all")
    parser.add_argument("--sues_horizontal_flip", action="store_true")
    parser.add_argument("--gta_query_mode", choices=("D2S",), default="D2S")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parity_tolerance", type=float, default=1e-4)
    parser.add_argument("--output_file", default="src/diagnostics/results/bottleneck_audit/representation_audit.json")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    device_name = "cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    student = build_student(args.checkpoint, device, args.temperature)
    print(
        "[RepresentationAudit] student_precision_path="
        f"input={next(student.backbone.parameters()).dtype}, similarity=float32"
    )
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    before = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
    results = {}
    datasets = DATASET_CHOICES if args.dataset == "all" else (args.dataset,)
    explicit_data_dir = args.data_dir
    for dataset in datasets:
        args.dataset = dataset
        configured_roots = {
            "1652": args.u1652_root,
            "SUES-200": args.sues_root,
            "GTA-UAV": args.gta_root,
        }
        args.data_dir = configured_roots[dataset] or (
            explicit_data_dir if len(datasets) == 1 else None
        )
        args.gta_split = "cross-area"
        loaders = build_formal_loaders(args)
        results[dataset] = {}
        for height, direction, pair in iter_loader_pairs(dataset, loaders):
            protocol = f"{height or 'all'}/{direction}"
            entry = {"identity_sha256": identity_fingerprint(*pair), "descriptors": {}}
            results[dataset][protocol] = entry
            final_metrics = None
            for descriptor_name, audit_key in AUDIT_DESCRIPTORS.items():
                wrapper = AuditDescriptor(student, audit_key).eval()
                flip = dataset == "SUES-200" and args.sues_horizontal_flip
                features = extract_pair(
                    wrapper, pair, device, f"audit:{descriptor_name}:{dataset}:{protocol}", flip
                )
                features = apply_formal_protocol_range(features, pair, dataset)
                metrics = formal_pipeline_metrics(
                    wrapper, pair, device, dataset,
                    f"Representation:{descriptor_name}:{dataset}:{protocol}", flip,
                )
                descriptor_result = {
                    "metrics": metrics,
                    "tensor_audit": {
                        "last_runtime_batch": wrapper.tensor_audit,
                        "query_shape": list(features["query_features"].shape),
                        "gallery_shape": list(features["gallery_features"].shape),
                        "query_dtype": str(features["query_features"].dtype).replace("torch.", ""),
                        "gallery_dtype": str(features["gallery_features"].dtype).replace("torch.", ""),
                        "finite": bool(
                            torch.isfinite(features["query_features"]).all()
                            and torch.isfinite(features["gallery_features"]).all()
                        ),
                    },
                }
                if descriptor_name == "final_descriptor":
                    diagnostic_metrics = raw_retrieval_metrics(features, dataset, device)
                    descriptor_result["parity_audit"] = parity_audit(
                        diagnostic_metrics, metrics, args.parity_tolerance
                    )
                    final_metrics = metrics
                entry["descriptors"][descriptor_name] = descriptor_result
            for descriptor_name, descriptor_result in entry["descriptors"].items():
                descriptor_result["delta_vs_formal_B0"] = {
                    name: value - final_metrics[name]
                    for name, value in descriptor_result["metrics"].items()
                }
    unchanged = all(torch.equal(before[name], value.detach().cpu()) for name, value in student.state_dict().items())
    if not unchanged:
        raise RuntimeError("representation audit modified student parameters or buffers")
    payload = {
        "checkpoint": args.checkpoint,
        "datasets": list(datasets),
        "zero_training": True,
        "parameters_and_buffers_unchanged": unchanged,
        "real_audit_tensors": [
            "f2", "f3", "f4", "f2_gap", "f3_gap", "f4_gap",
            "bn_input", "bn_output", "final_descriptor",
        ],
        "results": results,
    }
    write_json(Path(args.output_file), payload)


if __name__ == "__main__":
    main()
