"""Mine full-gallery U1652 train hard-negative identities for G4."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.dataset.datasets import read_rgb_image
from src.dataset.transforms import get_test_transforms
from src.diagnostics.runtime import build_student, build_teacher
from src.training.student_train import cast_images_to_model_dtype
from src.utils.train_eval_utils import select_model_descriptor


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class IdentityImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, identity_index = self.items[index]
        image = read_rgb_image(path)
        return self.transform(image=image)["image"], identity_index


def enumerate_train_domains(data_dir):
    roots = {
        "drone": Path(data_dir) / "drone",
        "satellite": Path(data_dir) / "satellite",
    }
    for root in roots.values():
        if not root.is_dir():
            raise FileNotFoundError(root)
    identities = sorted(
        pid for pid in os.listdir(roots["satellite"])
        if (roots["satellite"] / pid).is_dir()
        and (roots["drone"] / pid).is_dir()
    )
    domain_items = {}
    for domain, root in roots.items():
        items = []
        for identity_index, pid in enumerate(identities):
            paths = sorted(
                str(path) for path in (root / pid).iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not paths:
                raise RuntimeError(f"identity {pid} has no {domain} images")
            items.extend((path, identity_index) for path in paths)
        domain_items[domain] = items
    return identities, domain_items


@torch.inference_mode()
def extract_identity_prototypes(
    model, items, identity_count, transform, batch_size, num_workers, device
):
    loader = DataLoader(
        IdentityImageDataset(items, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    sums = None
    counts = torch.zeros(identity_count, dtype=torch.float32, device=device)
    for images, labels in loader:
        images = cast_images_to_model_dtype(
            model, images.to(device, non_blocking=True)
        )
        descriptors = select_model_descriptor(model(images))
        descriptors = F.normalize(descriptors.float(), dim=1)
        if sums is None:
            sums = torch.zeros(
                identity_count,
                descriptors.size(1),
                dtype=torch.float32,
                device=device,
            )
        labels = labels.to(device=device, dtype=torch.long)
        sums.index_add_(0, labels, descriptors)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    if sums is None or torch.any(counts == 0):
        raise RuntimeError("failed to extract every identity prototype")
    return F.normalize(sums / counts[:, None], dim=1).cpu()


def _positive_ranks(similarity):
    order = torch.argsort(similarity, dim=1, descending=True, stable=True)
    identity = torch.arange(similarity.size(0))[:, None]
    return order.eq(identity).to(torch.int64).argmax(dim=1) + 1, order


def mine_direction(
    identities,
    student_anchor,
    student_gallery,
    teacher_anchor,
    teacher_gallery,
    *,
    student_topk=20,
    teacher_adv_pool_size=4,
):
    student_similarity = (
        F.normalize(student_anchor.float(), dim=1)
        @ F.normalize(student_gallery.float(), dim=1).t()
    )
    teacher_similarity = (
        F.normalize(teacher_anchor.float(), dim=1)
        @ F.normalize(teacher_gallery.float(), dim=1).t()
    )
    student_rank, student_order = _positive_ranks(student_similarity)
    teacher_rank, _ = _positive_ranks(teacher_similarity)
    records = {}
    same_identity_count = 0
    duplicate_count = 0
    retained_wrong_top1 = 0
    candidate_counts = []
    for anchor_index, anchor_pid in enumerate(identities):
        wrong = [
            int(index) for index in student_order[anchor_index].tolist()
            if int(index) != anchor_index
        ][:student_topk]
        teacher_advantage = []
        if int(teacher_rank[anchor_index]) == 1:
            teacher_positive = teacher_similarity[anchor_index, anchor_index]
            teacher_advantage = [
                index for index in wrong
                if teacher_positive > teacher_similarity[anchor_index, index]
            ][:teacher_adv_pool_size]
        student_ids = [identities[index] for index in wrong]
        advantage_ids = [identities[index] for index in teacher_advantage]
        same_identity_count += sum(pid == anchor_pid for pid in student_ids)
        same_identity_count += sum(pid == anchor_pid for pid in advantage_ids)
        duplicate_count += len(student_ids) - len(set(student_ids))
        duplicate_count += len(advantage_ids) - len(set(advantage_ids))
        if (
            int(student_rank[anchor_index]) > 1
            and student_ids
            and student_ids[0] in advantage_ids
        ):
            retained_wrong_top1 += 1
        candidate_counts.append(len(advantage_ids))
        records[anchor_pid] = {
            "student_positive_rank": int(student_rank[anchor_index]),
            "teacher_positive_rank": int(teacher_rank[anchor_index]),
            "student_topk_negative_ids": student_ids,
            "teacher_advantage_negative_ids": advantage_ids,
        }
    count = len(identities)
    student_wrong = student_rank.gt(1)
    teacher_correct = teacher_rank.eq(1)
    student_wrong_count = int(student_wrong.sum().item())
    coverage = sum(value > 0 for value in candidate_counts)
    audit = {
        "identity_count": count,
        "student_top1_error_ratio": float(student_wrong.float().mean().item()),
        "teacher_top1_correct_ratio": float(teacher_correct.float().mean().item()),
        "teacher_correct_student_wrong_ratio": float(
            (teacher_correct & student_wrong).float().mean().item()
        ),
        "teacher_adv_candidate_identity_coverage": float(coverage / max(count, 1)),
        "student_top1_wrong_identity_retained_ratio": float(
            retained_wrong_top1 / max(student_wrong_count, 1)
        ),
        "candidate_count": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "mean": float(sum(candidate_counts) / max(len(candidate_counts), 1)),
            "max": max(candidate_counts) if candidate_counts else 0,
        },
        "same_identity_negative_count": same_identity_count,
        "duplicate_candidate_count": duplicate_count,
    }
    return records, audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/U1652/train")
    parser.add_argument(
        "--student_checkpoint",
        default="src/checkpoint/student/B0-2GPU-3090/best_model.pth",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        default="src/checkpoint/teacher/T0-3090/best_model.pth",
    )
    parser.add_argument("--teacher_metrics", default=None)
    parser.add_argument(
        "--output_dir",
        default="src/diagnostics/results/g4_mining",
    )
    parser.add_argument("--student_topk", type=int, default=20)
    parser.add_argument("--teacher_adv_pool_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.student_topk <= 0 or args.teacher_adv_pool_size <= 0:
        parser.error("mining pool sizes must be positive")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    identities, domain_items = enumerate_train_domains(args.data_dir)
    transform = get_test_transforms([args.img_size, args.img_size])
    student = build_student(args.student_checkpoint, device)
    teacher = build_teacher(
        args.teacher_checkpoint, args.teacher_metrics, device
    )
    prototypes = {"student": {}, "teacher": {}}
    for domain in ("drone", "satellite"):
        prototypes["student"][domain] = extract_identity_prototypes(
            student, domain_items[domain], len(identities), transform,
            args.batch_size, args.num_workers, device,
        )
        prototypes["teacher"][domain] = extract_identity_prototypes(
            teacher, domain_items[domain], len(identities), transform,
            args.batch_size, args.num_workers, device,
        )
    directions, audits = {}, {}
    for name, anchor_domain, gallery_domain in (
        ("D2S", "drone", "satellite"),
        ("S2D", "satellite", "drone"),
    ):
        directions[name], audits[name] = mine_direction(
            identities,
            prototypes["student"][anchor_domain],
            prototypes["student"][gallery_domain],
            prototypes["teacher"][anchor_domain],
            prototypes["teacher"][gallery_domain],
            student_topk=args.student_topk,
            teacher_adv_pool_size=args.teacher_adv_pool_size,
        )
    identity_hash = hashlib.sha256(
        "\n".join(identities).encode("utf-8")
    ).hexdigest()
    metadata = {
        "data_dir": args.data_dir,
        "student_checkpoint": args.student_checkpoint,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_metrics": args.teacher_metrics,
        "identity_count": len(identities),
        "identity_hash": identity_hash,
        "student_topk": args.student_topk,
        "teacher_adv_pool_size": args.teacher_adv_pool_size,
        "prototype_definition": "L2(image)->identity_domain_mean->L2",
        "descriptor_tensors_saved": False,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "g4_hard_negatives.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(
            {"metadata": metadata, "directions": directions},
            handle, indent=2, ensure_ascii=False,
        )
    with open(
        os.path.join(args.output_dir, "g4_mining_audit.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(
            {"metadata": metadata, "directions": audits},
            handle, indent=2, ensure_ascii=False,
        )
    print(json.dumps(audits, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
