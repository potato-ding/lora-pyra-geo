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


@torch.inference_mode()
def extract_image_descriptors(
    model, items, transform, batch_size, num_workers, device
):
    """Extract descriptors in deterministic dataset enumeration order."""
    loader = DataLoader(
        IdentityImageDataset(items, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    features, labels = [], []
    for images, batch_labels in loader:
        images = cast_images_to_model_dtype(
            model, images.to(device, non_blocking=True)
        )
        descriptors = select_model_descriptor(model(images))
        features.append(F.normalize(descriptors.float(), dim=1).cpu())
        labels.append(batch_labels.to(dtype=torch.long).cpu())
    if not features:
        raise RuntimeError("no images were enumerated for descriptor extraction")
    return torch.cat(features), torch.cat(labels)


def _positive_ranks(similarity):
    order = torch.argsort(similarity, dim=1, descending=True, stable=True)
    identity = torch.arange(similarity.size(0))[:, None]
    return order.eq(identity).to(torch.int64).argmax(dim=1) + 1, order


def official_identity_scores(
    query_features,
    gallery_features,
    gallery_labels,
    identity_count,
    query_labels=None,
):
    """Collapse the formal image ranking by each identity's first occurrence.

    The formal evaluator ranks every gallery image. The first occurrence of an
    identity in that ranking is exactly its maximum image-level similarity.
    This preserves the official multi-positive S2D gallery semantics without
    averaging gallery images into a prototype.
    """
    image_similarity = (
        F.normalize(query_features.float(), dim=1)
        @ F.normalize(gallery_features.float(), dim=1).t()
    )
    identity_scores = torch.full(
        (query_features.size(0), identity_count),
        -torch.inf,
        dtype=image_similarity.dtype,
    )
    for identity_index in range(identity_count):
        mask = gallery_labels.eq(identity_index)
        if not bool(mask.any()):
            raise RuntimeError(
                f"gallery has no image for identity index {identity_index}"
            )
        identity_scores[:, identity_index] = image_similarity[:, mask].max(dim=1).values
    formal_top1 = gallery_labels[image_similarity.argmax(dim=1)]
    collapsed_top1 = identity_scores.argmax(dim=1)
    matched = int(formal_top1.eq(collapsed_top1).sum().item())
    total = int(query_features.size(0))
    parity = {
        "matched_query_count": matched,
        "query_count": total,
        "ratio": float(matched / max(total, 1)),
        "passed": matched == total,
        "definition": (
            "formal image-level gallery rank versus first-occurrence "
            "identity collapse"
        ),
    }
    if query_labels is not None:
        formal_correct = int(formal_top1.eq(query_labels).sum().item())
        collapsed_correct = int(collapsed_top1.eq(query_labels).sum().item())
        parity.update({
            "formal_top1_correct_count": formal_correct,
            "collapsed_top1_correct_count": collapsed_correct,
            "formal_top1_correct_ratio": float(formal_correct / max(total, 1)),
            "collapsed_top1_correct_ratio": float(
                collapsed_correct / max(total, 1)
            ),
            "top1_correct_count_abs_diff": abs(
                formal_correct - collapsed_correct
            ),
        })
    if not parity["passed"]:
        raise RuntimeError(f"official evaluator identity parity failed: {parity}")
    return identity_scores, parity


def _distribution(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "p75": float(torch.quantile(tensor, 0.75).item()),
        "p90": float(torch.quantile(tensor, 0.9).item()),
    }


def mine_query_level_direction(
    identities,
    query_labels,
    student_scores,
    teacher_scores,
    *,
    student_topk=20,
    candidate_limit=4,
    official_parity=None,
):
    """Aggregate v2 candidates exclusively from real query-level rankings."""
    identity_count = len(identities)
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student and teacher score matrices must have equal shape")
    if student_scores.shape != (len(query_labels), identity_count):
        raise ValueError("score matrix shape does not match queries/identities")

    student_order = torch.argsort(
        student_scores, dim=1, descending=True, stable=True
    )
    teacher_order = torch.argsort(
        teacher_scores, dim=1, descending=True, stable=True
    )
    student_inverse = torch.empty_like(student_order)
    teacher_inverse = torch.empty_like(teacher_order)
    rank_values = torch.arange(identity_count)[None, :].expand_as(student_order)
    student_inverse.scatter_(1, student_order, rank_values)
    teacher_inverse.scatter_(1, teacher_order, rank_values)
    query_rows = torch.arange(len(query_labels))
    student_positive_rank = student_inverse[query_rows, query_labels] + 1
    teacher_positive_rank = teacher_inverse[query_rows, query_labels] + 1

    aggregates = {
        index: {
            "query_count": 0,
            "student_positive_ranks": [],
            "teacher_positive_ranks": [],
            "strict": {},
            "disagreement": {},
        }
        for index in range(identity_count)
    }
    student_error_count = 0
    teacher_correct_count = 0
    strict_query_count = 0
    retained_wrong_top1_count = 0
    same_identity_negative_count = 0
    duplicate_count = 0
    rank_gaps = []

    for query_index, anchor_index_tensor in enumerate(query_labels):
        anchor_index = int(anchor_index_tensor)
        aggregate = aggregates[anchor_index]
        student_rank = int(student_positive_rank[query_index])
        teacher_rank = int(teacher_positive_rank[query_index])
        aggregate["query_count"] += 1
        aggregate["student_positive_ranks"].append(student_rank)
        aggregate["teacher_positive_ranks"].append(teacher_rank)
        student_wrong = student_rank > 1
        teacher_correct = teacher_rank == 1
        student_error_count += int(student_wrong)
        teacher_correct_count += int(teacher_correct)

        student_wrong_order = [
            int(candidate)
            for candidate in student_order[query_index].tolist()
            if int(candidate) != anchor_index
        ][:student_topk]
        same_identity_negative_count += sum(
            candidate == anchor_index for candidate in student_wrong_order
        )
        duplicate_count += len(student_wrong_order) - len(set(student_wrong_order))

        if student_wrong and teacher_correct:
            strict_query_count += 1
            top1_wrong = int(student_order[query_index, 0])
            strict = aggregate["strict"].setdefault(
                top1_wrong,
                {"query_frequency": 0, "worst_student_positive_rank": 0},
            )
            strict["query_frequency"] += 1
            strict["worst_student_positive_rank"] = max(
                strict["worst_student_positive_rank"], student_rank
            )
            retained_wrong_top1_count += 1

        if teacher_correct:
            for candidate in student_wrong_order:
                student_negative_rank = int(
                    student_inverse[query_index, candidate]
                ) + 1
                teacher_negative_rank = int(
                    teacher_inverse[query_index, candidate]
                ) + 1
                rank_gap = teacher_negative_rank - student_negative_rank
                if rank_gap <= 0:
                    continue
                rank_gaps.append(rank_gap)
                disagreement = aggregate["disagreement"].setdefault(
                    candidate,
                    {
                        "student_positive_ranks": [],
                        "student_negative_ranks": [],
                        "teacher_negative_ranks": [],
                        "rank_gaps": [],
                    },
                )
                disagreement["student_positive_ranks"].append(student_rank)
                disagreement["student_negative_ranks"].append(student_negative_rank)
                disagreement["teacher_negative_ranks"].append(teacher_negative_rank)
                disagreement["rank_gaps"].append(rank_gap)

    records = {}
    strict_covered = 0
    disagreement_covered = 0
    candidate_counts = []
    for anchor_index, anchor_pid in enumerate(identities):
        aggregate = aggregates[anchor_index]
        strict_items = [
            {
                "candidate_id": identities[candidate],
                **summary,
            }
            for candidate, summary in aggregate["strict"].items()
        ]
        strict_items.sort(
            key=lambda item: (
                -item["query_frequency"],
                -item["worst_student_positive_rank"],
                item["candidate_id"],
            )
        )
        strict_items = strict_items[:candidate_limit]

        disagreement_items = []
        for candidate, observations in aggregate["disagreement"].items():
            gaps = observations["rank_gaps"]
            student_negative = observations["student_negative_ranks"]
            teacher_negative = observations["teacher_negative_ranks"]
            disagreement_items.append(
                {
                    "candidate_id": identities[candidate],
                    "query_frequency": len(gaps),
                    "worst_student_positive_rank": max(
                        observations["student_positive_ranks"]
                    ),
                    "student_negative_rank": {
                        "min": min(student_negative),
                        "mean": float(sum(student_negative) / len(student_negative)),
                        "max": max(student_negative),
                    },
                    "teacher_negative_rank": {
                        "min": min(teacher_negative),
                        "mean": float(sum(teacher_negative) / len(teacher_negative)),
                        "max": max(teacher_negative),
                    },
                    "rank_gap": {
                        "mean": float(sum(gaps) / len(gaps)),
                        "max": max(gaps),
                    },
                }
            )
        disagreement_items.sort(
            key=lambda item: (
                -item["query_frequency"],
                -item["worst_student_positive_rank"],
                -item["rank_gap"]["mean"],
                item["candidate_id"],
            )
        )
        disagreement_items = disagreement_items[:candidate_limit]
        strict_covered += int(bool(strict_items))
        disagreement_covered += int(bool(disagreement_items))
        candidate_counts.append(len(disagreement_items))
        query_count = aggregate["query_count"]
        records[anchor_pid] = {
            "query_count": query_count,
            "student_positive_rank": {
                "min": min(aggregate["student_positive_ranks"]),
                "mean": float(
                    sum(aggregate["student_positive_ranks"]) / query_count
                ),
                "max": max(aggregate["student_positive_ranks"]),
            },
            "teacher_positive_rank": {
                "min": min(aggregate["teacher_positive_ranks"]),
                "mean": float(
                    sum(aggregate["teacher_positive_ranks"]) / query_count
                ),
                "max": max(aggregate["teacher_positive_ranks"]),
            },
            "strict_teacher_advantage_ids": [
                item["candidate_id"] for item in strict_items
            ],
            "strict_teacher_advantage": strict_items,
            "teacher_rank_disagreement": disagreement_items,
        }

    query_count = len(query_labels)
    audit = {
        "query_image_count": query_count,
        "identity_count": identity_count,
        "student_query_level_top1_error_count": student_error_count,
        "student_query_level_top1_error_ratio": float(
            student_error_count / max(query_count, 1)
        ),
        "teacher_query_level_top1_correct_count": teacher_correct_count,
        "teacher_query_level_top1_correct_ratio": float(
            teacher_correct_count / max(query_count, 1)
        ),
        "strict_teacher_correct_student_wrong_query_count": strict_query_count,
        "strict_teacher_correct_student_wrong_query_ratio": float(
            strict_query_count / max(query_count, 1)
        ),
        "strict_teacher_adv_identity_coverage_count": strict_covered,
        "strict_teacher_adv_identity_coverage_ratio": float(
            strict_covered / max(identity_count, 1)
        ),
        "rank_disagreement_identity_coverage_count": disagreement_covered,
        "rank_disagreement_identity_coverage_ratio": float(
            disagreement_covered / max(identity_count, 1)
        ),
        "student_top1_wrong_identity_retained_count": retained_wrong_top1_count,
        "student_top1_wrong_identity_retained_ratio": float(
            retained_wrong_top1_count / max(student_error_count, 1)
        ),
        "candidate_count": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "mean": float(sum(candidate_counts) / max(len(candidate_counts), 1)),
            "max": max(candidate_counts) if candidate_counts else 0,
        },
        "rank_gap": _distribution(rank_gaps),
        "same_identity_negative_count": same_identity_negative_count,
        "duplicate_count": duplicate_count,
        "official_evaluator_parity": official_parity or {},
    }
    return records, audit


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
    parser.add_argument(
        "--version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 preserves prototype mining; v2 performs query-level mining",
    )
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


def _metadata(args, identities):
    identity_hash = hashlib.sha256(
        "\n".join(identities).encode("utf-8")
    ).hexdigest()
    return {
        "data_dir": args.data_dir,
        "student_checkpoint": args.student_checkpoint,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_metrics": args.teacher_metrics,
        "identity_count": len(identities),
        "identity_hash": identity_hash,
        "student_topk": args.student_topk,
        "teacher_adv_pool_size": args.teacher_adv_pool_size,
        "descriptor_tensors_saved": False,
    }


def run_v1(args, identities, domain_items, transform, device):
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
    metadata = {
        **_metadata(args, identities),
        "version": "v1",
        "prototype_definition": "L2(image)->identity_domain_mean->L2",
    }
    return metadata, directions, audits


def run_v2(args, identities, domain_items, transform, device):
    descriptors = {"student": {}, "teacher": {}}
    labels = {}
    student = build_student(args.student_checkpoint, device)
    for domain in ("drone", "satellite"):
        descriptors["student"][domain], labels[domain] = extract_image_descriptors(
            student,
            domain_items[domain],
            transform,
            args.batch_size,
            args.num_workers,
            device,
        )
    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()

    teacher = build_teacher(args.teacher_checkpoint, args.teacher_metrics, device)
    for domain in ("drone", "satellite"):
        teacher_features, teacher_labels = extract_image_descriptors(
            teacher,
            domain_items[domain],
            transform,
            args.batch_size,
            args.num_workers,
            device,
        )
        if not torch.equal(teacher_labels, labels[domain]):
            raise RuntimeError(
                f"student/teacher {domain} image enumeration order differs"
            )
        descriptors["teacher"][domain] = teacher_features
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    directions, audits = {}, {}
    for name, query_domain, gallery_domain in (
        ("D2S", "drone", "satellite"),
        ("S2D", "satellite", "drone"),
    ):
        student_scores, student_parity = official_identity_scores(
            descriptors["student"][query_domain],
            descriptors["student"][gallery_domain],
            labels[gallery_domain],
            len(identities),
            labels[query_domain],
        )
        teacher_scores, teacher_parity = official_identity_scores(
            descriptors["teacher"][query_domain],
            descriptors["teacher"][gallery_domain],
            labels[gallery_domain],
            len(identities),
            labels[query_domain],
        )
        directions[name], audits[name] = mine_query_level_direction(
            identities,
            labels[query_domain],
            student_scores,
            teacher_scores,
            student_topk=args.student_topk,
            candidate_limit=args.teacher_adv_pool_size,
            official_parity={
                "student": student_parity,
                "teacher": teacher_parity,
            },
        )
    metadata = {
        **_metadata(args, identities),
        "version": "v2",
        "query_definition": {
            "D2S": "every drone image",
            "S2D": "every satellite image",
        },
        "gallery_identity_ranking": (
            "formal image-level similarity ranking collapsed at the first "
            "occurrence of each identity (equivalent to per-identity max)"
        ),
        "transform": "deterministic get_test_transforms",
        "cross_model_raw_score_comparison": False,
        "candidate_aggregation_source": "query-level records only",
    }
    return metadata, directions, audits


def _write_results(args, metadata, directions, audits):
    suffix = "_v2" if args.version == "v2" else ""
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, f"g4_hard_negatives{suffix}.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(
            {"metadata": metadata, "directions": directions},
            handle, indent=2, ensure_ascii=False,
        )
    with open(
        os.path.join(args.output_dir, f"g4_mining_audit{suffix}.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(
            {"metadata": metadata, "directions": audits},
            handle, indent=2, ensure_ascii=False,
        )
    print(json.dumps(audits, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    device = torch.device(args.device)
    identities, domain_items = enumerate_train_domains(args.data_dir)
    transform = get_test_transforms([args.img_size, args.img_size])
    if args.version == "v2":
        results = run_v2(args, identities, domain_items, transform, device)
    else:
        results = run_v1(args, identities, domain_items, transform, device)
    _write_results(args, *results)


if __name__ == "__main__":
    main()
