import torch

from src.training.audit_negrank_selection import (
    audit_direction,
    deterministic_topk_indices,
    half_up_candidate_count,
)


def test_half_up_candidate_counts_for_31_negatives():
    assert half_up_candidate_count(0.25, 31) == 8
    assert half_up_candidate_count(0.50, 31) == 16
    assert half_up_candidate_count(0.75, 31) == 23
    assert half_up_candidate_count(1.00, 31) == 31


def test_deterministic_topk_breaks_confidence_ties_by_candidate_index():
    confidence = torch.tensor([0.5, 0.8, 0.8, 0.2])
    assert deterministic_topk_indices(confidence, 3).tolist() == [1, 2, 0]


def test_selection_is_per_anchor_and_full_softmax_mass_is_preserved_at_sel100():
    similarity = torch.tensor(
        [
            [1.0, 0.9, 0.2, -0.3],
            [0.8, 1.0, 0.1, -0.4],
            [0.3, 0.0, 1.0, -0.2],
            [-0.1, -0.5, -0.4, 1.0],
        ]
    )
    records = audit_direction(similarity, temperature=0.2, batch_index=1)

    assert records["SEL100"]["selected_candidate_count"] == 12
    assert records["SEL100"]["all_candidate_count"] == 12
    assert all(
        abs(mass - 1.0) < 1e-6
        for mass in records["SEL100"]["retained_probability_mass"]
    )
    assert len(records["SEL50"]["selected_candidates"]) == 4
    assert all(
        len(item["selected_global_candidate_indices"]) == 2
        for item in records["SEL50"]["selected_candidates"]
    )
