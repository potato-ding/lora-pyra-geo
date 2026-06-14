import os
import sys

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.student_model import StudentModel
from src.training.student_train import compute_brd_loss, compute_local_align_loss


def test_student_baseline_forward_outputs_normalized_f4_embedding():
    torch.manual_seed(7)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        embedding = model(x)

    assert embedding.shape == (2, 512)
    torch.testing.assert_close(
        embedding.norm(p=2, dim=1),
        torch.ones(2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_student_baseline_has_no_confidence_or_adapter_modules():
    model = StudentModel(ckpt_path=None).eval()

    forbidden_attrs = [
        "pool_type",
        "conf_pool_tau",
        "conf_alpha_max",
        "conf_apply_views",
        "drone_conf_head",
        "sat_conf_head",
        "alpha_raw_drone",
        "alpha_raw_sat",
        "view_adapter",
    ]
    for attr in forbidden_attrs:
        assert not hasattr(model, attr)


def test_student_baseline_matches_explicit_gap_bn_normalize_path():
    torch.manual_seed(13)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        embedding = model(x)
        _, _, _, f4 = model.backbone(x)
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        expected = F.normalize(model.neck(desc), dim=1)

    torch.testing.assert_close(embedding, expected, atol=1e-6, rtol=1e-6)


def test_student_return_fmap_keeps_default_embedding_path():
    torch.manual_seed(23)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        default_embedding = model(x)
        embedding, f4 = model(x, return_fmap=True)
        _, _, _, expected_f4 = model.backbone(x)

    assert embedding.shape == (2, 512)
    assert f4.shape[0] == 2
    assert f4.shape[1] == 512
    torch.testing.assert_close(default_embedding, embedding, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(f4, expected_f4, atol=1e-6, rtol=1e-6)


def test_local_align_loss_is_batch_level_contrastive():
    torch.manual_seed(31)
    fmap_drone = torch.randn(3, 512, 2, 3)
    fmap_sat = torch.randn(3, 512, 2, 3)

    loss, local_score, labels = compute_local_align_loss(
        fmap_drone,
        fmap_sat,
        topk=4,
        tau=0.07,
        return_debug=True,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert local_score.shape == (3, 3)
    torch.testing.assert_close(labels, torch.arange(3))


def test_brd_loss_is_ranking_level_and_backpropagates_to_student_only():
    class Args:
        brd_temperature = 0.07
        brd_risk_margin = 0.0
        brd_risk_tau = 0.05

    torch.manual_seed(37)
    raw_student_features = torch.randn(6, 512, requires_grad=True)
    student_features = F.normalize(raw_student_features, dim=1)
    teacher_features = F.normalize(torch.randn(6, 768), dim=1)

    loss, stats = compute_brd_loss(student_features, teacher_features, pair_batch_size=3, args=Args)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert raw_student_features.grad is not None
    assert teacher_features.grad is None
    assert set(stats) == {"loss_brd_d2s", "loss_brd_s2d", "brd_risk_d2s", "brd_risk_s2d"}
