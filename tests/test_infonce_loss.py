import math
import os
import sys
import unittest

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.loss.blocks_infoNCE import infonce


class InfonceLossTest(unittest.TestCase):
    def test_matching_diagonal_pairs_have_lower_loss(self):
        criterion = infonce()
        sat_feats = torch.eye(4, dtype=torch.float32)
        drone_feats = sat_feats.clone()
        wrong_drone_feats = torch.roll(drone_feats, shifts=1, dims=0)
        logit_scale = torch.tensor(math.log(10.0), dtype=torch.float32)

        good_loss = criterion(sat_feats, drone_feats, logit_scale)
        bad_loss = criterion(sat_feats, wrong_drone_feats, logit_scale)

        self.assertLess(good_loss.item(), bad_loss.item())

    def test_requires_one_to_one_pair_count(self):
        criterion = infonce()
        logit_scale = torch.tensor(math.log(10.0), dtype=torch.float32)

        with self.assertRaises(ValueError):
            criterion(torch.randn(3, 8), torch.randn(4, 8), logit_scale)

    def test_backward_keeps_feature_gradients(self):
        criterion = infonce()
        sat_feats = torch.randn(4, 8, requires_grad=True)
        drone_feats = torch.randn(4, 8, requires_grad=True)
        logit_scale = torch.tensor(math.log(10.0), dtype=torch.float32, requires_grad=True)

        loss = criterion(sat_feats, drone_feats, logit_scale)
        loss.backward()

        self.assertIsNotNone(sat_feats.grad)
        self.assertIsNotNone(drone_feats.grad)
        self.assertIsNotNone(logit_scale.grad)

    def test_directional_runtime_telemetry_does_not_change_returned_loss(self):
        criterion = infonce()
        sat_feats = torch.eye(4, dtype=torch.float32)
        drone_feats = sat_feats.clone()
        logit_scale = torch.tensor(math.log(10.0), dtype=torch.float32)

        loss = criterion(sat_feats, drone_feats, logit_scale)

        self.assertIsNotNone(criterion.last_loss_d2s)
        self.assertIsNotNone(criterion.last_loss_s2d)
        expected = (criterion.last_loss_d2s + criterion.last_loss_s2d) / 2.0
        self.assertTrue(torch.allclose(loss.detach(), expected))
        self.assertEqual(criterion.last_runtime_audit["similarity_logits_dtype"], "float32")
        self.assertEqual(criterion.last_runtime_audit["similarity_logits_shape"], (4, 4))


if __name__ == "__main__":
    unittest.main()
