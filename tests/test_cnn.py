"""Unit tests for Phase 3 Direct CNN Baseline model and loss functions."""

import unittest
import torch
from oceanembed.models.cnn_baseline import DirectCNNReconstruction
from oceanembed.train.losses import MaskedMSELoss


class TestDirectCNN(unittest.TestCase):
    def setUp(self):
        self.model = DirectCNNReconstruction(in_channels=18, out_channels=15)

    def test_parameter_count(self):
        params = self.model.get_parameter_count()
        # Strictly under 2M parameter cap for CPU execution (Rule 3.4)
        self.assertLess(params, 2_000_000)
        self.assertGreater(params, 100_000)

    def test_patch_forward_pass(self):
        # 64x64 patch forward
        x = torch.randn(2, 18, 64, 64)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 15, 64, 64))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_full_domain_forward_pass(self):
        # Fully convolutional test: runs directly on 101x241 grid without spatial resizing
        x_full = torch.randn(1, 18, 101, 241)
        out_full = self.model(x_full)
        self.assertEqual(out_full.shape, (1, 15, 101, 241))

    def test_masked_loss_and_gradients(self):
        pred = self.model(torch.randn(2, 18, 64, 64))
        target = torch.randn(2, 15, 64, 64)
        mask = (torch.rand(2, 15, 64, 64) > 0.4).float()

        criterion = MaskedMSELoss()
        loss = criterion(pred, target, mask)

        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        # Check gradients exist
        for p in self.model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)


if __name__ == "__main__":
    unittest.main()
