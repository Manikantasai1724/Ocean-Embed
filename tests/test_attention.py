"""Unit tests for Lightweight Attention Hybrid Model and Axial Self-Attention."""

import unittest
import torch
from oceanembed.models.attention_hybrid import AxialSelfAttention2D, AttentionHybridReconstruction
from oceanembed.train.losses import MaskedMSELoss


class TestAttentionHybrid(unittest.TestCase):
    def setUp(self):
        self.attn_layer = AxialSelfAttention2D(channels=64, num_heads=4)
        self.model = AttentionHybridReconstruction(in_channels=18, out_channels=15, hidden_dim=64, num_heads=4)

    def test_axial_attention_layer_shape(self):
        # Test decomposed axial attention on arbitrary 2D spatial dimensions
        x = torch.randn(2, 64, 32, 48)
        out = self.attn_layer(x)
        self.assertEqual(out.shape, (2, 64, 32, 48))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_parameter_count(self):
        params = self.model.get_parameter_count()
        # Strictly under 1.5M parameter CPU budget
        self.assertLess(params, 1_500_000)
        self.assertGreater(params, 50_000)

    def test_patch_forward_pass(self):
        # 64x64 patch forward
        x = torch.randn(2, 18, 64, 64)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 15, 64, 64))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_full_domain_forward_pass(self):
        # Verify axial attention functions seamlessly on full 101x241 grid
        x_full = torch.randn(1, 18, 101, 241)
        out_full = self.model(x_full)
        self.assertEqual(out_full.shape, (1, 15, 101, 241))
        self.assertTrue(torch.all(torch.isfinite(out_full)))

    def test_backward_and_gradients(self):
        x = torch.randn(2, 18, 32, 32)
        target = torch.randn(2, 15, 32, 32)
        mask = (torch.rand(2, 15, 32, 32) > 0.3).float()

        pred = self.model(x)
        criterion = MaskedMSELoss()
        loss = criterion(pred, target, mask)

        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")


if __name__ == "__main__":
    unittest.main()
