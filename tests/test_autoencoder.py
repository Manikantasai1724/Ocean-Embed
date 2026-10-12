"""Unit tests for Phase 4 Satellite Autoencoder and Supervised Embedding models."""

import unittest
import torch
from oceanembed.models.autoencoder import SatelliteEncoder, SatelliteDecoder, SatelliteAutoencoder
from oceanembed.models.unet_embed import SupervisedEmbeddingModel, SubsurfacePredictionHead


class TestEmbeddingModels(unittest.TestCase):
    def test_encoder_and_decoder_shapes(self):
        encoder = SatelliteEncoder(in_channels=18, latent_channels=64)
        decoder = SatelliteDecoder(latent_channels=64, out_channels=7)

        x = torch.randn(2, 18, 64, 64)
        emb = encoder(x)
        self.assertEqual(emb.shape, (2, 64, 16, 16))

        recon = decoder(emb)
        self.assertEqual(recon.shape, (2, 7, 64, 64))

    def test_autoencoder_roundtrip(self):
        ae = SatelliteAutoencoder(in_channels=18, out_channels=7, latent_channels=64)
        x = torch.randn(2, 18, 64, 64)
        emb, recon = ae(x)

        self.assertEqual(emb.shape, (2, 64, 16, 16))
        self.assertEqual(recon.shape, (2, 7, 64, 64))
        self.assertLess(ae.get_parameter_count(), 1_500_000)

    def test_supervised_embedding_freezing(self):
        model = SupervisedEmbeddingModel(in_channels=18, latent_channels=64, freeze_encoder=True)
        x = torch.randn(2, 18, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 15, 64, 64))

        # Check encoder weights have requires_grad == False
        for p in model.encoder.parameters():
            self.assertFalse(p.requires_grad)

        # Head weights must have requires_grad == True
        for p in model.head.parameters():
            self.assertTrue(p.requires_grad)

    def test_full_domain_preservation(self):
        # 101x241 odd dimension test
        model = SupervisedEmbeddingModel(in_channels=18, latent_channels=64, freeze_encoder=False)
        x_full = torch.randn(1, 18, 101, 241)
        out_full = model(x_full)
        self.assertEqual(out_full.shape, (1, 15, 101, 241))


if __name__ == "__main__":
    unittest.main()
