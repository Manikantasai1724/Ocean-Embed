"""Supervised Subsurface Temperature Model on Satellite Embeddings.

Combines the SatelliteEncoder with a 15-depth reconstruction head.
Supports:
1. Frozen pretrained embeddings: tests if self-supervised representations carry
   subsurface information without modifying encoder weights.
2. Fine-tuned embeddings: updates encoder and decoder end-to-end.
3. Preserves exact spatial dimensions (e.g. 101x241 grid) during full-domain inference.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from oceanembed.models.autoencoder import SatelliteEncoder


class SubsurfacePredictionHead(nn.Module):
    """Decodes latent embeddings into 15 vertical temperature anomaly layers."""

    def __init__(self, latent_channels: int = 64, out_depths: int = 15):
        super().__init__()
        # Upsample Stage 1: (B, latent, H/4, W/4) -> (B, 128, H/2, W/2)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Upsample Stage 2: (B, 128, H/2, W/2) -> (B, 64, H, W)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Output projection: (B, 64, H, W) -> (B, 15, H, W)
        self.out_conv = nn.Conv2d(64, out_depths, kernel_size=3, padding=1)

    def forward(self, embedding: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = self.up1(embedding)
        x = self.up2(x)
        out = self.out_conv(x)
        if target_size is not None and out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class SupervisedEmbeddingModel(nn.Module):
    """Complete Supervised Model mapping surface observations to 15-depth anomalies via embeddings."""

    def __init__(
        self,
        encoder: Optional[SatelliteEncoder] = None,
        in_channels: int = 18,
        latent_channels: int = 64,
        out_depths: int = 15,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = SatelliteEncoder(in_channels=in_channels, latent_channels=latent_channels)

        self.head = SubsurfacePredictionHead(latent_channels=latent_channels, out_depths=out_depths)
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            self.freeze_encoder_weights()

    def freeze_encoder_weights(self) -> None:
        """Freezes all encoder parameters for frozen-embedding evaluation."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze_encoder_weights(self) -> None:
        """Unfreezes encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[-2], x.shape[-1])
        if self.freeze_encoder:
            with torch.no_grad():
                embedding = self.encoder(x)
        else:
            embedding = self.encoder(x)

        pred_anomaly = self.head(embedding, target_size=target_size)
        return pred_anomaly

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
