"""Satellite Surface Embedding Autoencoder.

Implements self-supervised pretraining for remote-sensing surface fields:
1. SatelliteEncoder: maps 18 surface input channels to a low-dimensional bottleneck
   ("satellite embedding") with configurable dimension and spatial downsampling.
2. SatelliteDecoder: reconstructs the 7 observed surface physical fields.
3. SatelliteAutoencoder: end-to-end self-supervised pretraining model.

Strictly adheres to:
- Rule 3.5: Pretrain embeddings on TRAINING period only.
- CPU-friendly, modular, pluggable architecture under 1.5M parameters.
- Robust to odd spatial dimensions (e.g. 101x241 grid) via exact target-size interpolation.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SatelliteEncoder(nn.Module):
    """Convolutional encoder extracting compact spatial embeddings from surface fields."""

    def __init__(
        self,
        in_channels: int = 18,
        latent_channels: int = 64,
        hidden_dims: List[int] = [32, 64, 128],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Stem: (B, 18, H, W) -> (B, 32, H, W)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.GELU(),
        )

        # Downsample Stage 1: (B, 32, H, W) -> (B, 64, H/2, W/2)
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.GELU(),
            nn.Conv2d(hidden_dims[1], hidden_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.GELU(),
        )

        # Downsample Stage 2: (B, 64, H/2, W/2) -> (B, 128, H/4, W/4)
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.GELU(),
            nn.Conv2d(hidden_dims[2], hidden_dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.GELU(),
        )

        # Bottleneck projection: (B, 128, H/4, W/4) -> (B, latent_channels, H/4, W/4)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dims[2], latent_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the bottleneck embedding: shape (B, latent_channels, H/4, W/4)."""
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        embedding = self.bottleneck(x)
        return embedding


class SatelliteDecoder(nn.Module):
    """Convolutional decoder reconstructing the 7 physical surface fields from embeddings."""

    def __init__(
        self,
        latent_channels: int = 64,
        out_channels: int = 7,
        hidden_dims: List[int] = [128, 64, 32],
    ):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.GELU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.GELU(),
            nn.Conv2d(hidden_dims[2], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, embedding: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Reconstructs surface fields: shape (B, 7, H, W)."""
        x = self.up1(embedding)
        x = self.up2(x)
        reconstruction = self.head(x)
        if target_size is not None and reconstruction.shape[-2:] != target_size:
            reconstruction = F.interpolate(reconstruction, size=target_size, mode="bilinear", align_corners=False)
        return reconstruction


class SatelliteAutoencoder(nn.Module):
    """Full autoencoder for self-supervised surface-field reconstruction."""

    def __init__(
        self,
        in_channels: int = 18,
        out_channels: int = 7,
        latent_channels: int = 64,
    ):
        super().__init__()
        self.encoder = SatelliteEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self.decoder = SatelliteDecoder(latent_channels=latent_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_size = (x.shape[-2], x.shape[-1])
        embedding = self.encoder(x)
        recon = self.decoder(embedding, target_size=target_size)
        return embedding, recon

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
