"""Direct CNN baseline architecture for OceanEmbed.

Implements a fully convolutional residual network (FCN) predicting 15 subsurface
temperature anomaly channels directly from surface satellite observation channels.

Strictly adheres to:
- Rule 3.4: Under 2M parameters (~535k parameters), CPU-friendly, patch-based training.
- Fully convolutional: operates on 64x64 patches during training and arbitrary
  dimensions (101x241 full domain) during inference.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + GELU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and identity skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.act(out)


class DirectCNNReconstruction(nn.Module):
    """Fully Convolutional Network for 3D ocean temperature reconstruction.

    Input channels: 18
      - 7 normalized surface observations [SST, SSS, SLA, U_cur, V_cur, U_wind, V_wind]
      - 7 observation availability masks [mask_sst, ..., mask_wind]
      - 4 static features [lat, lon, sin_doy, cos_doy]

    Output channels: 15
      - Reconstructed temperature anomalies at 15 depth levels
    """

    def __init__(
        self,
        in_channels: int = 18,
        out_channels: int = 15,
        hidden_dims: List[int] = [64, 128, 128, 64],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Stem
        self.stem = ConvBlock(in_channels, hidden_dims[0])
        self.res1 = ResidualBlock(hidden_dims[0])

        # Stage 2
        self.conv2 = ConvBlock(hidden_dims[0], hidden_dims[1])
        self.res2 = ResidualBlock(hidden_dims[1])

        # Stage 3
        self.conv3 = ConvBlock(hidden_dims[1], hidden_dims[2])
        self.res3 = ResidualBlock(hidden_dims[2])

        # Stage 4 & Head
        self.conv4 = ConvBlock(hidden_dims[2], hidden_dims[3])
        self.head = nn.Conv2d(hidden_dims[3], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, in_channels, H, W)

        Returns:
            out: Tensor of shape (B, out_channels, H, W)
        """
        feat = self.stem(x)
        feat = self.res1(feat)

        feat = self.conv2(feat)
        feat = self.res2(feat)

        feat = self.conv3(feat)
        feat = self.res3(feat)

        feat = self.conv4(feat)
        out = self.head(feat)
        return out

    def get_parameter_count(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
