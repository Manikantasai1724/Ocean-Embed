"""Loss functions for OceanEmbed subsurface reconstructions.

Strictly adheres to:
- Rule 3.4: Masked MSE loss on valid ocean and depth points only.
- Missing and subterranean points are masked, never penalized as zero.
"""

from typing import Optional
import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """Mean Squared Error computed strictly over valid ocean and water-column depth points."""

    def __init__(self, depth_weights: Optional[torch.Tensor] = None, eps: float = 1e-6):
        """Args:

        depth_weights: optional 1D tensor of shape (15,) giving relative weight per depth
        eps: epsilon for numerical stability
        """
        super().__init__()
        self.depth_weights = depth_weights
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:

        pred: (B, 15, H, W) predicted values
        target: (B, 15, H, W) ground truth values
        mask: (B, 15, H, W) binary mask (1.0 for valid water column, 0.0 otherwise)
        """
        diff_sq = (pred - target) ** 2

        if self.depth_weights is not None:
            # Broadcast depth_weights across batch and spatial dims
            weights = self.depth_weights.view(1, -1, 1, 1).to(pred.device)
            diff_sq = diff_sq * weights
            mask = mask * weights

        masked_loss = diff_sq * mask
        total_valid = torch.sum(mask)

        loss = torch.sum(masked_loss) / (total_valid + self.eps)
        return loss


class MaskedSmoothL1Loss(nn.Module):
    """Smooth L1 (Huber) loss computed over valid points (more robust to extreme eddy outliers)."""

    def __init__(self, beta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        cond = diff < self.beta
        loss_val = torch.where(cond, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta)
        masked_loss = loss_val * mask
        return torch.sum(masked_loss) / (torch.sum(mask) + self.eps)
