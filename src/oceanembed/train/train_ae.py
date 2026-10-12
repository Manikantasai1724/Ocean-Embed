"""Self-supervised autoencoder pretraining for OceanEmbed.

Pretrains the SatelliteAutoencoder strictly on the TRAINING period (2018-01-01 to 2018-01-21)
to reconstruct the 7 surface satellite fields from compact 64-channel embeddings.

Strictly adheres to:
- Rule 3.5: Pretrain embeddings on TRAINING period only.
- Masked loss over valid ocean cells.
"""

import os
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import zarr

from oceanembed.models.autoencoder import SatelliteAutoencoder
from oceanembed.train.train_cnn import set_seed


class SurfaceAutoencoderDataset(Dataset):
    """Dataset extracting 64x64 patches for self-supervised surface reconstruction."""

    def __init__(
        self,
        zarr_path: str,
        time_indices: List[int],
        patch_size: int = 64,
        is_train: bool = True,
    ):
        self.root = zarr.open_group(zarr_path, mode="r")
        self.time_indices = time_indices
        self.patch_size = patch_size
        self.is_train = is_train

        self.all_times = [str(t) for t in self.root["time"][:]]
        self.lats = self.root["latitude"][:]
        self.lons = self.root["longitude"][:]
        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)

        self.input_vars = ["sst", "sss", "sla", "u_cur", "v_cur", "u_wind", "v_wind"]

        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        self.lat_norm = (lat_grid - 17.5) / 12.5
        self.lon_norm = (lon_grid - 75.0) / 30.0

        self.mask_ocean = self.root["mask_ocean"][:]

        # Compute stats on training indices
        self.stats = {}
        for var in self.input_vars:
            data = self.root[var][self.time_indices]
            valid = ~np.isnan(data)
            mean = float(np.mean(data[valid])) if np.any(valid) else 0.0
            std = float(np.std(data[valid])) if np.any(valid) else 1.0
            self.stats[var] = (mean, max(std, 1e-6))

    def __len__(self) -> int:
        return len(self.time_indices)

    def __getitem__(self, idx: int):
        t_global = self.time_indices[idx]
        d_str = self.all_times[t_global]

        if self.is_train and self.patch_size < min(self.n_lat, self.n_lon):
            top = np.random.randint(0, self.n_lat - self.patch_size + 1)
            left = np.random.randint(0, self.n_lon - self.patch_size + 1)
            h_slice = slice(top, top + self.patch_size)
            w_slice = slice(left, left + self.patch_size)
        else:
            h_slice = slice(0, self.n_lat)
            w_slice = slice(0, self.n_lon)

        input_channels = []
        mask_channels = []
        target_surface = []

        for var in self.input_vars:
            raw = self.root[var][t_global, h_slice, w_slice].copy()
            valid = ~np.isnan(raw)
            mean, std = self.stats[var]

            normed = np.zeros_like(raw, dtype=np.float32)
            normed[valid] = (raw[valid] - mean) / std

            input_channels.append(normed)
            mask_channels.append(valid.astype(np.float32))
            target_surface.append(normed)  # Target is the normalized surface field

        lat_patch = self.lat_norm[h_slice, w_slice].astype(np.float32)
        lon_patch = self.lon_norm[h_slice, w_slice].astype(np.float32)

        parts = d_str.split("-")
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        day_of_year = month * 30 + day

        sin_doy = np.full_like(lat_patch, np.sin(2 * np.pi * day_of_year / 365.25), dtype=np.float32)
        cos_doy = np.full_like(lat_patch, np.cos(2 * np.pi * day_of_year / 365.25), dtype=np.float32)

        all_inputs = input_channels + mask_channels + [lat_patch, lon_patch, sin_doy, cos_doy]
        x = torch.from_numpy(np.stack(all_inputs, axis=0)).float()

        # Target surface: 7 channels
        y_surf = torch.from_numpy(np.stack(target_surface, axis=0)).float()
        # Ocean mask: (7, H, W)
        surf_mask = torch.from_numpy(np.stack(mask_channels, axis=0)).float()

        return x, y_surf, surf_mask


def pretrain_autoencoder(
    zarr_path: str = "data/processed/oceanembed_slice_201801.zarr",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
) -> str:
    """Pretrains SatelliteAutoencoder on training slice."""
    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    root = zarr.open_group(zarr_path, mode="r")

    n_times = root["time"].shape[0]
    n_train = int(n_times * 0.70)
    train_indices = list(range(0, n_train))

    dataset = SurfaceAutoencoderDataset(zarr_path, train_indices, patch_size=64, is_train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SatelliteAutoencoder(in_channels=18, out_channels=7, latent_channels=64)
    param_count = model.get_parameter_count()
    print(f"\n[OceanEmbed Autoencoder] Pretraining Autoencoder ({param_count:,} parameters) on training split only...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")
    checkpoint_path = os.path.join(checkpoint_dir, "autoencoder_pretrained.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for x, y_surf, surf_mask in loader:
            optimizer.zero_grad()
            _, recon = model(x)

            # Masked MSE on valid surface cells
            diff_sq = (recon - y_surf) ** 2
            loss = torch.sum(diff_sq * surf_mask) / (torch.sum(surf_mask) + 1e-6)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(1, len(loader))
        epoch_time = time.time() - t0
        print(f"AE Epoch {epoch:2d}/{epochs:2d} | Surface Recon Loss: {epoch_loss:.5f} | Time: {epoch_time:.2f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "latent_channels": 64,
                "loss": epoch_loss,
                "stats": dataset.stats,
            }, checkpoint_path)

    print(f"[OceanEmbed Autoencoder] Saved best pretrained autoencoder checkpoint to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    pretrain_autoencoder()
