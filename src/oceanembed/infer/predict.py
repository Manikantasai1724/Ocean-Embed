"""Inference Engine for OceanEmbed 3D Ocean Temperature Reconstruction.

Supports:
- Attention Hybrid Model (default, state-of-the-art axial attention)
- Direct CNN Baseline
- Climatology Baseline
- In-memory caching for sub-100ms API response latency
- 2D horizontal slices and click-a-point 1D vertical profiles (0-1000m)
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import zarr

from oceanembed.models.baselines import ClimatologyBaseline
from oceanembed.models.cnn_baseline import DirectCNNReconstruction
from oceanembed.models.attention_hybrid import AttentionHybridReconstruction
from oceanembed.train.train_cnn import AnomalyPatchDataset
from oceanembed.data.masks import create_realistic_nio_bathymetry


class OceanPredictor:
    """Manages inference, model loading, and feature extraction."""

    def __init__(
        self,
        zarr_path: str = "data/processed/oceanembed_slice_201801.zarr",
        checkpoint_dir: str = "checkpoints",
    ):
        self.zarr_path = zarr_path
        self.checkpoint_dir = checkpoint_dir

        self.root = zarr.open_group(zarr_path, mode="r")
        self.times = [str(t) for t in self.root["time"][:]]
        self.depths = [float(d) for d in self.root["depth"][:]]
        self.lats = self.root["latitude"][:]
        self.lons = self.root["longitude"][:]
        self.mask_depth = self.root["mask_depth"][:] == 1

        # Realistic bathymetry
        self.bathymetry, self.is_ocean = create_realistic_nio_bathymetry(self.lats, self.lons)

        # 1. Climatology baseline (fit on training partition)
        n_train = int(len(self.times) * 0.70)
        train_indices = list(range(0, n_train))
        self.clim_model = ClimatologyBaseline(
            n_depths=len(self.depths),
            n_lat=len(self.lats),
            n_lon=len(self.lons),
        )
        self.clim_model.fit(
            [self.times[i] for i in train_indices],
            self.root["temperature"][train_indices],
            self.mask_depth,
        )

        # 2. Dataset for normalization statistics & consistent feature extraction
        self.train_dataset = AnomalyPatchDataset(
            zarr_path, self.clim_model, train_indices, patch_size=64, is_train=True
        )
        self.full_dataset = AnomalyPatchDataset(
            zarr_path,
            self.clim_model,
            list(range(len(self.times))),
            patch_size=64,
            is_train=False,
            normalization_stats=self.train_dataset.stats,
        )

        # 3. Models
        self.models: Dict[str, Any] = {}
        self._load_models()

        # 4. In-memory prediction cache: (date, model_name) -> 3D ndarray (15, 101, 241)
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

    def _load_models(self) -> None:
        """Loads available model checkpoints onto CPU."""
        # Attention Hybrid
        attn_path = os.path.join(self.checkpoint_dir, "attention_hybrid_best.pt")
        if os.path.exists(attn_path):
            ckpt = torch.load(attn_path, map_location="cpu", weights_only=False)
            model = AttentionHybridReconstruction(in_channels=18, out_channels=15, hidden_dim=64, num_heads=4)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.models["attention_hybrid"] = model
            print("[OceanPredictor] Loaded Attention Hybrid checkpoint.")

        # Direct CNN
        cnn_path = os.path.join(self.checkpoint_dir, "direct_cnn_best.pt")
        if os.path.exists(cnn_path):
            ckpt = torch.load(cnn_path, map_location="cpu", weights_only=False)
            model = DirectCNNReconstruction(in_channels=18, out_channels=15)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.models["direct_cnn"] = model
            print("[OceanPredictor] Loaded Direct CNN checkpoint.")

    def get_metadata(self) -> Dict[str, Any]:
        """Returns coordinate grids, depth levels, and time bounds."""
        return {
            "domain_name": "North Indian Ocean",
            "latitudes": [round(float(lat), 3) for lat in self.lats],
            "longitudes": [round(float(lon), 3) for lon in self.lons],
            "target_depths": self.depths,
            "available_dates": self.times,
            "available_models": ["climatology"] + list(self.models.keys()),
            "grid_shape": [len(self.lats), len(self.lons)],
        }

    def _get_input_tensor(self, date_idx: int) -> torch.Tensor:
        """Returns normalized 18-channel surface input tensor from dataset."""
        x_val, _, _ = self.full_dataset[date_idx]
        return x_val.unsqueeze(0)  # (1, 18, 101, 241)

    def predict_3d(self, date: str, model_name: str = "attention_hybrid") -> np.ndarray:
        """Returns 3D temperature field (15, H, W) with land/subterranean masked as NaN."""
        cache_key = (date, model_name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if date not in self.times:
            raise ValueError(f"Date {date} not found. Must be one of {self.times[0]} to {self.times[-1]}")

        date_idx = self.times.index(date)
        clim_field = self.clim_model.predict_for_date(date)

        if model_name == "climatology" or model_name not in self.models:
            field_3d = np.copy(clim_field)
        else:
            model = self.models[model_name]
            x_tensor = self._get_input_tensor(date_idx)
            with torch.no_grad():
                pred_anomaly = model(x_tensor).squeeze(0).numpy()
            field_3d = pred_anomaly + clim_field

        # Mask subterranean and land cells
        for k in range(len(self.depths)):
            valid_k = self.mask_depth[k]
            field_3d[k, ~valid_k] = np.nan

        self._cache[cache_key] = field_3d
        return field_3d

    def get_field_slice(
        self,
        date: str,
        depth_m: float,
        model_name: str = "attention_hybrid",
    ) -> Dict[str, Any]:
        """Returns 2D horizontal slice for web dashboard visualization."""
        field_3d = self.predict_3d(date, model_name)

        # Find closest depth level
        k = int(np.argmin([abs(d - depth_m) for d in self.depths]))
        slice_2d = field_3d[k]

        valid_vals = slice_2d[~np.isnan(slice_2d)]
        v_min = float(np.min(valid_vals)) if len(valid_vals) > 0 else 0.0
        v_max = float(np.max(valid_vals)) if len(valid_vals) > 0 else 30.0

        grid_data = [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in slice_2d]

        return {
            "date": date,
            "depth_m": self.depths[k],
            "model_name": model_name,
            "grid": grid_data,
            "min_temp": round(v_min, 2),
            "max_temp": round(v_max, 2),
            "units": "°C",
        }

    def get_profile(
        self,
        date: str,
        lat: float,
        lon: float,
        model_name: str = "attention_hybrid",
    ) -> Dict[str, Any]:
        """Extracts vertical temperature profile at a point (0-1000m)."""
        lat_idx = int(np.argmin(np.abs(self.lats - lat)))
        lon_idx = int(np.argmin(np.abs(self.lons - lon)))
        actual_lat = float(self.lats[lat_idx])
        actual_lon = float(self.lons[lon_idx])

        # Prediction
        pred_3d = self.predict_3d(date, model_name)
        pred_prof = pred_3d[:, lat_idx, lon_idx]

        # Ground truth
        date_idx = self.times.index(date)
        truth_3d = self.root["temperature"][date_idx]
        truth_prof = truth_3d[:, lat_idx, lon_idx]

        # Climatology
        clim_3d = self.clim_model.predict_for_date(date)
        clim_prof = clim_3d[:, lat_idx, lon_idx]

        seafloor_depth = float(self.bathymetry[lat_idx, lon_idx])
        is_water = seafloor_depth > 0

        profile_data = []
        for k, d in enumerate(self.depths):
            is_valid_level = self.mask_depth[k, lat_idx, lon_idx]
            profile_data.append({
                "depth_m": d,
                "predicted_temp": round(float(pred_prof[k]), 2) if is_valid_level and not np.isnan(pred_prof[k]) else None,
                "truth_temp": round(float(truth_prof[k]), 2) if is_valid_level and not np.isnan(truth_prof[k]) else None,
                "climatology_temp": round(float(clim_prof[k]), 2) if is_valid_level and not np.isnan(clim_prof[k]) else None,
            })

        return {
            "requested_coord": {"lat": lat, "lon": lon},
            "snapped_coord": {"lat": actual_lat, "lon": actual_lon},
            "is_ocean": is_water,
            "seafloor_depth_m": round(seafloor_depth, 1),
            "date": date,
            "model_name": model_name,
            "profile": profile_data,
        }

    def compare_slice(
        self,
        date: str,
        depth_m: float,
        model_name: str = "attention_hybrid",
    ) -> Dict[str, Any]:
        """Returns side-by-side Truth, Prediction, and Error slice."""
        k = int(np.argmin([abs(d - depth_m) for d in self.depths]))
        date_idx = self.times.index(date)

        pred_3d = self.predict_3d(date, model_name)
        pred_2d = pred_3d[k]

        truth_2d = self.root["temperature"][date_idx, k]
        mask_k = self.mask_depth[k]

        pred_2d_masked = np.copy(pred_2d)
        truth_2d_masked = np.copy(truth_2d)
        pred_2d_masked[~mask_k] = np.nan
        truth_2d_masked[~mask_k] = np.nan

        diff_2d = pred_2d_masked - truth_2d_masked

        valid = mask_k & (~np.isnan(pred_2d_masked)) & (~np.isnan(truth_2d_masked))
        rmse = float(np.sqrt(np.mean(diff_2d[valid] ** 2))) if np.sum(valid) > 0 else 0.0

        return {
            "date": date,
            "depth_m": self.depths[k],
            "model_name": model_name,
            "rmse_c": round(rmse, 3),
            "prediction_grid": [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in pred_2d_masked],
            "truth_grid": [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in truth_2d_masked],
            "error_grid": [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in diff_2d],
        }
