"""Baseline model implementations for OceanEmbed.

Implements:
1. Climatology baseline (mean day-of-year profile per depth level and pixel).
2. Ridge / Linear regression baseline from surface satellite inputs to 15 depths.

Strictly adheres to:
- Rule 3.3: Fit on training period only, applied unchanged to validation and test.
- Rule 3.4: Mandatory model order: (1) Climatology, (2) Ridge/linear.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
from sklearn.linear_model import Ridge


class ClimatologyBaseline:
    """Climatology Baseline: day-of-year mean temperature per depth level and spatial cell."""

    def __init__(self, n_depths: int = 15, n_lat: int = 101, n_lon: int = 241):
        self.n_depths = n_depths
        self.n_lat = n_lat
        self.n_lon = n_lon
        # Climatology table: key is doy (1..366), value is array of shape (n_depths, n_lat, n_lon)
        self.climatology_map: Dict[int, np.ndarray] = {}
        # Global depth mean fallback
        self.overall_depth_mean = np.zeros(n_depths, dtype=np.float32)

    def fit(self, dates: List[str], target_temperatures: np.ndarray, mask_depth: np.ndarray) -> None:
        """Fits climatology on training split only.

        Args:
            dates: list of T date strings (YYYY-MM-DD)
            target_temperatures: array of shape (T, n_depths, n_lat, n_lon)
            mask_depth: boolean array of shape (n_depths, n_lat, n_lon)
        """
        doy_buckets: Dict[int, List[np.ndarray]] = {}

        for t_idx, d_str in enumerate(dates):
            parts = d_str.split("-")
            month = int(parts[1]) if len(parts) > 1 else 1
            day = int(parts[2]) if len(parts) > 2 else 1
            doy = month * 30 + day  # Approximate DOY bucket

            if doy not in doy_buckets:
                doy_buckets[doy] = []
            doy_buckets[doy].append(target_temperatures[t_idx])

        # Compute mean per DOY bucket
        for doy, samples in doy_buckets.items():
            stacked = np.stack(samples, axis=0)  # (N, 15, H, W)
            mean_field = np.nanmean(stacked, axis=0)
            self.climatology_map[doy] = mean_field.astype(np.float32)

        # Compute overall mean per depth across all training samples
        all_stacked = np.stack(list(self.climatology_map.values()), axis=0)  # (K, 15, H, W)
        for k in range(self.n_depths):
            vals = all_stacked[:, k, :, :]
            valid = ~np.isnan(vals)
            self.overall_depth_mean[k] = float(np.mean(vals[valid])) if np.any(valid) else 15.0

    def predict_for_date(self, date_str: str) -> np.ndarray:
        """Predicts the 3D temperature field for a given date using climatology.

        Returns array of shape (n_depths, n_lat, n_lon).
        """
        parts = date_str.split("-")
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        doy = month * 30 + day

        if doy in self.climatology_map:
            return self.climatology_map[doy].copy()

        # If DOY not found, find closest DOY in table
        if len(self.climatology_map) > 0:
            closest_doy = min(self.climatology_map.keys(), key=lambda k: abs(k - doy))
            return self.climatology_map[closest_doy].copy()

        # Fallback to depth-wise mean broadcast
        fallback = np.zeros((self.n_depths, self.n_lat, self.n_lon), dtype=np.float32)
        for k in range(self.n_depths):
            fallback[k, :, :] = self.overall_depth_mean[k]
        return fallback

    def compute_anomalies(self, date_str: str, target_field: np.ndarray) -> np.ndarray:
        """Computes temperature anomaly = target - climatology."""
        clim = self.predict_for_date(date_str)
        return target_field - clim


class RidgeSurfaceBaseline:
    """Ridge regression baseline mapping surface satellite observations to 15 depths."""

    def __init__(self, alpha: float = 1.0, n_depths: int = 15):
        self.alpha = alpha
        self.n_depths = n_depths
        self.models: List[Ridge] = [Ridge(alpha=self.alpha) for _ in range(n_depths)]
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def _extract_feature_matrix(
        self,
        surface_dict: Dict[str, np.ndarray],
        mask_ocean: np.ndarray,
        time_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts flattened (N_samples, N_features) matrix over ocean cells.

        Features: [SST, SSS, SLA, U_cur, V_cur, U_wind, V_wind]
        """
        var_names = ["sst", "sss", "sla", "u_cur", "v_cur", "u_wind", "v_wind"]
        n_times = surface_dict["sst"].shape[0]
        if time_indices is None:
            time_indices = list(range(n_times))

        features_list = []
        is_valid_mask = mask_ocean.copy()

        for t in time_indices:
            t_features = []
            for v in var_names:
                t_features.append(surface_dict[v][t][is_valid_mask])
            stacked_t = np.column_stack(t_features)  # (N_ocean, 7)
            features_list.append(stacked_t)

        X = np.vstack(features_list)  # (N_times * N_ocean, 7)
        return X, is_valid_mask

    def fit(
        self,
        surface_dict: Dict[str, np.ndarray],
        target_temperature: np.ndarray,
        mask_ocean: np.ndarray,
        mask_depth: np.ndarray,
        time_indices: Optional[List[int]] = None,
    ) -> None:
        """Fits separate Ridge regression models per depth level on training set."""
        X, is_ocean = self._extract_feature_matrix(surface_dict, mask_ocean, time_indices)

        # Standardize features ignoring NaNs
        self.feature_means = np.nanmean(X, axis=0)
        self.feature_stds = np.nanstd(X, axis=0)
        self.feature_stds[self.feature_stds < 1e-6] = 1.0

        X_norm = np.nan_to_num((X - self.feature_means) / self.feature_stds, nan=0.0)

        n_times = surface_dict["sst"].shape[0]
        if time_indices is None:
            time_indices = list(range(n_times))

        for k in range(self.n_depths):
            y_list = []
            for t in time_indices:
                y_list.append(target_temperature[t, k][is_ocean])
            y_k = np.concatenate(y_list)

            # Valid targets at this depth (excluding subterranean points)
            valid_y = ~np.isnan(y_k)
            if np.sum(valid_y) > 10:
                self.models[k].fit(X_norm[valid_y], y_k[valid_y])

    def predict(
        self,
        surface_dict: Dict[str, np.ndarray],
        mask_ocean: np.ndarray,
        mask_depth: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """Predicts full 3D temperature field (15, n_lat, n_lon) for a given day."""
        var_names = ["sst", "sss", "sla", "u_cur", "v_cur", "u_wind", "v_wind"]
        n_lat, n_lon = mask_ocean.shape
        t_features = [surface_dict[v][time_idx][mask_ocean] for v in var_names]
        X = np.column_stack(t_features)

        X_norm = np.nan_to_num((X - self.feature_means) / self.feature_stds, nan=0.0)

        out_pred = np.full((self.n_depths, n_lat, n_lon), np.nan, dtype=np.float32)

        for k in range(self.n_depths):
            pred_vals = self.models[k].predict(X_norm)
            plane = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            plane[mask_ocean] = pred_vals
            # Apply bathymetry mask
            valid_k = mask_depth[k] == 1
            plane[~valid_k] = np.nan
            out_pred[k] = plane

        return out_pred
