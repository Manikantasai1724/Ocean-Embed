"""Evaluation metrics for OceanEmbed subsurface reconstructions.

Implements:
- RMSE, Mean Bias, MAE, Pearson Correlation (R) per depth level.
- Skill Score (SS) versus Climatology per depth level:
  SS = 1 - (MSE_model / MSE_climatology).
- Regional sub-basin disaggregation:
  * Arabian Sea (AS): 5°N–26°N, 50°E–76°E
  * Bay of Bengal (BoB): 5°N–22°N, 80°E–98°E
  * Full North Indian Ocean Domain.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np


def compute_level_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
    clim_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes statistical metrics over valid masked ocean points.

    Args:
        y_pred: 2D or 3D predicted array
        y_true: 2D or 3D ground truth array
        mask: boolean array (True where comparison is valid)
        clim_pred: optional climatology predictions for Skill Score

    Returns:
        dict containing rmse, mae, bias, correlation, skill_score
    """
    valid = mask & (~np.isnan(y_pred)) & (~np.isnan(y_true))
    n_pts = int(np.sum(valid))

    if n_pts < 5:
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "corr": np.nan, "skill_score": np.nan, "samples": 0}

    p = y_pred[valid]
    t = y_true[valid]
    diff = p - t

    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    std_p = np.std(p)
    std_t = np.std(t)
    if std_p > 1e-5 and std_t > 1e-5:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = 1.0 if std_p == std_t else 0.0

    skill_score = np.nan
    if clim_pred is not None:
        c = clim_pred[valid]
        mse_clim = float(np.mean((c - t) ** 2))
        if mse_clim > 1e-6:
            skill_score = 1.0 - (mse / mse_clim)
        else:
            skill_score = 0.0

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "bias": round(bias, 4),
        "corr": round(corr, 4),
        "skill_score": round(skill_score, 4) if not np.isnan(skill_score) else np.nan,
        "samples": n_pts,
    }


def evaluate_model_per_depth_and_region(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    climatology: np.ndarray,
    mask_depth: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: List[float],
) -> Dict[str, Any]:
    """Computes comprehensive metrics per depth level across Full Domain, Arabian Sea, and Bay of Bengal.

    Args:
        predictions: (T, 15, H, W)
        ground_truth: (T, 15, H, W)
        climatology: (T, 15, H, W)
        mask_depth: (15, H, W)
        lats: (H,)
        lons: (W,)
        depths: (15,)
    """
    n_times, n_depths, n_lat, n_lon = predictions.shape
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Sub-basin masks
    as_mask = (lat_grid >= 5.0) & (lat_grid <= 26.0) & (lon_grid >= 50.0) & (lon_grid <= 76.0)
    bob_mask = (lat_grid >= 5.0) & (lat_grid <= 22.0) & (lon_grid >= 80.0) & (lon_grid <= 98.0)
    full_mask = np.ones((n_lat, n_lon), dtype=bool)

    regions = {
        "full_domain": full_mask,
        "arabian_sea": as_mask,
        "bay_of_bengal": bob_mask,
    }

    results: Dict[str, Any] = {"regions": {}}

    for reg_name, reg_spatial_mask in regions.items():
        reg_depth_stats = []

        for k, d in enumerate(depths):
            # 3D boolean mask for this depth level over all T days
            k_mask = (mask_depth[k] == 1) & reg_spatial_mask
            k_mask_3d = np.broadcast_to(k_mask[None, :, :], (n_times, n_lat, n_lon))

            p_k = predictions[:, k, :, :]
            t_k = ground_truth[:, k, :, :]
            c_k = climatology[:, k, :, :]

            metrics = compute_level_metrics(p_k, t_k, k_mask_3d, clim_pred=c_k)
            metrics["depth_m"] = float(d)
            reg_depth_stats.append(metrics)

        results["regions"][reg_name] = reg_depth_stats

    return results
