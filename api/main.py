"""FastAPI Backend for OceanEmbed.

Provides endpoints for interactive frontend exploration:
- /health: API and model status
- /meta: Coordinate grids, depths, dates, available models
- /field: 2D horizontal temperature slices
- /profile: Click-a-point vertical temperature profiles (0-1000m)
- /compare: Side-by-side Truth vs Prediction vs Error maps
- /metrics: Benchmarks and in-situ Argo validation statistics
"""

import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from oceanembed.infer.predict import OceanPredictor

app = FastAPI(
    title="OceanEmbed API",
    description="3D Subsurface Ocean Temperature Reconstruction from Daily Satellite Observations",
    version="1.0.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor singleton loaded at startup
predictor: Optional[OceanPredictor] = None


@app.on_event("startup")
def startup_event():
    global predictor
    zarr_path = os.getenv("OCEANEMBED_ZARR", "data/processed/oceanembed_slice_201801.zarr")
    chkpt_dir = os.getenv("OCEANEMBED_CHECKPOINTS", "checkpoints")
    print(f"[OceanEmbed API] Initializing OceanPredictor with {zarr_path}...")
    predictor = OceanPredictor(zarr_path=zarr_path, checkpoint_dir=chkpt_dir)
    print("[OceanEmbed API] Ready for inference requests.")


@app.get("/health")
def health_check():
    """Returns system status, active device, and loaded models."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor initializing")
    return {
        "status": "healthy",
        "service": "OceanEmbed Subsurface Reconstruction API",
        "device": "CPU",
        "models": list(predictor.models.keys()) + ["climatology"],
        "num_depth_levels": len(predictor.depths),
        "spatial_grid": f"{len(predictor.lats)}x{len(predictor.lons)}",
    }


@app.get("/meta")
def get_metadata():
    """Returns grid coordinates, depths, dates, and domain boundaries."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor initializing")
    return predictor.get_metadata()


@app.get("/field")
def get_field(
    date: str = Query("2018-01-25", description="Date string (YYYY-MM-DD)"),
    depth: float = Query(50.0, description="Target depth in meters (0 to 1000)"),
    model: str = Query("attention_hybrid", description="Model name: attention_hybrid, direct_cnn, or climatology"),
):
    """Retrieves 2D horizontal temperature field slice for map display."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor initializing")
    try:
        return predictor.get_field_slice(date=date, depth_m=depth, model_name=model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/profile")
def get_profile(
    date: str = Query("2018-01-25", description="Date string (YYYY-MM-DD)"),
    lat: float = Query(15.0, description="Latitude in degrees (5.0 to 30.0)"),
    lon: float = Query(65.0, description="Longitude in degrees (45.0 to 105.0)"),
    model: str = Query("attention_hybrid", description="Model name: attention_hybrid, direct_cnn, or climatology"),
):
    """Extracts 1D vertical temperature profile at specified coordinate."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor initializing")
    try:
        return predictor.get_profile(date=date, lat=lat, lon=lon, model_name=model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/compare")
def compare_field(
    date: str = Query("2018-01-25", description="Date string (YYYY-MM-DD)"),
    depth: float = Query(50.0, description="Target depth in meters (0 to 1000)"),
    model: str = Query("attention_hybrid", description="Model name: attention_hybrid, direct_cnn, or climatology"),
):
    """Retrieves side-by-side Truth, Prediction, and Error grids with RMSE."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor initializing")
    try:
        return predictor.compare_slice(date=date, depth_m=depth, model_name=model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


import math


def sanitize_floats(obj):
    """Recursively replaces NaN and Inf floats with None for strict JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj


@app.get("/metrics")
def get_metrics(
    model: str = Query("attention_hybrid", description="Model name"),
):
    """Returns precomputed evaluation metrics and Argo validation statistics."""
    # Check for attention hybrid or cnn results
    results_map = {
        "attention_hybrid": "data/attention_hybrid_results.json",
        "direct_cnn": "data/cnn_baseline_results.json",
    }
    argo_path = "data/argo_validation_results.json"

    model_metrics = None
    target_file = results_map.get(model, "data/attention_hybrid_results.json")
    if os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8") as f:
            model_metrics = json.load(f)

    argo_metrics = None
    if os.path.exists(argo_path):
        with open(argo_path, "r", encoding="utf-8") as f:
            argo_metrics = json.load(f)

    return sanitize_floats({
        "model_name": model,
        "evaluation_metrics": model_metrics,
        "in_situ_argo_validation": argo_metrics,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
