# OceanEmbed

**Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations**

* **Organization:** MoES / INCOIS (Ocean Valley)
* **Category:** Software
* **Theme:** Disaster Management
* **Domain:** North Indian Ocean ($5^\circ\text{N} - 30^\circ\text{N}, 45^\circ\text{E} - 105^\circ\text{E}$)
* **Target Grid:** $0.25^\circ \times 0.25^\circ$ ($101 \times 241$ grid points, daily UTC)
* **Target Depths (15 levels in meters):** `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]`

---

## 1. Project Overview
OceanEmbed reconstructs 3D ocean temperature fields in the North Indian Ocean using strictly daily surface observations:
- **Sea Surface Temperature (SST)**: OSTIA L4
- **Sea Surface Salinity (SSS)**: SMAP/SMOS L4
- **Sea Level Anomaly (SLA / SSH)**: DUACS L4
- **Ocean Surface Currents (U, V)**: OSCAR L4 V2.0
- **Surface Vector Winds (U, V)**: CCMP L4 V3.1 (or ASCAT Coastal)
- **Subsurface Ground Truth Target**: GLORYS12V1 Ocean Physics Reanalysis
- **Validation**: INCOIS Gridded ARGO / GDAC ARGO

---

## 2. Directory Structure
```
oceanembed/
  configs/         # YAML runtime and dataset configurations
  data/            # gitignored local data storage (raw, interim, processed, argo)
  src/oceanembed/  # Core Python package
    data/          # Download, regridding, harmonization, mask generation
    models/        # Baselines, direct CNN, autoencoder, attention hybrids
    train/         # Training pipelines, loss functions, early stopping
    eval/          # Evaluation metrics, ARGO collocation, report generator
    infer/         # Inference scripts, gridded export routines
  api/             # FastAPI backend server
  dashboard/       # React + Vite + TypeScript interactive web dashboard
  notebooks/       # Jupyter exploratory notebooks
  tests/           # Pytest / Unittest validation suite
```

---

## 3. Verified Candidate Datasets Table
See the Phase 0 verification documentation for detailed coverage, access protocols, native coordinates, and vertical level mapping.
