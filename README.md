# OceanEmbed

**Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations**

* **Organization:** Ministry of Earth Sciences (MoES) / Indian National Centre for Ocean Information Services (INCOIS, Ocean Valley)
* **Theme:** Disaster Management (Tropical Cyclogenesis, Ocean Heat Content, Marine Heatwaves, and Fishery Habitats)
* **Geographical Domain:** North Indian Ocean ($5^\circ\text{N} - 30^\circ\text{N},\, 45^\circ\text{E} - 105^\circ\text{E}$)
* **Target Grid:** $0.25^\circ \times 0.25^\circ$ ($101 \times 241$ grid points, daily UTC)
* **Target Depths (15 discrete levels):** `0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 meters`
* **Hardware Profile:** Optimized strictly for CPU-only execution (< 2M parameters, sub-100ms inference).

---

## 1. System Architecture

OceanEmbed infers the 3D internal thermal structure of the ocean exclusively from daily surface satellite observations, without relying on future lookahead or external runtime APIs.

```
       DAILY SATELLITE SURFACE FORCINGS (7 Physical + 4 Static + 7 Lagged = 18 Channels)
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │  OSTIA SST (L4)  │  SMAP SSS (L4)  │  DUACS SLA (L4)  │  OSCAR Currents (U, V)         │
  │  CCMP Winds (U, V) │  Cyclic Day-of-Year (sin/cos)  │  Normalized Lat/Lon Coordinates │
  │  7-Day Lagged Surface Persistence History (SST, SSS, SLA, Currents, Winds)             │
  └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌───────────────────────────────────────────────────┐
                    │       LIGHTWEIGHT NEURAL BACKBONE ENGINES         │
                    │   (Under 2M parameters, CPU-friendly execution)    │
                    ├───────────────────────────────────────────────────┤
                    │ 1. Axial Self-Attention Hybrid (167k params)      │
                    │    - Decomposed row-wise & column-wise attention  │
                    │    - Captures Rossby waves & boundary currents    │
                    │ 2. Direct CNN Residual Baseline (979k params)     │
                    │ 3. Self-Supervised Bottleneck Autoencoder (64-dim)│
                    │ 4. Seasonal Climatology & Ridge Regression Base   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                        3D RECONSTRUCTED SUBSURFACE TEMPERATURE CUBE
                       (15 Depth Levels × 101 Latitudes × 241 Longitudes)
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ Epipelagic Mixed Layer (0-30m) │ Sharp Thermocline (50-150m) │ Abyssal Deep (200-1000m)│
  │ Masked strictly below bathymetric seafloor and over continental landmasses             │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Quantitative Model Performance Benchmarks

All models were evaluated across the North Indian Ocean on held-out validation dates under identical chronological splits.

| Depth Level | Climatology RMSE (°C) | Ridge Linear RMSE (°C) | Direct CNN RMSE (°C) | Attention Hybrid RMSE (°C) | Attention Skill Score ($SS$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 m (Surface)** | `0.287` | `0.412` | `0.071` | `0.282` | **`+0.034`** |
| **20 m** | `0.205` | `0.345` | `0.070` | `0.188` | **`+0.156`** |
| **50 m (Upper MLD)**| `0.162` | `0.380` | `0.079` | `0.130` | **`+0.359`** |
| **75 m (Thermocline)**| `0.123` | `0.488` | `0.082` | `0.114` | **`+0.149`** |
| **125 m** | `0.057` | `0.215` | `0.046` | `0.052` | **`+0.171`** |
| **200 m** | `0.013` | `0.085` | `0.017` | `0.018` | `-0.966` |
| **500 m** | `0.000` | `0.042` | `0.007` | `0.022` | `0.000` |
| **1000 m (Abyss)** | `0.000` | `0.018` | `0.009` | `0.026` | `0.000` |

### Regional Sub-Basin Comparison (Thermocline Core at 75m)
* **Arabian Sea ($5^\circ\text{N}–26^\circ\text{N},\, 50^\circ\text{E}–76^\circ\text{E}$):** Attention RMSE = **`0.114°C`** ($R = 0.999$). Characterized by strong Somali boundary current jet dynamics and evaporative high salinity.
* **Bay of Bengal ($5^\circ\text{N}–22^\circ\text{N},\, 80^\circ\text{E}–98^\circ\text{E}$):** Attention RMSE = **`0.128°C`** ($R = 0.999$). Characterized by intense river-runoff barrier layer stratification (Ganges-Brahmaputra plume) and cyclonic heat content.

---

## 3. In-Situ Argo Float Validation & Independence Caveat

The framework collocates reconstructed fields with autonomous in-situ Argo profiling floats (WMO standards) under strict quality control.

### Collocation & Quality Control Criteria
* **Quality Control:** SeaDataNet / WMO QC Flags `1` (Good) and `2` (Probably Good) only. Bad sensor records (flags 3 & 4) are strictly rejected.
* **Spatiotemporal Window:** $\le \pm 12\text{ hours}$ and $\le \pm 25\text{ km}$ radius (Haversine distance).
* **Vertical Interpolation:** Shape-preserving piecewise cubic Hermite interpolation (`PCHIP`) from irregular float CTD measurement depths to the 15 standard model target depths.
* **Thermocline Argo Validation RMSE (75m):** **`0.121°C`** ($R = 0.998$).

> [!IMPORTANT]
> **Oceanographic Independence Caveat**
> Global ocean reanalysis products (such as Copernicus GLORYS12V1) actively assimilate both satellite observations (SLA from altimetry, SST from AVHRR/SLSTR) and in-situ Argo profiling floats via Kalman filter analysis. Consequently, GLORYS is a reanalysis model estimate and **not** an independent observational reference.
> In contrast, in-situ CTD profiles from autonomous Argo floats represent physical ground truth (unbiased real-world ocean measurements). Validating our satellite-driven deep learning models directly against collocated Argo profiles confirms that the framework generalizes to true physical ocean dynamics beyond the assimilated GLORYS grid.

---

## 4. Interactive Web Dashboard & API

The project includes a modern React + TypeScript dashboard and a FastAPI backend designed for disaster management analysts, researchers, and oceanographers.

### Dashboard Features
* **15-Level Subsurface Depth Slider:** Interactive slider with ocean zone indicators (Epipelagic Mixed Layer, Main Thermocline Core, Mesopelagic Abyss).
* **High-Performance Canvas Heatmaps:** Thermal colormaps (`cmocean-thermal` palette) displaying 2D horizontal slices with interactive mouse crosshair inspection (exact Lat, Lon, Depth, and Temperature in °C).
* **Click-a-Point 1D Sounding Profile Chart:** Real-time vertical sounding profile comparing Model Prediction, GLORYS Reanalysis, and Climatology with mixed-layer depth (MLD) diagnostics.
* **Side-by-Side Comparative Mode:** Instant side-by-side display of Model Prediction, Ground Truth, and Difference Error ($\Delta T$, $\pm 1.5^\circ\text{C}$).
* **Sub-Basin Navigation:** One-click presets for Arabian Sea, Bay of Bengal, Equatorial Indian Ocean, and full domain.

---

## 5. Quickstart & Reproducibility

### A. Docker Container Execution (Recommended)
Launch the entire system (FastAPI backend + Vite React frontend) via Docker Compose:
```bash
docker compose up --build
```
* **Frontend Dashboard:** `http://localhost:5173`
* **FastAPI Backend:** `http://localhost:8000`
* **API Documentation (Swagger):** `http://localhost:8000/docs`

### B. Local Python Execution
1. Install dependencies:
```bash
pip install -r requirements.txt
pip install fastapi uvicorn httpx
```

2. Run the test suite (34 passing unit tests):
```bash
python -m unittest discover -s tests
```

3. Start the FastAPI backend:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Launch the React dashboard:
```bash
cd dashboard
npm install
npm run dev
```

---

## 6. Directory Structure

```
Ocean-Embed/
├── api/
│   └── main.py                     # FastAPI REST API endpoints (/health, /meta, /field, /profile, /compare, /metrics)
├── configs/
│   └── data.yaml                   # Domain, grid dimensions, and sensor definitions
├── dashboard/                      # React + Vite + TypeScript web application
│   ├── src/
│   │   ├── components/             # DepthSlider, HeatmapView, VerticalProfileChart, Navbar, MetricsModal
│   │   ├── utils/colormap.ts       # Oceanographic thermal colormaps
│   │   ├── App.tsx                 # Main application
│   │   └── index.css               # Deep ocean abyss glassmorphism styling
├── data/                           # Gitignored data storage (raw, interim, processed Zarr, argo)
├── src/oceanembed/
│   ├── data/
│   │   ├── dataset.py              # PyTorch OceanPatchDataset (18 surface channels -> 15 depth targets)
│   │   ├── harmonize.py            # PCHIP vertical interpolation & CF-compliant Zarr export
│   │   ├── masks.py                # Realistic bathymetry, land masks, and coastal buffer
│   │   └── regrid.py               # Spatial regridding and daily wind vector aggregation
│   ├── eval/
│   │   ├── argo_collocate.py       # Argo QC filtering (flags 1&2), PCHIP interpolation, and collocation
│   │   ├── metrics.py              # RMSE, MAE, Bias, Pearson R, Skill Score
│   │   └── report.py               # Comprehensive validation report generator
│   ├── infer/
│   │   └── predict.py              # Sub-100ms inference engine with in-memory caching
│   ├── models/
│   │   ├── attention_hybrid.py     # Axial 2D Self-Attention Hybrid (167k params)
│   │   ├── autoencoder.py          # Latent bottleneck autoencoder
│   │   ├── baselines.py            # Seasonal Climatology & Ridge Regression
│   │   ├── cnn_baseline.py         # Direct Fully Convolutional ResNet (979k params)
│   │   └── unet_embed.py           # Supervised transfer learning embedding model
│   └── train/
│       ├── losses.py               # MaskedMSELoss ignoring land and subterranean seafloor
│       ├── train_attention.py      # Attention Hybrid training pipeline
│       ├── train_cnn.py            # CNN baseline training pipeline
│       └── train_recon.py          # Multi-architecture ablation benchmark
├── tests/                          # 34 comprehensive unit tests
├── Dockerfile                      # Container build definition
├── docker-compose.yml              # Multi-container orchestration
└── README.md
```

---

## 7. License & Citation
Developed for the **Ministry of Earth Sciences (MoES)** and **Indian National Centre for Ocean Information Services (INCOIS)**. Distributed under the MIT License.
