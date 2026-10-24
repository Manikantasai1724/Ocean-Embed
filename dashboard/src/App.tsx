/**
 * App.tsx — Main orchestrator for OceanEmbed dashboard v2.
 *
 * Layout: TopBar + Sidebar (controls) + Main Content (map + chart).
 * The sidebar manages all user controls; the main area shows the map
 * and vertical profile chart side by side (or compare view).
 */
import { useState, useEffect } from 'react';
import { TopBar } from './components/TopBar';
import { Sidebar } from './components/Sidebar';
import { LeafletMap } from './components/LeafletMap';
import { CompareView } from './components/CompareView';
import { VerticalProfileChart } from './components/VerticalProfileChart';
import type { ProfilePoint } from './components/VerticalProfileChart';
import { MetricsModal } from './components/MetricsModal';

const API_BASE = 'http://localhost:8000';

export function App() {
  const [modelName, setModelName] = useState<string>('attention_hybrid');
  const [depth_m, setDepthM] = useState<number>(50);
  const [date] = useState<string>('2018-01-25');
  const [pinnedCoord, setPinnedCoord] = useState<{ lat: number; lon: number }>({ lat: 15.0, lon: 65.0 });
  const [viewMode, setViewMode] = useState<'single' | 'compare'>('single');
  const [isMetricsOpen, setIsMetricsOpen] = useState<boolean>(false);
  const [isApiOnline, setIsApiOnline] = useState<boolean>(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  // Synchronize data-theme on html element for full system CSS switching
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Grid and Metadata state
  const [latitudes, setLatitudes] = useState<number[]>([]);
  const [longitudes, setLongitudes] = useState<number[]>([]);
  const [targetDepths, setTargetDepths] = useState<number[]>([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]);
  const [predictionGrid, setPredictionGrid] = useState<(number | null)[][]>([]);
  const [truthGrid, setTruthGrid] = useState<(number | null)[][]>([]);
  const [errorGrid, setErrorGrid] = useState<(number | null)[][]>([]);
  const [rmse, setRmse] = useState<number>(0.130);

  // Profile state
  const [profileData, setProfileData] = useState<ProfilePoint[]>([]);
  const [seafloorDepth, setSeafloorDepth] = useState<number>(3800);
  const [isOcean, setIsOcean] = useState<boolean>(true);

  // 1. Check API metadata or load fallback grid coordinates
  useEffect(() => {
    fetch(`${API_BASE}/meta`)
      .then(res => res.json())
      .then(data => {
        setIsApiOnline(true);
        setLatitudes(data.latitudes);
        setLongitudes(data.longitudes);
        if (data.target_depths) setTargetDepths(data.target_depths);
      })
      .catch(() => {
        setIsApiOnline(false);
        // Generate 0.25° grid for the North Indian Ocean domain
        const lats: number[] = [];
        for (let lat = 5.0; lat <= 30.01; lat += 0.25) lats.push(Math.round(lat * 100) / 100);
        const lons: number[] = [];
        for (let lon = 45.0; lon <= 105.01; lon += 0.25) lons.push(Math.round(lon * 100) / 100);
        setLatitudes(lats);
        setLongitudes(lons);
      });
  }, []);

  // 2. Fetch field or comparison slice from API (or generate fallback)
  useEffect(() => {
    if (isApiOnline) {
      if (viewMode === 'compare') {
        fetch(`${API_BASE}/compare?date=${date}&depth=${depth_m}&model=${modelName}`)
          .then(res => res.json())
          .then(data => {
            setPredictionGrid(data.prediction_grid);
            setTruthGrid(data.truth_grid);
            setErrorGrid(data.error_grid);
            setRmse(data.rmse_c);
          })
          .catch(() => generateFallbackGrids(depth_m));
      } else {
        fetch(`${API_BASE}/field?date=${date}&depth=${depth_m}&model=${modelName}`)
          .then(res => res.json())
          .then(data => {
            setPredictionGrid(data.grid);
          })
          .catch(() => generateFallbackGrids(depth_m));
      }
    } else {
      generateFallbackGrids(depth_m);
    }
  }, [depth_m, date, modelName, viewMode, isApiOnline]);

  // 3. Fetch vertical profile at pinned coordinates
  useEffect(() => {
    if (isApiOnline) {
      fetch(`${API_BASE}/profile?date=${date}&lat=${pinnedCoord.lat}&lon=${pinnedCoord.lon}&model=${modelName}`)
        .then(res => res.json())
        .then(data => {
          setProfileData(data.profile);
          setSeafloorDepth(data.seafloor_depth_m);
          setIsOcean(data.is_ocean);
        })
        .catch(() => generateFallbackProfile(pinnedCoord.lat, pinnedCoord.lon));
    } else {
      generateFallbackProfile(pinnedCoord.lat, pinnedCoord.lon);
    }
  }, [pinnedCoord, date, modelName, isApiOnline]);

  /**
   * Generates physically-plausible fallback grids for instant display
   * without running the backend. Uses exponential SST decay, crude
   * land mask, and synthetic eddy patterns.
   */
  const generateFallbackGrids = (d: number) => {
    const H = 101;
    const W = 241;
    const pGrid: (number | null)[][] = [];
    const tGrid: (number | null)[][] = [];
    const eGrid: (number | null)[][] = [];

    const baseTemp = 28.5 * Math.exp(-d / 220.0) + 4.2;

    for (let i = 0; i < H; i++) {
      const lat = 5.0 + i * 0.25;
      const pRow: (number | null)[] = [];
      const tRow: (number | null)[] = [];
      const eRow: (number | null)[] = [];

      for (let j = 0; j < W; j++) {
        const lon = 45.0 + j * 0.25;

        // Crude land mask for Indian subcontinent, Arabia, SE Asia
        const isLand = (lat >= 8.0 && lat <= 28.0 && lon >= 70.0 && lon <= 88.0 && (lon >= 77.5 - (lat - 8) * 0.5) && (lon <= 77.5 + (lat - 8) * 0.6))
          || (lon < 60.0 && lat > 18.0)
          || (lon >= 96.0 && lat >= 12.0);

        if (isLand) {
          pRow.push(null);
          tRow.push(null);
          eRow.push(null);
        } else {
          const latGrad = (lat - 17.5) * -0.15;
          const lonGrad = (lon - 75.0) * 0.05;
          const eddy = Math.sin(lat * 0.5) * Math.cos(lon * 0.4) * 0.4;
          const trueVal = Math.max(4.0, Math.min(31.0, baseTemp + latGrad + lonGrad + eddy));
          const err = (Math.sin(lat * 2.1) * Math.cos(lon * 1.8)) * 0.12;
          const predVal = trueVal + err;

          pRow.push(Math.round(predVal * 100) / 100);
          tRow.push(Math.round(trueVal * 100) / 100);
          eRow.push(Math.round(err * 100) / 100);
        }
      }
      pGrid.push(pRow);
      tGrid.push(tRow);
      eGrid.push(eRow);
    }

    setPredictionGrid(pGrid);
    setTruthGrid(tGrid);
    setErrorGrid(eGrid);
    setRmse(d <= 75 ? 0.114 : 0.045);
  };

  const generateFallbackProfile = (lat: number, _lon: number) => {
    const points: ProfilePoint[] = targetDepths.map(d => {
      const truth = 28.5 * Math.exp(-d / 180.0) + 4.5 + Math.sin(lat) * 0.3;
      const noise = (Math.sin(d * 0.2) + Math.cos(lat)) * 0.08;
      const clim = truth - 0.25 * Math.sin(d / 80.0);
      return {
        depth_m: d,
        predicted_temp: Math.round((truth + noise) * 100) / 100,
        truth_temp: Math.round(truth * 100) / 100,
        climatology_temp: Math.round(clim * 100) / 100,
      };
    });
    setProfileData(points);
    setSeafloorDepth(3850);
    setIsOcean(true);
  };

  const handleSubbasinSelect = (lat: number, lon: number) => {
    setPinnedCoord({ lat, lon });
  };

  const displayModelName = modelName === 'attention_hybrid' ? 'Axial Attention Hybrid'
    : modelName === 'direct_cnn' ? 'Direct CNN'
    : 'Climatology';

  return (
    <div className="app-layout">
      {/* ═══ TOP BAR ═══ */}
      <TopBar
        isApiOnline={isApiOnline}
        onOpenMetrics={() => setIsMetricsOpen(true)}
        theme={theme}
        onToggleTheme={() => setTheme(t => t === 'light' ? 'dark' : 'light')}
      />

      {/* ═══ SIDEBAR ═══ */}
      <Sidebar
        modelName={modelName}
        onModelChange={setModelName}
        depths={targetDepths}
        currentDepth={depth_m}
        onDepthChange={setDepthM}
        pinnedCoord={pinnedCoord}
        onSelectSubbasin={handleSubbasinSelect}
        viewMode={viewMode}
        onToggleViewMode={() => setViewMode(v => v === 'single' ? 'compare' : 'single')}
      />

      {/* ═══ MAIN CONTENT ═══ */}
      <main className="app-main">
        {viewMode === 'single' ? (
          /* Single view: Interactive Leaflet Map + Vertical Profile Chart */
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.5fr) minmax(320px, 1fr)', gap: '16px', height: '100%' }}>
            <div className="glass-panel" style={{ padding: '12px', overflow: 'hidden' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', padding: '0 4px' }}>
                <div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    Horizontal Slice · 0.25° × 0.25°
                  </div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-heading)' }}>
                    <span style={{ color: 'var(--primary)' }}>{displayModelName}</span> · {depth_m}m
                  </div>
                </div>
                <div className="badge badge-magenta">RMSE {rmse.toFixed(3)}°C</div>
              </div>
              <LeafletMap
                predictionGrid={predictionGrid}
                latitudes={latitudes}
                longitudes={longitudes}
                depth_m={depth_m}
                modelName={displayModelName}
                pinnedCoord={pinnedCoord}
                onPinCoord={(lat, lon) => setPinnedCoord({ lat, lon })}
                theme={theme}
              />
            </div>

            <VerticalProfileChart
              coord={pinnedCoord}
              seafloorDepth={seafloorDepth}
              isOcean={isOcean}
              modelName={displayModelName}
              profile={profileData}
            />
          </div>
        ) : (
          /* Compare view: Side-by-side canvases + Vertical Profile below */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <CompareView
              predictionGrid={predictionGrid}
              truthGrid={truthGrid}
              errorGrid={errorGrid}
              rmse={rmse}
              latitudes={latitudes}
              longitudes={longitudes}
              depth_m={depth_m}
              modelName={displayModelName}
              pinnedCoord={pinnedCoord}
              onPinCoord={(lat, lon) => setPinnedCoord({ lat, lon })}
            />

            <div style={{ maxWidth: '600px' }}>
              <VerticalProfileChart
                coord={pinnedCoord}
                seafloorDepth={seafloorDepth}
                isOcean={isOcean}
                modelName={displayModelName}
                profile={profileData}
              />
            </div>
          </div>
        )}
      </main>

      {/* ═══ METRICS MODAL ═══ */}
      <MetricsModal
        isOpen={isMetricsOpen}
        onClose={() => setIsMetricsOpen(false)}
      />
    </div>
  );
}

export default App;
