import React, { useRef, useEffect, useState } from 'react';
import { getThermalColor, getErrorColor } from '../utils/colormap';
import { MapPin, Columns } from 'lucide-react';

interface HeatmapViewProps {
  predictionGrid: (number | null)[][];
  truthGrid?: (number | null)[][];
  errorGrid?: (number | null)[][];
  rmse?: number;
  latitudes: number[];
  longitudes: number[];
  depth_m: number;
  modelName: string;
  pinnedCoord: { lat: number; lon: number };
  onPinCoord: (lat: number, lon: number) => void;
  viewMode: 'single' | 'compare';
  onToggleViewMode: () => void;
}

export const HeatmapView: React.FC<HeatmapViewProps> = ({
  predictionGrid,
  truthGrid,
  errorGrid,
  rmse,
  latitudes,
  longitudes,
  depth_m,
  modelName,
  pinnedCoord,
  onPinCoord,
  viewMode,
  onToggleViewMode,
}) => {
  const predCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const truthCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const errCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [hoverInfo, setHoverInfo] = useState<{
    lat: number;
    lon: number;
    temp: number | null;
    x: number;
    y: number;
  } | null>(null);

  const H = latitudes.length || 101;
  const W = longitudes.length || 241;

  const renderCanvas = (
    canvas: HTMLCanvasElement,
    grid: (number | null)[][],
    isError: boolean = false
  ) => {
    const ctx = canvas.getContext('2d');
    if (!ctx || !grid || grid.length === 0) return;

    canvas.width = W;
    canvas.height = H;
    const imgData = ctx.createImageData(W, H);
    const data = imgData.data;

    for (let i = 0; i < H; i++) {
      const gridRow = H - 1 - i;
      const row = grid[gridRow];

      for (let j = 0; j < W; j++) {
        const val = row ? row[j] : null;
        const color = isError
          ? getErrorColor(val !== null ? val : NaN)
          : getThermalColor(val !== null ? val : NaN, 4.0, 31.0);

        const idx = (i * W + j) * 4;
        data[idx] = color.r;
        data[idx + 1] = color.g;
        data[idx + 2] = color.b;
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    if (pinnedCoord) {
      const latIdx = Math.round(
        ((pinnedCoord.lat - latitudes[0]) / (latitudes[latitudes.length - 1] - latitudes[0])) * (H - 1)
      );
      const lonIdx = Math.round(
        ((pinnedCoord.lon - longitudes[0]) / (longitudes[longitudes.length - 1] - longitudes[0])) * (W - 1)
      );
      const canvasY = H - 1 - latIdx;
      const canvasX = lonIdx;

      ctx.strokeStyle = '#F3F3E7';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 4, 0, 2 * Math.PI);
      ctx.stroke();

      ctx.fillStyle = '#890149';
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  useEffect(() => {
    if (predCanvasRef.current && predictionGrid.length > 0) {
      renderCanvas(predCanvasRef.current, predictionGrid, false);
    }
    if (truthCanvasRef.current && truthGrid && truthGrid.length > 0) {
      renderCanvas(truthCanvasRef.current, truthGrid, false);
    }
    if (errCanvasRef.current && errorGrid && errorGrid.length > 0) {
      renderCanvas(errCanvasRef.current, errorGrid, true);
    }
  }, [predictionGrid, truthGrid, errorGrid, pinnedCoord, viewMode]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;

    const lonIdx = Math.min(W - 1, Math.max(0, Math.floor(relX * W)));
    const invLatIdx = Math.min(H - 1, Math.max(0, Math.floor(relY * H)));
    const latIdx = H - 1 - invLatIdx;

    const lat = latitudes[latIdx] || 5.0 + latIdx * 0.25;
    const lon = longitudes[lonIdx] || 45.0 + lonIdx * 0.25;

    const row = predictionGrid[latIdx];
    const temp = row ? row[lonIdx] : null;

    setHoverInfo({
      lat: Math.round(lat * 100) / 100,
      lon: Math.round(lon * 100) / 100,
      temp: temp !== null ? Math.round(temp * 100) / 100 : null,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;

    const lonIdx = Math.min(W - 1, Math.max(0, Math.floor(relX * W)));
    const invLatIdx = Math.min(H - 1, Math.max(0, Math.floor(relY * H)));
    const latIdx = H - 1 - invLatIdx;

    const lat = latitudes[latIdx] || 5.0 + latIdx * 0.25;
    const lon = longitudes[lonIdx] || 45.0 + lonIdx * 0.25;
    onPinCoord(lat, lon);
  };

  return (
    <div className="glass-panel" style={{ padding: '20px 24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Horizontal Temperature Field Slices (0.25° × 0.25°)
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff' }}>
            {viewMode === 'single' ? (
              <span>Model Reconstruction: <span style={{ color: 'var(--cyan-glow)' }}>{modelName}</span></span>
            ) : (
              <span>Comparative Analysis: <span style={{ color: 'var(--cyan-glow)' }}>Reconstruction</span> vs <span style={{ color: 'var(--coral-hot)' }}>GLORYS Truth</span></span>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {rmse !== undefined && (
            <div style={{
              background: 'rgba(6, 214, 160, 0.1)',
              border: '1px solid rgba(6, 214, 160, 0.3)',
              padding: '6px 12px',
              borderRadius: '8px',
              fontSize: '0.8rem',
              fontFamily: 'var(--font-mono)',
              color: 'var(--teal-glow)'
            }}>
              RMSE: <strong>{rmse.toFixed(3)}°C</strong>
            </div>
          )}

          <button
            onClick={onToggleViewMode}
            className={`btn-select ${viewMode === 'compare' ? 'active' : ''}`}
            title="Toggle Side-by-side Truth vs Prediction"
          >
            <Columns size={16} />
            <span>{viewMode === 'single' ? 'Compare Truth' : 'Single View'}</span>
          </button>
        </div>
      </div>

      {viewMode === 'single' ? (
        <div className="map-canvas-wrapper" style={{ position: 'relative' }}>
          <canvas
            ref={predCanvasRef}
            className="map-canvas"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            onClick={handleClick}
          />

          {hoverInfo && (
            <div style={{
              position: 'absolute',
              left: `${Math.min(hoverInfo.x + 15, 600)}px`,
              top: `${Math.max(hoverInfo.y - 45, 10)}px`,
              background: 'rgba(10, 20, 42, 0.95)',
              border: '1px solid var(--cyan-glow)',
              boxShadow: '0 4px 16px rgba(0,0,0,0.6)',
              borderRadius: '8px',
              padding: '6px 10px',
              pointerEvents: 'none',
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              zIndex: 10,
            }}>
              <div style={{ color: '#ffffff', fontWeight: 600 }}>
                {hoverInfo.lat.toFixed(2)}°N, {hoverInfo.lon.toFixed(2)}°E
              </div>
              <div style={{ color: hoverInfo.temp !== null ? 'var(--cyan-glow)' : 'var(--text-muted)' }}>
                {hoverInfo.temp !== null ? `T: ${hoverInfo.temp}°C at ${depth_m}m` : 'Land / Subterranean'}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--cyan-glow)', marginBottom: '6px', fontWeight: 600 }}>
              AI Prediction ({modelName})
            </div>
            <div className="map-canvas-wrapper">
              <canvas
                ref={predCanvasRef}
                className="map-canvas"
                onClick={handleClick}
              />
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--coral-hot)', marginBottom: '6px', fontWeight: 600 }}>
              GLORYS Reanalysis Ground Truth
            </div>
            <div className="map-canvas-wrapper">
              <canvas
                ref={truthCanvasRef}
                className="map-canvas"
                onClick={handleClick}
              />
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--amber-warn)', marginBottom: '6px', fontWeight: 600 }}>
              Difference Error (Pred − Truth, ±1.5°C)
            </div>
            <div className="map-canvas-wrapper">
              <canvas
                ref={errCanvasRef}
                className="map-canvas"
                onClick={handleClick}
              />
            </div>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>4°C (Deep/Cold)</span>
            <div style={{
              width: '120px',
              height: '10px',
              borderRadius: '4px',
              background: 'linear-gradient(90deg, #1d1145, #25567b, #00b4d8, #ffb703, #ff006e)',
              border: '1px solid rgba(255,255,255,0.1)'
            }} />
            <span>31°C (Warm Tropical)</span>
          </div>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          background: 'rgba(0, 240, 255, 0.08)',
          border: '1px solid rgba(0, 240, 255, 0.2)',
          padding: '4px 12px',
          borderRadius: '9999px',
          fontSize: '0.75rem',
          fontFamily: 'var(--font-mono)'
        }}>
          <MapPin size={14} color="var(--cyan-glow)" />
          <span>Active Profile Pin: <strong>{pinnedCoord.lat.toFixed(2)}°N, {pinnedCoord.lon.toFixed(2)}°E</strong></span>
          <span style={{ color: 'var(--text-muted)' }}>(Click anywhere on map to pin)</span>
        </div>
      </div>
    </div>
  );
};
