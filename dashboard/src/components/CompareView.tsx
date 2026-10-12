/**
 * CompareView.tsx — Side-by-side canvas comparison of Prediction vs Truth vs Error.
 *
 * Used when the user toggles "Compare Truth" mode.
 * Retains the fast canvas rendering for 101×241 grids.
 */
import React, { useRef, useEffect } from 'react';
import { getThermalColor, getErrorColor } from '../utils/colormap';

interface CompareViewProps {
  predictionGrid: (number | null)[][];
  truthGrid: (number | null)[][];
  errorGrid: (number | null)[][];
  rmse: number;
  latitudes: number[];
  longitudes: number[];
  depth_m: number;
  modelName: string;
  pinnedCoord: { lat: number; lon: number };
  onPinCoord: (lat: number, lon: number) => void;
}

export const CompareView: React.FC<CompareViewProps> = ({
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
}) => {
  const predRef = useRef<HTMLCanvasElement>(null);
  const truthRef = useRef<HTMLCanvasElement>(null);
  const errRef = useRef<HTMLCanvasElement>(null);

  const H = latitudes.length || 101;
  const W = longitudes.length || 241;

  const renderCanvas = (canvas: HTMLCanvasElement, grid: (number | null)[][], isError = false) => {
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

    // Draw pinned marker
    if (pinnedCoord && latitudes.length > 0 && longitudes.length > 0) {
      const latIdx = Math.round(((pinnedCoord.lat - latitudes[0]) / (latitudes[latitudes.length - 1] - latitudes[0])) * (H - 1));
      const lonIdx = Math.round(((pinnedCoord.lon - longitudes[0]) / (longitudes[longitudes.length - 1] - longitudes[0])) * (W - 1));
      const cy = H - 1 - latIdx;
      const cx = lonIdx;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = '#00f0ff';
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  useEffect(() => {
    if (predRef.current && predictionGrid.length > 0) renderCanvas(predRef.current, predictionGrid);
    if (truthRef.current && truthGrid.length > 0) renderCanvas(truthRef.current, truthGrid);
    if (errRef.current && errorGrid.length > 0) renderCanvas(errRef.current, errorGrid, true);
  }, [predictionGrid, truthGrid, errorGrid, pinnedCoord]);

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

  const panels = [
    { label: 'AI Prediction', sub: modelName, ref: predRef, color: '#00f0ff' },
    { label: 'GLORYS Truth', sub: 'Reanalysis Reference', ref: truthRef, color: '#ff006e' },
    { label: 'Error (Pred − Truth)', sub: `RMSE: ${rmse.toFixed(3)}°C`, ref: errRef, color: '#ffb703' },
  ];

  return (
    <div className="glass-panel fade-in" style={{ padding: '16px 18px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Comparative Analysis
          </div>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>
            <span style={{ color: 'var(--cyan-glow)' }}>Reconstruction</span> vs <span style={{ color: 'var(--coral-hot)' }}>GLORYS</span> at {depth_m}m
          </div>
        </div>
        <div className="badge badge-teal">
          RMSE {rmse.toFixed(3)}°C
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
        {panels.map(p => (
          <div key={p.label}>
            <div style={{ marginBottom: '6px' }}>
              <span style={{ fontSize: '0.7rem', fontWeight: 600, color: p.color }}>{p.label}</span>
              <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginLeft: '6px' }}>{p.sub}</span>
            </div>
            <div className="map-canvas-wrapper">
              <canvas
                ref={p.ref}
                className="map-canvas"
                onClick={handleClick}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Color legend */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px', gap: '12px', flexWrap: 'wrap' }}>
        <div className="colorbar" style={{ flex: 1 }}>
          <span>4°C</span>
          <div className="colorbar-gradient" />
          <span>31°C</span>
        </div>
        <div className="colorbar" style={{ flex: 1 }}>
          <span style={{ color: '#4488ff' }}>-1.5°C</span>
          <div className="colorbar-gradient" style={{
            background: 'linear-gradient(90deg, #1e44aa, #c8c8ff, #ffffff, #ffc8c8, #ff2222)',
          }} />
          <span style={{ color: '#ff4444' }}>+1.5°C</span>
        </div>
      </div>
    </div>
  );
};
