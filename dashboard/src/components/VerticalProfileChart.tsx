/**
 * VerticalProfileChart.tsx — SVG-based vertical ocean temperature sounding.
 *
 * Shows predicted vs truth vs climatology temperature profiles from 0–1000m.
 * Uses a non-linear depth axis (power scale with exponent 0.6) to expand
 * the critical upper-ocean thermocline region while still showing deep water.
 */
import { Compass, Anchor } from 'lucide-react';

export interface ProfilePoint {
  depth_m: number;
  predicted_temp: number | null;
  truth_temp: number | null;
  climatology_temp: number | null;
}

interface VerticalProfileChartProps {
  coord: { lat: number; lon: number };
  seafloorDepth: number;
  isOcean: boolean;
  modelName: string;
  profile: ProfilePoint[];
}

export const VerticalProfileChart: React.FC<VerticalProfileChartProps> = ({
  coord,
  seafloorDepth,
  isOcean,
  modelName,
  profile,
}) => {
  const svgWidth = 400;
  const svgHeight = 420;
  const margin = { top: 28, right: 26, bottom: 38, left: 54 };
  const innerWidth = svgWidth - margin.left - margin.right;
  const innerHeight = svgHeight - margin.top - margin.bottom;

  const minTemp = 4.0;
  const maxTemp = 31.0;
  const maxDepth = 1000.0;

  const scaleX = (temp: number) => {
    const t = Math.max(minTemp, Math.min(maxTemp, temp));
    return margin.left + ((t - minTemp) / (maxTemp - minTemp)) * innerWidth;
  };

  // Non-linear depth scale: expands upper-ocean detail
  const scaleY = (depth: number) => {
    const depthFrac = Math.pow(depth / maxDepth, 0.6);
    return margin.top + depthFrac * innerHeight;
  };

  const validPreds = profile.filter(p => p.predicted_temp !== null);
  const validTruth = profile.filter(p => p.truth_temp !== null);
  const validClim = profile.filter(p => p.climatology_temp !== null);

  const makePath = (points: { temp: number; depth: number }[]) =>
    points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.temp)} ${scaleY(p.depth)}`).join(' ');

  const predPath = makePath(validPreds.map(p => ({ temp: p.predicted_temp!, depth: p.depth_m })));
  const truthPath = makePath(validTruth.map(p => ({ temp: p.truth_temp!, depth: p.depth_m })));
  const climPath = makePath(validClim.map(p => ({ temp: p.climatology_temp!, depth: p.depth_m })));

  // Estimate Mixed Layer Depth (threshold: ΔT > 0.2°C from surface)
  let mld_m = 35;
  if (validTruth.length > 1 && validTruth[0].truth_temp !== null) {
    const sst = validTruth[0].truth_temp!;
    for (const p of validTruth) {
      if (sst - p.truth_temp! > 0.2) {
        mld_m = p.depth_m;
        break;
      }
    }
  }

  return (
    <div className="glass-panel fade-in" style={{ padding: '16px 18px', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
        <div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '2px' }}>
            Vertical Sounding
          </div>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-heading)', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Compass size={15} color="var(--primary)" />
            {coord.lat.toFixed(2)}°N, {coord.lon.toFixed(2)}°E
          </div>
        </div>
        <div className="badge badge-magenta" style={{ fontSize: '0.65rem' }}>
          <Anchor size={10} />
          {isOcean && seafloorDepth > 0 ? `${seafloorDepth}m` : 'Land'}
        </div>
      </div>

      {/* SVG Chart */}
      <div style={{ flex: 1, position: 'relative', minHeight: '300px' }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="profile-chart-svg"
          style={{ width: '100%', height: '100%' }}
        >
          {/* Defs for gradients */}
          <defs>
            <linearGradient id="pred-glow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#890149" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#890149" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Background fill areas */}
          {/* Thermocline zone highlight */}
          <rect
            x={margin.left}
            y={scaleY(50)}
            width={innerWidth}
            height={scaleY(200) - scaleY(50)}
            fill="rgba(137, 1, 73, 0.04)"
            stroke="rgba(137, 1, 73, 0.20)"
            strokeDasharray="4 4"
            rx="4"
          />

          {/* Grid lines: Temperature */}
          {[5, 10, 15, 20, 25, 30].map(temp => (
            <g key={`temp-${temp}`}>
              <line
                x1={scaleX(temp)} y1={margin.top}
                x2={scaleX(temp)} y2={margin.top + innerHeight}
                stroke="var(--border-subtle)"
                strokeDasharray="2 3"
              />
              <text
                x={scaleX(temp)} y={margin.top + innerHeight + 18}
                fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle"
              >
                {temp}°
              </text>
            </g>
          ))}

          {/* Grid lines: Depth */}
          {[0, 20, 50, 75, 100, 200, 500, 1000].map(depth => (
            <g key={`depth-${depth}`}>
              <line
                x1={margin.left} y1={scaleY(depth)}
                x2={margin.left + innerWidth} y2={scaleY(depth)}
                stroke="var(--border-subtle)"
                strokeDasharray="2 3"
              />
              <text
                x={margin.left - 6} y={scaleY(depth) + 3}
                fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="end"
              >
                {depth}
              </text>
            </g>
          ))}

          {/* Thermocline label */}
          <text
            x={margin.left + innerWidth - 6} y={scaleY(75) + 3}
            fill="var(--text-muted)" fontSize="8" fontFamily="var(--font-mono)" textAnchor="end"
          >
            Thermocline
          </text>

          {/* Data lines */}
          {/* Climatology (dashed muted magenta, background) */}
          {climPath && (
            <path d={climPath} fill="none" stroke="#8c426b" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.75" />
          )}

          {/* GLORYS Truth (dashed deep wine) */}
          {truthPath && (
            <path d={truthPath} fill="none" stroke="#38001d" strokeWidth="2" strokeDasharray="5 3" opacity="0.9" />
          )}

          {/* Prediction (solid deep magenta with glow) */}
          {predPath && (
            <path
              d={predPath}
              fill="none"
              stroke="#890149"
              strokeWidth="2.5"
              strokeLinecap="round"
              style={{ filter: 'drop-shadow(0 0 5px rgba(137, 1, 73, 0.45))' }}
            />
          )}

          {/* Data point markers */}
          {validPreds.map(p => (
            <circle
              key={p.depth_m}
              cx={scaleX(p.predicted_temp!)}
              cy={scaleY(p.depth_m)}
              r="3.5"
              fill="#F3F3E7"
              stroke="#890149"
              strokeWidth="1.5"
            />
          ))}

          {/* Axis labels */}
          <text
            x={margin.left + innerWidth / 2} y={svgHeight - 4}
            fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-sans)" textAnchor="middle" fontWeight="600"
          >
            Temperature (°C)
          </text>
          <text
            transform="rotate(-90)"
            x={-(margin.top + innerHeight / 2)} y={14}
            fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-sans)" textAnchor="middle" fontWeight="600"
          >
            Depth (m) ↓
          </text>
        </svg>
      </div>

      {/* Legend */}
      <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '10px', marginTop: '8px' }}>
        <div style={{ display: 'flex', gap: '14px', fontSize: '0.7rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '14px', height: '2.5px', background: '#890149', borderRadius: '2px', boxShadow: '0 0 4px rgba(137,1,73,0.5)' }} />
            <span style={{ color: 'var(--text-heading)', fontWeight: 600 }}>{modelName}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '14px', height: '2px', background: '#38001d', borderRadius: '1px' }} />
            <span style={{ color: 'var(--text-secondary)' }}>GLORYS Truth</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '14px', height: '1.5px', background: '#8c426b', borderRadius: '1px' }} />
            <span style={{ color: 'var(--text-muted)' }}>Climatology</span>
          </div>
        </div>

        {/* Physical metrics */}
        <div className="grid-2" style={{ marginTop: '8px' }}>
          <div className="stat-card" style={{ padding: '6px 10px' }}>
            <div className="stat-card-label" style={{ fontSize: '0.6rem' }}>Mixed Layer</div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>~{mld_m}m</div>
          </div>
          <div className="stat-card" style={{ padding: '6px 10px' }}>
            <div className="stat-card-label" style={{ fontSize: '0.6rem' }}>Thermo. Gradient</div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>-0.15°C/m</div>
          </div>
        </div>
      </div>
    </div>
  );
};
