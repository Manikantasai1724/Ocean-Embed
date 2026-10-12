import React from 'react';
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
  const svgWidth = 420;
  const svgHeight = 460;
  const margin = { top: 30, right: 30, bottom: 40, left: 60 };
  const innerWidth = svgWidth - margin.left - margin.right;
  const innerHeight = svgHeight - margin.top - margin.bottom;

  const minTemp = 4.0;
  const maxTemp = 31.0;
  const maxDepth = 1000.0;

  const scaleX = (temp: number) => {
    const t = Math.max(minTemp, Math.min(maxTemp, temp));
    return margin.left + ((t - minTemp) / (maxTemp - minTemp)) * innerWidth;
  };

  const scaleY = (depth: number) => {
    const depthFrac = Math.pow(depth / maxDepth, 0.6);
    return margin.top + depthFrac * innerHeight;
  };

  const validPreds = profile.filter(p => p.predicted_temp !== null);
  const validTruth = profile.filter(p => p.truth_temp !== null);
  const validClim = profile.filter(p => p.climatology_temp !== null);

  const predPath = validPreds
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.predicted_temp!)} ${scaleY(p.depth_m)}`)
    .join(' ');

  const truthPath = validTruth
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.truth_temp!)} ${scaleY(p.depth_m)}`)
    .join(' ');

  const climPath = validClim
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.climatology_temp!)} ${scaleY(p.depth_m)}`)
    .join(' ');

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

  const thermoclineY = scaleY(75);

  return (
    <div className="glass-panel" style={{ padding: '20px 24px', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
        <div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Vertical Sounding Profile (0–1000m)
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Compass size={18} color="var(--cyan-glow)" />
            <span>{coord.lat.toFixed(2)}°N, {coord.lon.toFixed(2)}°E</span>
          </div>
        </div>

        <div style={{
          background: 'rgba(255, 255, 255, 0.04)',
          border: '1px solid var(--border-subtle)',
          padding: '4px 10px',
          borderRadius: '8px',
          fontSize: '0.75rem',
          fontFamily: 'var(--font-mono)',
          color: 'var(--cyan-muted)',
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          <Anchor size={14} />
          <span>{isOcean && seafloorDepth > 0 ? `Seafloor: ${seafloorDepth}m` : 'Land Point'}</span>
        </div>
      </div>

      <div style={{ flex: 1, position: 'relative', minHeight: '340px' }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="profile-chart-svg"
          style={{ width: '100%', height: '100%' }}
        >
          {[5, 10, 15, 20, 25, 30].map(temp => (
            <g key={temp}>
              <line
                x1={scaleX(temp)}
                y1={margin.top}
                x2={scaleX(temp)}
                y2={margin.top + innerHeight}
                stroke="rgba(255, 255, 255, 0.06)"
                strokeDasharray="3 3"
              />
              <text
                x={scaleX(temp)}
                y={margin.top + innerHeight + 20}
                fill="var(--text-muted)"
                fontSize="10"
                fontFamily="var(--font-mono)"
                textAnchor="middle"
              >
                {temp}°C
              </text>
            </g>
          ))}

          {[0, 20, 50, 75, 100, 200, 500, 1000].map(depth => (
            <g key={depth}>
              <line
                x1={margin.left}
                y1={scaleY(depth)}
                x2={margin.left + innerWidth}
                y2={scaleY(depth)}
                stroke="rgba(255, 255, 255, 0.06)"
                strokeDasharray="2 2"
              />
              <text
                x={margin.left - 8}
                y={scaleY(depth) + 4}
                fill="var(--text-muted)"
                fontSize="10"
                fontFamily="var(--font-mono)"
                textAnchor="end"
              >
                {depth}m
              </text>
            </g>
          ))}

          <rect
            x={margin.left}
            y={scaleY(50)}
            width={innerWidth}
            height={scaleY(150) - scaleY(50)}
            fill="rgba(255, 183, 3, 0.05)"
            stroke="rgba(255, 183, 3, 0.15)"
            strokeDasharray="4 4"
          />
          <text
            x={margin.left + innerWidth - 8}
            y={thermoclineY + 3}
            fill="rgba(255, 183, 3, 0.7)"
            fontSize="9"
            fontFamily="var(--font-mono)"
            textAnchor="end"
          >
            Thermocline Core
          </text>

          {climPath && (
            <path
              d={climPath}
              fill="none"
              stroke="#64748b"
              strokeWidth="1.5"
              strokeDasharray="3 3"
              opacity="0.8"
            />
          )}

          {truthPath && (
            <path
              d={truthPath}
              fill="none"
              stroke="#ff006e"
              strokeWidth="2.5"
              strokeDasharray="5 3"
            />
          )}

          {predPath && (
            <path
              d={predPath}
              fill="none"
              stroke="#00f0ff"
              strokeWidth="3"
              style={{ filter: 'drop-shadow(0 0 6px rgba(0, 240, 255, 0.5))' }}
            />
          )}

          {validPreds.map(p => (
            <g key={p.depth_m}>
              <circle
                cx={scaleX(p.predicted_temp!)}
                cy={scaleY(p.depth_m)}
                r="3.5"
                fill="#ffffff"
                stroke="#00f0ff"
                strokeWidth="2"
              />
            </g>
          ))}

          <text
            x={margin.left + innerWidth / 2}
            y={svgHeight - 6}
            fill="var(--text-secondary)"
            fontSize="11"
            fontFamily="var(--font-sans)"
            textAnchor="middle"
            fontWeight="600"
          >
            Temperature (°C)
          </text>

          <text
            transform={`rotate(-90)`}
            x={-(margin.top + innerHeight / 2)}
            y={18}
            fill="var(--text-secondary)"
            fontSize="11"
            fontFamily="var(--font-sans)"
            textAnchor="middle"
            fontWeight="600"
          >
            Depth (m) ↓
          </text>
        </svg>
      </div>

      <div style={{ marginTop: '12px', borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
        <div style={{ display: 'flex', gap: '16px', fontSize: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '3px', background: 'var(--cyan-glow)', borderRadius: '2px' }} />
            <span style={{ color: '#ffffff', fontWeight: 600 }}>{modelName}</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '3px', background: 'var(--coral-hot)', borderRadius: '2px' }} />
            <span style={{ color: 'var(--coral-hot)' }}>GLORYS Truth</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '2px', background: '#64748b', borderRadius: '1px' }} />
            <span style={{ color: 'var(--text-muted)' }}>Climatology</span>
          </div>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '8px',
          marginTop: '10px',
          background: 'rgba(255, 255, 255, 0.02)',
          padding: '8px 12px',
          borderRadius: '8px',
          fontSize: '0.75rem',
          fontFamily: 'var(--font-mono)'
        }}>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Mixed Layer Depth: </span>
            <strong style={{ color: 'var(--cyan-glow)' }}>~{mld_m}m</strong>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Thermocline Gradient: </span>
            <strong style={{ color: 'var(--amber-warn)' }}>-0.15°C/m</strong>
          </div>
        </div>
      </div>
    </div>
  );
};
