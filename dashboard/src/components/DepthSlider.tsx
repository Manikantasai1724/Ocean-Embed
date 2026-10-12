import React from 'react';
import { Layers, ChevronLeft, ChevronRight } from 'lucide-react';

interface DepthSliderProps {
  depths: number[];
  currentDepth: number;
  onDepthChange: (depth: number) => void;
}

export const DepthSlider: React.FC<DepthSliderProps> = ({
  depths,
  currentDepth,
  onDepthChange,
}) => {
  const currentIndex = depths.indexOf(currentDepth);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const idx = parseInt(e.target.value, 10);
    onDepthChange(depths[idx]);
  };

  const stepDepth = (delta: number) => {
    const nextIdx = Math.max(0, Math.min(depths.length - 1, currentIndex + delta));
    onDepthChange(depths[nextIdx]);
  };

  let zoneName = 'Epipelagic (Mixed Layer)';
  let zoneColor = '#00f0ff';
  let zoneDesc = 'Direct solar heating & strong wind-driven turbulent mixing';
  if (currentDepth >= 50 && currentDepth <= 200) {
    zoneName = 'Main Thermocline Layer';
    zoneColor = '#ffb703';
    zoneDesc = 'Steep vertical temperature gradient (dT/dz) & internal waves';
  } else if (currentDepth > 200) {
    zoneName = 'Mesopelagic / Deep Water';
    zoneColor = '#3a86ff';
    zoneDesc = 'Stable abyssal water masses; near-constant cold temperatures';
  }

  return (
    <div className="glass-panel" style={{ padding: '20px 24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            background: 'rgba(0, 240, 255, 0.1)',
            padding: '8px',
            borderRadius: '8px',
            color: 'var(--cyan-glow)'
          }}>
            <Layers size={20} />
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Subsurface Depth Target
            </div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#ffffff', display: 'flex', alignItems: 'baseline', gap: '6px' }}>
              <span>{currentDepth}</span>
              <span style={{ fontSize: '0.9rem', color: 'var(--cyan-glow)', fontWeight: 500 }}>meters</span>
            </div>
          </div>
        </div>

        <div style={{
          textAlign: 'right',
          background: 'rgba(255, 255, 255, 0.03)',
          border: `1px solid ${zoneColor}40`,
          padding: '6px 14px',
          borderRadius: '9999px',
        }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: zoneColor }}>
            {zoneName}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {zoneDesc}
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <button
          onClick={() => stepDepth(-1)}
          disabled={currentIndex <= 0}
          className="btn-select"
          style={{ padding: '8px', opacity: currentIndex <= 0 ? 0.3 : 1 }}
          title="Shallower Depth"
        >
          <ChevronLeft size={16} />
        </button>

        <div style={{ flex: 1, position: 'relative' }}>
          <input
            type="range"
            min={0}
            max={depths.length - 1}
            value={currentIndex >= 0 ? currentIndex : 0}
            onChange={handleSliderChange}
            className="depth-slider"
          />
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: '6px',
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-muted)'
          }}>
            {depths.map((d) => (
              <span
                key={d}
                onClick={() => onDepthChange(d)}
                style={{
                  cursor: 'pointer',
                  color: d === currentDepth ? 'var(--cyan-glow)' : undefined,
                  fontWeight: d === currentDepth ? 700 : 400,
                }}
              >
                {d === 0 ? '0m' : d === 75 ? '75m' : d === 200 ? '200m' : d === 1000 ? '1000m' : ''}
              </span>
            ))}
          </div>
        </div>

        <button
          onClick={() => stepDepth(1)}
          disabled={currentIndex >= depths.length - 1}
          className="btn-select"
          style={{ padding: '8px', opacity: currentIndex >= depths.length - 1 ? 0.3 : 1 }}
          title="Deeper Depth"
        >
          <ChevronRight size={16} />
        </button>
      </div>

      <div style={{ display: 'flex', gap: '8px', marginTop: '14px', flexWrap: 'wrap' }}>
        {[
          { label: 'Surface (0m)', depth: 0 },
          { label: 'Mixed Layer (30m)', depth: 30 },
          { label: 'Thermocline Core (75m)', depth: 75 },
          { label: 'Sub-thermocline (150m)', depth: 150 },
          { label: 'Mesopelagic (500m)', depth: 500 },
          { label: 'Abyssal Floor (1000m)', depth: 1000 },
        ].map(preset => (
          <button
            key={preset.depth}
            onClick={() => onDepthChange(preset.depth)}
            className={`btn-select ${currentDepth === preset.depth ? 'active' : ''}`}
            style={{ fontSize: '0.75rem', padding: '4px 10px' }}
          >
            {preset.label}
          </button>
        ))}
      </div>
    </div>
  );
};
