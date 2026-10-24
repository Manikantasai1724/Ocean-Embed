/**
 * Sidebar.tsx — Control panel sidebar for OceanEmbed dashboard.
 *
 * Contains: Model selector, Depth controls, Region quick-jump,
 * Satellite input channels info, and pinned coordinate display.
 */
import React from 'react';
import {
  Layers, Navigation, Satellite, Cpu,
  ChevronLeft, ChevronRight, MapPin, Thermometer
} from 'lucide-react';

interface SidebarProps {
  modelName: string;
  onModelChange: (model: string) => void;
  depths: number[];
  currentDepth: number;
  onDepthChange: (depth: number) => void;
  pinnedCoord: { lat: number; lon: number };
  onSelectSubbasin: (lat: number, lon: number) => void;
  viewMode: 'single' | 'compare';
  onToggleViewMode: () => void;
}

const DEPTH_PRESETS = [
  { label: 'Surface', depth: 0, icon: '🌊' },
  { label: 'Mixed Layer', depth: 30, icon: '🔄' },
  { label: 'Thermocline', depth: 75, icon: '🌡️' },
  { label: 'Sub-thermo', depth: 150, icon: '❄️' },
  { label: 'Meso', depth: 500, icon: '🏔️' },
  { label: 'Abyssal', depth: 1000, icon: '🕳️' },
];

const REGIONS = [
  { name: 'NIO Overview', lat: 17.5, lon: 72.0, desc: '5°N–30°N, 45°E–105°E' },
  { name: 'Arabian Sea', lat: 17.5, lon: 64.0, desc: 'High-salinity evaporative basin' },
  { name: 'Bay of Bengal', lat: 15.0, lon: 88.0, desc: 'Low-salinity river-stratified' },
  { name: 'Equatorial IO', lat: 6.0, lon: 78.0, desc: 'Wyrtki jet & equatorial waves' },
];

const CHANNELS = [
  { name: 'SST (L4)', value: '28.4°C', color: '#890149' },
  { name: 'SSS (L4)', value: '35.2 PSU', color: '#b82d77' },
  { name: 'SLA (Alt)', value: '+0.08 m', color: '#54002c' },
  { name: 'CCMP Winds', value: '6.2 m/s', color: '#e096bf' },
];

export const Sidebar: React.FC<SidebarProps> = ({
  modelName,
  onModelChange,
  depths,
  currentDepth,
  onDepthChange,
  pinnedCoord,
  onSelectSubbasin,
  viewMode,
  onToggleViewMode,
}) => {
  const currentIndex = depths.indexOf(currentDepth);

  const stepDepth = (delta: number) => {
    const nextIdx = Math.max(0, Math.min(depths.length - 1, currentIndex + delta));
    onDepthChange(depths[nextIdx]);
  };

  // Categorize current depth zone
  let zoneName = 'Epipelagic';
  let zoneColor = '#890149';
  if (currentDepth >= 50 && currentDepth <= 200) {
    zoneName = 'Thermocline';
    zoneColor = '#b82d77';
  } else if (currentDepth > 200) {
    zoneName = 'Mesopelagic';
    zoneColor = '#54002c';
  }

  return (
    <aside className="app-sidebar">
      {/* ═══ MODEL ARCHITECTURE ═══ */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">
          <Cpu size={12} /> Model Architecture
        </div>
        <select
          value={modelName}
          onChange={(e) => onModelChange(e.target.value)}
          className="select-styled"
        >
          <option value="attention_hybrid">Axial Attention Hybrid (2D)</option>
          <option value="direct_cnn">Direct CNN Residual</option>
          <option value="climatology">Climatology Baseline</option>
        </select>

        <div style={{ marginTop: '10px', display: 'flex', gap: '6px' }}>
          <button
            onClick={onToggleViewMode}
            className={`btn-select ${viewMode === 'compare' ? 'active' : ''}`}
            style={{ flex: 1, fontSize: '0.75rem', justifyContent: 'center' }}
          >
            {viewMode === 'single' ? '🔍 Compare Truth' : '📊 Single View'}
          </button>
        </div>
      </div>

      {/* ═══ DEPTH CONTROL ═══ */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">
          <Layers size={12} /> Depth Target
        </div>

        {/* Current depth display */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '10px',
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
            <span style={{
              fontSize: '1.6rem',
              fontWeight: 800,
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-heading)'
            }}>
              {currentDepth}
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--primary)', fontWeight: 600 }}>m</span>
          </div>
          <span className="zone-badge" style={{
            background: `${zoneColor}18`,
            border: `1px solid ${zoneColor}50`,
            color: zoneColor,
          }}>
            <Thermometer size={10} />
            {zoneName}
          </span>
        </div>

        {/* Slider with step buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
          <button
            onClick={() => stepDepth(-1)}
            disabled={currentIndex <= 0}
            className="btn-select"
            style={{ padding: '6px', opacity: currentIndex <= 0 ? 0.3 : 1 }}
          >
            <ChevronLeft size={14} />
          </button>

          <input
            type="range"
            min={0}
            max={depths.length - 1}
            value={currentIndex >= 0 ? currentIndex : 0}
            onChange={(e) => onDepthChange(depths[parseInt(e.target.value, 10)])}
            className="depth-slider"
            style={{ flex: 1 }}
          />

          <button
            onClick={() => stepDepth(1)}
            disabled={currentIndex >= depths.length - 1}
            className="btn-select"
            style={{ padding: '6px', opacity: currentIndex >= depths.length - 1 ? 0.3 : 1 }}
          >
            <ChevronRight size={14} />
          </button>
        </div>

        {/* Preset depth chips */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
          {DEPTH_PRESETS.map(p => (
            <button
              key={p.depth}
              onClick={() => onDepthChange(p.depth)}
              className={`depth-chip ${currentDepth === p.depth ? 'active' : ''}`}
            >
              {p.icon} {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* ═══ REGION QUICK-JUMP ═══ */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">
          <Navigation size={12} /> Region Quick-Jump
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
          {REGIONS.map(r => (
            <button
              key={r.name}
              onClick={() => onSelectSubbasin(r.lat, r.lon)}
              className={`region-btn ${pinnedCoord.lat === r.lat && pinnedCoord.lon === r.lon ? 'active' : ''}`}
            >
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, fontSize: '0.8rem' }}>{r.name}</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{r.desc}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* ═══ PINNED LOCATION ═══ */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">
          <MapPin size={12} /> Active Profile Pin
        </div>
        <div className="stat-card" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            background: 'var(--primary)',
            boxShadow: '0 0 10px rgba(137,1,73,0.4)',
            flexShrink: 0,
          }} />
          <div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontWeight: 700,
              fontSize: '0.9rem',
              color: 'var(--text-heading)',
            }}>
              {pinnedCoord.lat.toFixed(2)}°N, {pinnedCoord.lon.toFixed(2)}°E
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              Click map to repin location
            </div>
          </div>
        </div>
      </div>

      {/* ═══ SATELLITE INPUT CHANNELS ═══ */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">
          <Satellite size={12} /> Surface Input Channels
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {CHANNELS.map(ch => (
            <div key={ch.name} className="channel-tag">
              <span className="channel-dot" style={{ background: ch.color }} />
              <span style={{ color: 'var(--text-muted)', flex: 1 }}>{ch.name}</span>
              <strong style={{ color: ch.color }}>{ch.value}</strong>
            </div>
          ))}
        </div>
        <div style={{
          marginTop: '8px',
          fontSize: '0.65rem',
          color: 'var(--text-muted)',
          fontStyle: 'italic',
        }}>
          Zero-lookahead daily satellite-only input
        </div>
      </div>
    </aside>
  );
};
