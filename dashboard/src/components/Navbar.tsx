import React from 'react';
import { Waves, Award, Wifi, WifiOff } from 'lucide-react';

interface NavbarProps {
  modelName: string;
  onModelChange: (model: string) => void;
  onOpenMetrics: () => void;
  isApiOnline: boolean;
  onSelectSubbasin: (lat: number, lon: number) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  modelName,
  onModelChange,
  onOpenMetrics,
  isApiOnline,
  onSelectSubbasin,
}) => {
  return (
    <header className="glass-panel" style={{
      padding: '14px 28px',
      marginBottom: '20px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      flexWrap: 'wrap',
      gap: '14px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{
          width: '42px',
          height: '42px',
          borderRadius: '12px',
          background: 'linear-gradient(135deg, #890149 0%, #aa025b 50%, #54002c 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#F3F3E7',
          boxShadow: 'var(--shadow-glow)'
        }}>
          <Waves size={24} />
        </div>

        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1 style={{ fontSize: '1.4rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text-heading)' }}>
              Ocean<span style={{ color: 'var(--primary)' }}>Embed</span>
            </h1>
            <span className="badge badge-magenta">MoES / INCOIS</span>
            <span className="badge badge-solid">Disaster Management</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            Satellite Embedding Deep Learning Framework for 3D Subsurface Ocean Temperature Reconstruction
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginRight: '4px' }}>Region:</span>
        {[
          { name: 'NIO Full', lat: 15.0, lon: 65.0 },
          { name: 'Arabian Sea', lat: 17.5, lon: 64.0 },
          { name: 'Bay of Bengal', lat: 15.0, lon: 88.0 },
          { name: 'Equatorial', lat: 6.0, lon: 78.0 },
        ].map(b => (
          <button
            key={b.name}
            onClick={() => onSelectSubbasin(b.lat, b.lon)}
            className="btn-select"
            style={{ fontSize: '0.75rem', padding: '5px 10px' }}
          >
            {b.name}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Architecture:</span>
          <select
            value={modelName}
            onChange={(e) => onModelChange(e.target.value)}
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-glass)',
              color: 'var(--cyan-glow)',
              padding: '6px 12px',
              borderRadius: '8px',
              fontSize: '0.8rem',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              outline: 'none',
              cursor: 'pointer'
            }}
          >
            <option value="attention_hybrid">Attention Hybrid (Axial 2D)</option>
            <option value="direct_cnn">Direct CNN Residual</option>
            <option value="climatology">Climatology Baseline</option>
          </select>
        </div>

        <button
          onClick={onOpenMetrics}
          className="btn-select"
          style={{
            borderColor: 'rgba(0, 240, 255, 0.4)',
            color: 'var(--cyan-glow)',
            boxShadow: '0 0 10px rgba(0, 240, 255, 0.15)'
          }}
        >
          <Award size={16} />
          <span>Verification & Argo</span>
        </button>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          background: isApiOnline ? 'rgba(6, 214, 160, 0.1)' : 'rgba(255, 183, 3, 0.1)',
          border: `1px solid ${isApiOnline ? 'rgba(6, 214, 160, 0.3)' : 'rgba(255, 183, 3, 0.3)'}`,
          padding: '5px 10px',
          borderRadius: '9999px',
          fontSize: '0.7rem',
          fontFamily: 'var(--font-mono)',
          color: isApiOnline ? 'var(--teal-glow)' : 'var(--amber-warn)'
        }}>
          {isApiOnline ? <Wifi size={12} /> : <WifiOff size={12} />}
          <span>{isApiOnline ? 'FastAPI Online' : 'Local Cached Demo'}</span>
        </div>
      </div>
    </header>
  );
};
