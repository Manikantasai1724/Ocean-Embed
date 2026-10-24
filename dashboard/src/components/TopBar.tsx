/**
 * TopBar.tsx — Minimal top navigation bar for OceanEmbed.
 *
 * Displays brand, status indicators, and action buttons.
 * Strictly adheres to Alabaster (#F3F3E7) & Deep Magenta (#890149).
 */
import { Waves, Award, Sun, Moon } from 'lucide-react';

interface TopBarProps {
  isApiOnline: boolean;
  onOpenMetrics: () => void;
  theme?: 'light' | 'dark';
  onToggleTheme?: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  isApiOnline,
  onOpenMetrics,
  theme = 'light',
  onToggleTheme,
}) => {
  return (
    <header className="topbar app-topbar">
      {/* Left: Brand */}
      <div className="topbar-brand">
        <div className="topbar-logo">
          <Waves size={20} />
        </div>
        <div>
          <div className="topbar-title">
            Ocean<span>Embed</span>
          </div>
          <div className="topbar-subtitle">
            Satellite Embedding DL Framework for 3D Subsurface Ocean Temperature
          </div>
        </div>
      </div>

      {/* Center: Badges */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span className="badge badge-magenta">MoES / INCOIS</span>
        <span className="badge badge-solid">Disaster Mgmt</span>
        <span className="badge badge-alabaster">North Indian Ocean</span>
      </div>

      {/* Right: Actions */}
      <div className="topbar-actions">
        {onToggleTheme && (
          <button
            onClick={onToggleTheme}
            className="btn-select"
            title={`Switch to ${theme === 'light' ? 'Deep Magenta' : 'Alabaster'} mode`}
            style={{ padding: '6px 10px', fontSize: '0.75rem' }}
          >
            {theme === 'light' ? <Moon size={13} /> : <Sun size={13} />}
            <span>{theme === 'light' ? 'Jewel Dark' : 'Alabaster'}</span>
          </button>
        )}

        <button
          onClick={onOpenMetrics}
          className="btn-primary"
        >
          <Award size={14} />
          Verification & Argo
        </button>

        <a
          href="https://github.com/Manikantasai1724/Ocean-Embed"
          target="_blank"
          rel="noreferrer"
          className="btn-select"
          style={{ textDecoration: 'none', padding: '7px 10px' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
          </svg>
        </a>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 10px',
        }}>
          <span className={`status-dot ${isApiOnline ? 'online' : 'offline'}`} />
          <span style={{
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            color: 'var(--primary)',
          }}>
            {isApiOnline ? 'API Live' : 'Demo Mode'}
          </span>
        </div>
      </div>
    </header>
  );
};
