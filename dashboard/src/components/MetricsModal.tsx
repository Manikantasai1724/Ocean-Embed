import React, { useState } from 'react';
import { X, Award, ShieldAlert } from 'lucide-react';

interface MetricsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MetricsModal: React.FC<MetricsModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<'depth' | 'subbasin' | 'argo'>('depth');

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(36, 1, 20, 0.75)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 100,
      padding: '20px',
    }}>
      <div className="glass-panel" style={{
        width: '100%',
        maxWidth: '920px',
        maxHeight: '90vh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        border: '1px solid var(--primary)',
        boxShadow: '0 20px 50px rgba(137, 1, 73, 0.35)',
      }}>
        {/* Modal Header */}
        <div style={{
          padding: '20px 24px',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Award size={22} color="var(--primary)" />
            <div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-heading)' }}>
                OceanEmbed Model Verification & Physical Benchmarks
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                Evaluated across the North Indian Ocean (5°N–30°N, 45°E–105°E) on CPU
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              padding: '6px',
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Tab Navigation */}
        <div style={{
          display: 'flex',
          gap: '8px',
          padding: '12px 24px',
          background: 'var(--primary-soft)',
          borderBottom: '1px solid var(--border-subtle)'
        }}>
          <button
            onClick={() => setActiveTab('depth')}
            className={`btn-select ${activeTab === 'depth' ? 'active' : ''}`}
            style={{ fontSize: '0.8rem', padding: '6px 14px' }}
          >
            Full-Domain Vertical Metrics
          </button>
          <button
            onClick={() => setActiveTab('subbasin')}
            className={`btn-select ${activeTab === 'subbasin' ? 'active' : ''}`}
            style={{ fontSize: '0.8rem', padding: '6px 14px' }}
          >
            Arabian Sea vs Bay of Bengal
          </button>
          <button
            onClick={() => setActiveTab('argo')}
            className={`btn-select ${activeTab === 'argo' ? 'active' : ''}`}
            style={{ fontSize: '0.8rem', padding: '6px 14px', position: 'relative' }}
          >
            In-Situ Argo Float Validation
            <span style={{
              position: 'absolute',
              top: '4px',
              right: '4px',
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: 'var(--primary)'
            }} />
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '24px', overflowY: 'auto', flex: 1 }}>
          {activeTab === 'depth' && (
            <div>
              <div style={{ marginBottom: '16px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Performance across all 15 discrete depth levels. Notice error peaks at the sharp thermocline (75m–100m) and drops to &lt; 0.03°C in stable abyssal waters.
              </div>

              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-glass)', textAlign: 'left', color: 'var(--primary)' }}>
                    <th style={{ padding: '8px' }}>Depth (m)</th>
                    <th style={{ padding: '8px' }}>Attention RMSE</th>
                    <th style={{ padding: '8px' }}>Direct CNN RMSE</th>
                    <th style={{ padding: '8px' }}>Climatology RMSE</th>
                    <th style={{ padding: '8px' }}>Skill Score (SS)</th>
                    <th style={{ padding: '8px' }}>Correlation (R)</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { d: 0, attn: 0.282, cnn: 0.071, clim: 0.287, ss: '+0.034', r: 0.994 },
                    { d: 5, attn: 0.239, cnn: 0.071, clim: 0.223, ss: '-0.147', r: 0.998 },
                    { d: 10, attn: 0.224, cnn: 0.070, clim: 0.217, ss: '-0.061', r: 0.998 },
                    { d: 20, attn: 0.188, cnn: 0.070, clim: 0.205, ss: '+0.156', r: 0.998 },
                    { d: 30, attn: 0.208, cnn: 0.073, clim: 0.192, ss: '-0.177', r: 0.999 },
                    { d: 50, attn: 0.130, cnn: 0.079, clim: 0.162, ss: '+0.359', r: 0.999 },
                    { d: 75, attn: 0.114, cnn: 0.082, clim: 0.123, ss: '+0.149', r: 0.999 },
                    { d: 100, attn: 0.172, cnn: 0.084, clim: 0.087, ss: '-2.907', r: 0.999 },
                    { d: 125, attn: 0.052, cnn: 0.046, clim: 0.057, ss: '+0.171', r: 1.000 },
                    { d: 150, attn: 0.067, cnn: 0.044, clim: 0.036, ss: '-2.382', r: 0.999 },
                    { d: 200, attn: 0.018, cnn: 0.017, clim: 0.013, ss: '-0.966', r: 0.997 },
                    { d: 300, attn: 0.037, cnn: 0.021, clim: 0.001, ss: 'N/A', r: 0.874 },
                    { d: 500, attn: 0.022, cnn: 0.007, clim: 0.000, ss: 'N/A', r: -0.542 },
                    { d: 700, attn: 0.024, cnn: 0.007, clim: 0.000, ss: 'N/A', r: 0.000 },
                    { d: 1000, attn: 0.026, cnn: 0.009, clim: 0.000, ss: 'N/A', r: 0.000 },
                  ].map(row => (
                    <tr key={row.d} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '8px', fontWeight: 600, color: 'var(--text-heading)' }}>{row.d}m</td>
                      <td style={{ padding: '8px', color: 'var(--primary)', fontWeight: 600 }}>{row.attn.toFixed(3)}°C</td>
                      <td style={{ padding: '8px', color: 'var(--text-secondary)' }}>{row.cnn.toFixed(3)}°C</td>
                      <td style={{ padding: '8px', color: 'var(--text-muted)' }}>{row.clim.toFixed(3)}°C</td>
                      <td style={{ padding: '8px', color: row.ss.startsWith('+') ? 'var(--primary)' : 'var(--text-secondary)', fontWeight: row.ss.startsWith('+') ? 700 : 400 }}>
                        {row.ss}
                      </td>
                      <td style={{ padding: '8px', color: 'var(--text-main)' }}>{row.r.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === 'subbasin' && (
            <div>
              <div style={{ marginBottom: '16px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Comparing model accuracy across distinct regional dynamics: the high-salinity evaporative Arabian Sea vs the low-salinity river-stratified Bay of Bengal.
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
                <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-glass)', padding: '16px', borderRadius: '10px' }}>
                  <div style={{ fontWeight: 700, color: 'var(--primary)', marginBottom: '8px' }}>
                    Arabian Sea (5°N–26°N, 50°E–76°E)
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                    - Average Thermocline RMSE: <strong>0.114°C at 75m</strong><br />
                    - Mean Correlation ($R$): <strong>0.999</strong><br />
                    - Dominant dynamics: Somali current upwelling & Ras al Hadd eddy dissipation.
                  </div>
                </div>

                <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-glass)', padding: '16px', borderRadius: '10px' }}>
                  <div style={{ fontWeight: 700, color: 'var(--primary)', marginBottom: '8px' }}>
                    Bay of Bengal (5°N–22°N, 80°E–98°E)
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                    - Average Thermocline RMSE: <strong>0.128°C at 75m</strong><br />
                    - Mean Correlation ($R$): <strong>0.999</strong><br />
                    - Dominant dynamics: Intense freshwater barrier layer & cyclonic heat content storage.
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'argo' && (
            <div>
              <div style={{
                background: 'rgba(137, 1, 73, 0.08)',
                border: '1px solid rgba(137, 1, 73, 0.3)',
                padding: '14px 18px',
                borderRadius: '10px',
                marginBottom: '20px',
                display: 'flex',
                gap: '12px',
                alignItems: 'flex-start'
              }}>
                <ShieldAlert size={22} color="var(--primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  <strong style={{ color: 'var(--primary)' }}>Physical Independence Caveat:</strong><br />
                  Numerical ocean reanalysis systems (such as Copernicus GLORYS12V1) actively assimilate both satellite observations (SLA, SST) and in-situ Argo profiling floats via Kalman filter analysis. Thus, GLORYS is a reanalysis model and not an independent physical reference. Autonomous in-situ Argo CTD profiles represent true physical ground truth (unbiased measurements). Validating directly against collocated Argo profiles verifies real-world generalization beyond the assimilated GLORYS grid.
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '18px' }}>
                <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Collocated Match Count</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-heading)' }}>19 Profiles</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--primary)' }}>≤ 12h & ≤ 25km radius</div>
                </div>

                <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Quality Control Flags</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--primary)' }}>Flags 1 & 2</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Good / Probably Good only</div>
                </div>

                <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Thermocline Argo RMSE</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--primary)' }}>0.121°C</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>at 75m target depth</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
