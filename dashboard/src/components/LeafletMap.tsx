/**
 * LeafletMap.tsx — Interactive geo-referenced heatmap using react-leaflet.
 *
 * Design decision: We render the temperature grid as a Leaflet ImageOverlay
 * by painting onto an offscreen canvas, then using its data URL. This gives us
 * real coastlines, zoom/pan, and geographic context while keeping the fast
 * pixel-level rendering of our 101×241 grid.
 */
import React, { useRef, useEffect, useMemo, useState } from 'react';
import { MapContainer, TileLayer, ImageOverlay, Marker, Popup, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { getThermalColor } from '../utils/colormap';

interface LeafletMapProps {
  predictionGrid: (number | null)[][];
  latitudes: number[];
  longitudes: number[];
  depth_m: number;
  modelName: string;
  pinnedCoord: { lat: number; lon: number };
  onPinCoord: (lat: number, lon: number) => void;
  theme?: 'light' | 'dark';
}

// Deep Magenta pin marker with Alabaster border
const pinIcon = L.divIcon({
  className: '',
  html: `<div style="
    width: 16px; height: 16px;
    background: #890149;
    border: 2px solid #F3F3E7;
    border-radius: 50%;
    box-shadow: 0 0 12px rgba(137,1,73,0.7), 0 0 24px rgba(137,1,73,0.35);
    transform: translate(-8px, -8px);
  "></div>`,
  iconSize: [16, 16],
  iconAnchor: [0, 0],
});

/**
 * Inner component to handle click-to-pin events on the map.
 * Must be inside <MapContainer> to use useMapEvents.
 */
function ClickHandler({ onPinCoord }: { onPinCoord: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      const lat = Math.round(e.latlng.lat * 100) / 100;
      const lon = Math.round(e.latlng.lng * 100) / 100;
      // Only pin if within the North Indian Ocean domain
      if (lat >= 5 && lat <= 30 && lon >= 45 && lon <= 105) {
        onPinCoord(lat, lon);
      }
    },
  });
  return null;
}

/**
 * Handles re-centering when pinnedCoord changes (e.g. from region buttons).
 */
function MapUpdater({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.panTo(center, { animate: true, duration: 0.5 });
  }, [center[0], center[1]]);
  return null;
}

export const LeafletMap: React.FC<LeafletMapProps> = ({
  predictionGrid,
  latitudes,
  longitudes,
  depth_m,
  modelName,
  pinnedCoord,
  onPinCoord,
  theme = 'light',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [overlayUrl, setOverlayUrl] = useState<string>('');

  const H = latitudes.length || 101;
  const W = longitudes.length || 241;

  // Define geo bounds for the overlay: SW corner to NE corner
  const latMin = latitudes[0] ?? 5.0;
  const latMax = latitudes[latitudes.length - 1] ?? 30.0;
  const lonMin = longitudes[0] ?? 45.0;
  const lonMax = longitudes[longitudes.length - 1] ?? 105.0;

  const bounds: L.LatLngBoundsExpression = [[latMin, lonMin], [latMax, lonMax]];

  // Paint the temperature grid onto an offscreen canvas -> data URL
  useEffect(() => {
    if (!predictionGrid || predictionGrid.length === 0 || H === 0 || W === 0) return;

    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
    }
    const canvas = canvasRef.current;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(W, H);
    const data = imgData.data;

    for (let i = 0; i < H; i++) {
      const gridRow = H - 1 - i;
      const row = predictionGrid[gridRow];
      for (let j = 0; j < W; j++) {
        const val = row ? row[j] : null;
        const color = getThermalColor(val !== null && val !== undefined ? val : NaN, 4.0, 31.0);
        const idx = (i * W + j) * 4;
        data[idx] = color.r;
        data[idx + 1] = color.g;
        data[idx + 2] = color.b;
        data[idx + 3] = (val === null || val === undefined) ? 0 : 215;
      }
    }
    ctx.putImageData(imgData, 0, 0);
    setOverlayUrl(canvas.toDataURL('image/png'));
  }, [predictionGrid, H, W]);

  const mapCenter: [number, number] = useMemo(() => [pinnedCoord.lat, pinnedCoord.lon], [pinnedCoord.lat, pinnedCoord.lon]);
  const isDark = theme === 'dark';
  const tileBase = isDark
    ? 'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png'
    : 'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png';
  const tileLabels = isDark
    ? 'https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png'
    : 'https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png';

  return (
    <div className="map-container" style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-glass)' }}>
      <MapContainer
        center={[17.5, 72.0]}
        zoom={4}
        minZoom={3}
        maxZoom={8}
        style={{ width: '100%', height: '100%', minHeight: '460px' }}
        maxBounds={[[0, 35], [35, 115]]}
        maxBoundsViscosity={1.0}
      >
        <TileLayer
          key={tileBase}
          url={tileBase}
          attribution='&copy; <a href="https://carto.com/">CARTO</a> | &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
        />

        {overlayUrl && (
          <ImageOverlay
            url={overlayUrl}
            bounds={bounds}
            opacity={0.88}
            zIndex={10}
          />
        )}

        <TileLayer
          key={tileLabels}
          url={tileLabels}
          zIndex={20}
        />

        <Marker position={[pinnedCoord.lat, pinnedCoord.lon]} icon={pinIcon}>
          <Popup>
            <div style={{ color: 'var(--text-main)', lineHeight: 1.6 }}>
              <div style={{ fontWeight: 700, color: 'var(--primary)', marginBottom: '4px' }}>
                📍 Profile Location
              </div>
              <div>Lat: <strong>{pinnedCoord.lat.toFixed(2)}°N</strong></div>
              <div>Lon: <strong>{pinnedCoord.lon.toFixed(2)}°E</strong></div>
              <div>Depth: <strong>{depth_m}m</strong></div>
              <div>Model: <strong>{modelName}</strong></div>
            </div>
          </Popup>
        </Marker>

        <ClickHandler onPinCoord={onPinCoord} />
        <MapUpdater center={mapCenter} />
      </MapContainer>

      {/* Floating colorbar legend */}
      <div style={{
        position: 'absolute',
        bottom: '36px',
        left: '12px',
        right: '12px',
        maxWidth: '340px',
        zIndex: 1000,
        background: 'var(--bg-glass-card)',
        backdropFilter: 'blur(16px)',
        border: '1px solid var(--border-glass)',
        borderRadius: 'var(--radius-sm)',
        padding: '8px 12px',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        fontSize: '0.7rem',
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-secondary)',
        boxShadow: 'var(--shadow-md)',
      }}>
        <span style={{ whiteSpace: 'nowrap', fontWeight: 600 }}>4°C</span>
        <div className="colorbar-gradient" />
        <span style={{ whiteSpace: 'nowrap', fontWeight: 600 }}>31°C</span>
        <span style={{ color: 'var(--text-muted)', marginLeft: '4px', whiteSpace: 'nowrap' }}>
          @ {depth_m}m
        </span>
      </div>

      {/* Floating model info badge */}
      <div style={{
        position: 'absolute',
        top: '12px',
        right: '12px',
        zIndex: 1000,
        background: 'var(--bg-glass-card)',
        backdropFilter: 'blur(16px)',
        border: '1px solid var(--border-glass)',
        borderRadius: 'var(--radius-sm)',
        padding: '6px 12px',
        fontSize: '0.7rem',
        fontFamily: 'var(--font-mono)',
        color: 'var(--primary)',
        fontWeight: 700,
        boxShadow: 'var(--shadow-sm)',
      }}>
        {modelName} · {depth_m}m
      </div>
    </div>
  );
};
