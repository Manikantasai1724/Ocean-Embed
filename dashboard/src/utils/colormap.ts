/**
 * Oceanographic Colormap Utilities for OceanEmbed
 * Maps temperature in °C (4°C to 32°C) to RGB values.
 * Uses a smooth ocean thermal colormap (purple/dark blue -> cyan -> yellow -> coral/red).
 */

export interface RGB {
  r: number;
  g: number;
  b: number;
}

export function getThermalColor(val: number, min: number = 4.0, max: number = 31.0): RGB {
  if (isNaN(val) || val === null) {
    // Land mask / subterranean
    return { r: 15, g: 23, b: 42 }; // Dark slate land
  }

  // Normalize to [0, 1]
  const t = Math.max(0, Math.min(1, (val - min) / Math.max(1e-4, max - min)));

  // 5-stop thermal gradient
  // 0.0: Deep cold abyss (#1d1145 - 29, 17, 69)
  // 0.25: Cool sub-thermocline (#25567b - 37, 86, 123)
  // 0.5: Mild thermocline (#00b4d8 - 0, 180, 216)
  // 0.75: Warm surface (#ffb703 - 255, 183, 3)
  // 1.0: Hot tropical surface (#ff006e - 255, 0, 110)

  let r = 0, g = 0, b = 0;

  if (t < 0.25) {
    const s = t / 0.25;
    r = 29 + s * (37 - 29);
    g = 17 + s * (86 - 17);
    b = 69 + s * (123 - 69);
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    r = 37 + s * (0 - 37);
    g = 86 + s * (180 - 86);
    b = 123 + s * (216 - 123);
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    r = 0 + s * (255 - 0);
    g = 180 + s * (183 - 180);
    b = 216 + s * (3 - 216);
  } else {
    const s = (t - 0.75) / 0.25;
    r = 255 + s * (255 - 255);
    g = 183 + s * (0 - 183);
    b = 3 + s * (110 - 3);
  }

  return { r: Math.round(r), g: Math.round(g), b: Math.round(b) };
}

export function getErrorColor(val: number, maxAbsErr: number = 1.5): RGB {
  if (isNaN(val) || val === null) {
    return { r: 15, g: 23, b: 42 };
  }
  // Diverging colormap: Blue (negative/cold bias) -> White/Gray (0) -> Red (positive/warm bias)
  const norm = Math.max(-1, Math.min(1, val / maxAbsErr));
  if (norm < 0) {
    const s = -norm;
    return {
      r: Math.round(200 - s * 170),
      g: Math.round(200 - s * 100),
      b: 255,
    };
  } else {
    const s = norm;
    return {
      r: 255,
      g: Math.round(200 - s * 170),
      b: Math.round(200 - s * 170),
    };
  }
}
