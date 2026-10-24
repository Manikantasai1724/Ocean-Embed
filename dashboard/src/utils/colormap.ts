/**
 * Oceanographic Colormap Utilities for OceanEmbed
 * Strictly mapped to:
 * - Color 1: Alabaster (#F3F3E7 / rgb(243, 243, 231)) - Warm Neutral
 * - Color 2: Deep Magenta (#890149 / rgb(137, 1, 73)) - Rich Jewel Tone
 */

export interface RGB {
  r: number;
  g: number;
  b: number;
}

export function getThermalColor(val: number, min: number = 4.0, max: number = 31.0): RGB {
  if (isNaN(val) || val === null) {
    // Land mask / subterranean
    return { r: 50, g: 1, b: 27 };
  }

  // Normalize to [0, 1]
  const t = Math.max(0, Math.min(1, (val - min) / Math.max(1e-4, max - min)));

  // Thermal Gradient strictly from Deep Magenta to Alabaster:
  // 0.0: Abyss (#38001d - 56, 0, 29)
  // 0.3: Deep water (#890149 - 137, 1, 73) [Deep Magenta]
  // 0.6: Thermocline (#b82d77 - 184, 45, 119)
  // 0.85: Upper ocean (#e096bf - 224, 150, 191)
  // 1.0: Warm surface (#F3F3E7 - 243, 243, 231) [Alabaster]

  let r = 0, g = 0, b = 0;

  if (t < 0.3) {
    const s = t / 0.3;
    r = 56 + s * (137 - 56);
    g = 0 + s * (1 - 0);
    b = 29 + s * (73 - 29);
  } else if (t < 0.6) {
    const s = (t - 0.3) / 0.3;
    r = 137 + s * (184 - 137);
    g = 1 + s * (45 - 1);
    b = 73 + s * (119 - 73);
  } else if (t < 0.85) {
    const s = (t - 0.6) / 0.25;
    r = 184 + s * (224 - 184);
    g = 45 + s * (150 - 45);
    b = 119 + s * (191 - 119);
  } else {
    const s = (t - 0.85) / 0.15;
    r = 224 + s * (243 - 224);
    g = 150 + s * (243 - 150);
    b = 191 + s * (231 - 191);
  }

  return { r: Math.round(r), g: Math.round(g), b: Math.round(b) };
}

export function getErrorColor(val: number, maxAbsErr: number = 1.5): RGB {
  if (isNaN(val) || val === null) {
    return { r: 50, g: 1, b: 27 };
  }
  // Diverging colormap: Deep Magenta (-1.5) -> Alabaster Neutral (0.0) -> Saturated Magenta (+1.5)
  const norm = Math.max(-1, Math.min(1, val / maxAbsErr));
  if (norm < 0) {
    const s = -norm;
    return {
      r: Math.round(243 - s * (243 - 137)),
      g: Math.round(243 - s * (243 - 1)),
      b: Math.round(231 - s * (231 - 73)),
    };
  } else {
    const s = norm;
    return {
      r: Math.round(243 - s * (243 - 215)),
      g: Math.round(243 - s * (243 - 40)),
      b: Math.round(231 - s * (231 - 130)),
    };
  }
}
