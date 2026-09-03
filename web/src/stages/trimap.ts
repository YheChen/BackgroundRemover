/**
 * Stage 3 — uncertainty band derivation.
 *
 * Port of the Python stage. Two sources of "unknown", unioned:
 *   a) low model confidence, which is the better signal;
 *   b) a guaranteed minimum band around the boundary, because a confident
 *      model returns a razor-thin ambiguous region and matting needs room.
 *
 * Only the band goes to stage 4, so band width is the main quality/speed
 * dial. It is quoted in px at REFERENCE_EDGE and rescaled — an absolute
 * pixel band is a hairline on a 4000px photo and 12% of a thumbnail.
 */

import { TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN, type Plane } from "../types";

export const REFERENCE_EDGE = 1024;

export interface TrimapOptions {
  fgThreshold?: number;
  bgThreshold?: number;
  /** px at REFERENCE_EDGE; rescaled to the actual image. */
  bandWidth?: number;
}

export function derive(prob: Plane, opts: TrimapOptions = {}): Uint8Array {
  const { fgThreshold = 0.95, bgThreshold = 0.05, bandWidth = 12 } = opts;
  if (!(bgThreshold >= 0 && bgThreshold < fgThreshold && fgThreshold <= 1)) {
    throw new Error(
      `need 0 <= bgThreshold (${bgThreshold}) < fgThreshold (${fgThreshold}) <= 1`,
    );
  }

  const { data, width: w, height: h } = prob;
  const n = w * h;
  const out = new Uint8Array(n);

  const iterations = scaledBand(bandWidth, w, h);

  // Pass 1: confidence.
  const solid = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const p = data[i];
    solid[i] = p >= 0.5 ? 1 : 0;
    out[i] = p >= fgThreshold ? TRIMAP_FG : p <= bgThreshold ? TRIMAP_BG : TRIMAP_UNKNOWN;
  }

  // Pass 2: morphological band, as dilate(solid) AND NOT erode(solid).
  // Both are separable with a square structuring element, so this is two
  // 1-D passes each rather than a full 2-D neighbourhood scan.
  if (iterations > 0) {
    const outer = morph(solid, w, h, iterations, true);
    const inner = morph(solid, w, h, iterations, false);
    for (let i = 0; i < n; i++) {
      if (outer[i] === 1 && inner[i] === 0) out[i] = TRIMAP_UNKNOWN;
    }
    // Confidence must not be overridden by the band for definite pixels that
    // the band happens to miss; re-assert FG where the model was certain.
    for (let i = 0; i < n; i++) {
      if (data[i] >= fgThreshold && out[i] !== TRIMAP_UNKNOWN) out[i] = TRIMAP_FG;
    }
  }
  return out;
}

export function scaledBand(bandWidth: number, w: number, h: number): number {
  if (bandWidth <= 0) return 0;
  return Math.max(1, Math.round((bandWidth * Math.min(w, h)) / REFERENCE_EDGE));
}

export function bandFraction(trimap: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < trimap.length; i++) if (trimap[i] === TRIMAP_UNKNOWN) n++;
  return n / trimap.length;
}

/** Separable square dilation (`dilate`) or erosion, radius `r`. */
function morph(src: Uint8Array, w: number, h: number, r: number, dilate: boolean): Uint8Array {
  const pick = dilate ? Math.max : Math.min;
  const tmp = new Uint8Array(w * h);
  const dst = new Uint8Array(w * h);
  const edge = dilate ? 0 : 1; // outside the image: empty for dilate, solid for erode

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = edge;
      for (let k = -r; k <= r; k++) {
        const xx = x + k;
        const s = xx < 0 || xx >= w ? edge : src[row + xx];
        v = pick(v, s);
      }
      tmp[row + x] = v;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let v = edge;
      for (let k = -r; k <= r; k++) {
        const yy = y + k;
        const s = yy < 0 || yy >= h ? edge : tmp[yy * w + x];
        v = pick(v, s);
      }
      dst[y * w + x] = v;
    }
  }
  return dst;
}
