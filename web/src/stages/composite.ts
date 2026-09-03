/**
 * Stage 6 — reprojection and compositing.
 *
 * The matte is solved at working resolution; the source may be much larger.
 * Plain bilinear upsampling throws away the edge detail stages 3-5 just paid
 * for, so alpha is upsampled *guided by* the full-resolution image: a guided
 * filter (He et al.) with luminance as the guide, which snaps the matte back
 * onto real edges.
 */

import { plane, type Plane, type Rgb } from "../types";
import { resizeBilinear } from "./segment";

export function upsampleAlpha(
  alpha: Plane,
  image: Rgb,
  radius = 4,
  eps = 1e-4,
): Plane {
  const w = image.width;
  const h = image.height;
  const coarse = resizeBilinear(alpha, w, h);
  if (alpha.width === w && alpha.height === h) return coarse;

  const guide = plane(w, h);
  for (let i = 0; i < w * h; i++) {
    guide.data[i] =
      0.2126 * image.data[i * 3] + 0.7152 * image.data[i * 3 + 1] + 0.0722 * image.data[i * 3 + 2];
  }

  const meanG = box(guide, radius);
  const meanA = box(coarse, radius);
  const gg = box(mul(guide, guide), radius);
  const ga = box(mul(guide, coarse), radius);

  const a = plane(w, h);
  const b = plane(w, h);
  for (let i = 0; i < w * h; i++) {
    const varG = gg.data[i] - meanG.data[i] * meanG.data[i];
    const covGA = ga.data[i] - meanG.data[i] * meanA.data[i];
    a.data[i] = covGA / (varG + eps);
    b.data[i] = meanA.data[i] - a.data[i] * meanG.data[i];
  }

  const ma = box(a, radius);
  const mb = box(b, radius);
  const out = plane(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = ma.data[i] * guide.data[i] + mb.data[i];
    out.data[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

/** Flatten to RGBA bytes for a canvas. `background` null keeps transparency. */
export function toImageData(
  colour: Rgb,
  alpha: Plane,
  background: [number, number, number] | null,
): ImageData {
  const w = colour.width;
  const h = colour.height;
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const a = alpha.data[i];
    for (let c = 0; c < 3; c++) {
      const fg = colour.data[i * 3 + c] * 255;
      out[i * 4 + c] = background ? fg * a + background[c] * (1 - a) : fg;
    }
    out[i * 4 + 3] = background ? 255 : a * 255;
  }
  return new ImageData(out, w, h);
}

/** Tight box around the subject as [left, top, right, bottom), exclusive. */
export function boundingBox(alpha: Plane, threshold = 0.03): [number, number, number, number] {
  const { data, width: w, height: h } = alpha;
  let left = w, top = h, right = 0, bottom = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[y * w + x] > threshold) {
        if (x < left) left = x;
        if (x >= right) right = x + 1;
        if (y < top) top = y;
        if (y >= bottom) bottom = y + 1;
      }
    }
  }
  // An empty matte returns the full frame, so callers never special-case it.
  if (right <= left || bottom <= top) return [0, 0, w, h];
  return [left, top, right, bottom];
}

function mul(a: Plane, b: Plane): Plane {
  const out = plane(a.width, a.height);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] * b.data[i];
  return out;
}

/** Separable box mean over a (2r+1) square, via prefix sums. */
function box(src: Plane, r: number): Plane {
  const { width: w, height: h } = src;
  const tmp = plane(w, h);
  const dst = plane(w, h);

  for (let y = 0; y < h; y++) {
    const row = y * w;
    let acc = 0;
    for (let x = 0; x <= Math.min(r, w - 1); x++) acc += src.data[row + x];
    for (let x = 0; x < w; x++) {
      const lo = x - r - 1;
      const hi = x + r;
      if (hi < w) acc += src.data[row + hi];
      if (lo >= 0) acc -= src.data[row + lo];
      const count = Math.min(w - 1, x + r) - Math.max(0, x - r) + 1;
      tmp.data[row + x] = acc / count;
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = 0; y <= Math.min(r, h - 1); y++) acc += tmp.data[y * w + x];
    for (let y = 0; y < h; y++) {
      const lo = y - r - 1;
      const hi = y + r;
      if (hi < h) acc += tmp.data[hi * w + x];
      if (lo >= 0) acc -= tmp.data[lo * w + x];
      const count = Math.min(h - 1, y + r) - Math.max(0, y - r) + 1;
      dst.data[y * w + x] = acc / count;
    }
  }
  return dst;
}
