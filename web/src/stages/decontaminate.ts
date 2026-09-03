/**
 * Stage 5 — foreground colour estimation ("decontamination").
 *
 * Faithful port of Fast Multi-Level Foreground Estimation (Germer et al.,
 * 2020) as implemented in pymatting, so the browser and Python paths agree.
 *
 * Why it exists: a semi-transparent pixel's observed colour is a blend of
 * foreground and whatever was behind it. Cut a person out of a green room
 * and their hair edges stay green. So the observed colour must be REPLACED
 * with an estimate of the uncontaminated foreground wherever alpha < 1.
 * Photoshop calls this "Decontaminate Colors".
 *
 * Method: solve a 2x2 system per pixel per channel
 *
 *     [a00 a01] [F]   [b0]
 *     [a01 a11] [B] = [b1]
 *
 * where the diagonal picks up a regularisation term from the four
 * neighbours, weighted by the local alpha gradient so that smoothing does
 * not cross a real edge. Solved coarse-to-fine over an image pyramid, which
 * is what makes it fast enough to run unconditionally.
 */

import { rgb, type Plane, type Rgb } from "../types";

export interface DecontaminateOptions {
  regularization?: number;
  nSmallIterations?: number;
  nBigIterations?: number;
  smallSize?: number;
  gradientWeight?: number;
}

export function estimate(
  image: Rgb,
  alpha: Plane,
  opts: DecontaminateOptions = {},
): Rgb {
  const {
    regularization = 1e-5,
    nSmallIterations = 10,
    nBigIterations = 2,
    smallSize = 32,
    gradientWeight = 1.0,
  } = opts;

  const w0 = image.width;
  const h0 = image.height;
  if (alpha.width !== w0 || alpha.height !== h0) {
    throw new Error("image and alpha must have the same dimensions");
  }

  // Seed both planes with the mean confident-foreground / -background colour.
  const fMean = new Float32Array(3);
  const bMean = new Float32Array(3);
  let fCount = 0;
  let bCount = 0;
  for (let i = 0; i < w0 * h0; i++) {
    const a = alpha.data[i];
    if (a > 0.9) {
      for (let c = 0; c < 3; c++) fMean[c] += image.data[i * 3 + c];
      fCount++;
    } else if (a < 0.1) {
      for (let c = 0; c < 3; c++) bMean[c] += image.data[i * 3 + c];
      bCount++;
    }
  }
  for (let c = 0; c < 3; c++) {
    fMean[c] /= fCount + 1e-5;
    bMean[c] /= bCount + 1e-5;
  }

  let fPrev = rgb(1, 1);
  let bPrev = rgb(1, 1);
  for (let c = 0; c < 3; c++) {
    fPrev.data[c] = fMean[c];
    bPrev.data[c] = bMean[c];
  }

  const nLevels = Math.ceil(Math.log2(Math.max(w0, h0)));
  const dx = [-1, 1, 0, 0];
  const dy = [0, 0, -1, 1];

  for (let level = 0; level <= nLevels; level++) {
    const w = Math.round(Math.pow(w0, level / nLevels));
    const h = Math.round(Math.pow(h0, level / nLevels));

    const img = resizeRgbNearest(image, w, h);
    const a = resizePlaneNearest(alpha, w, h);
    const F = resizeRgbNearest(fPrev, w, h);
    const B = resizeRgbNearest(bPrev, w, h);

    const nIter = w <= smallSize && h <= smallSize ? nSmallIterations : nBigIterations;

    for (let iter = 0; iter < nIter; iter++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = y * w + x;
          const a0 = a.data[idx];
          const a1 = 1.0 - a0;

          let a00 = a0 * a0;
          const a01 = a0 * a1;
          let a11 = a1 * a1;

          const b0 = [0, 0, 0];
          const b1 = [0, 0, 0];
          for (let c = 0; c < 3; c++) {
            const v = img.data[idx * 3 + c];
            b0[c] = a0 * v;
            b1[c] = a1 * v;
          }

          for (let d = 0; d < 4; d++) {
            const x2 = Math.max(0, Math.min(w - 1, x + dx[d]));
            const y2 = Math.max(0, Math.min(h - 1, y + dy[d]));
            const n = y2 * w + x2;

            // Weighting by the alpha gradient stops the regulariser from
            // smoothing colour across a genuine foreground/background edge.
            const da = regularization + gradientWeight * Math.abs(a0 - a.data[n]);
            a00 += da;
            a11 += da;
            for (let c = 0; c < 3; c++) {
              b0[c] += da * F.data[n * 3 + c];
              b1[c] += da * B.data[n * 3 + c];
            }
          }

          const invDet = 1.0 / (a00 * a11 - a01 * a01);
          const i00 = invDet * a11;
          const i01 = invDet * -a01;
          const i11 = invDet * a00;

          for (let c = 0; c < 3; c++) {
            const fc = i00 * b0[c] + i01 * b1[c];
            const bc = i01 * b0[c] + i11 * b1[c];
            F.data[idx * 3 + c] = clamp01(fc);
            B.data[idx * 3 + c] = clamp01(bc);
          }
        }
      }
    }
    fPrev = F;
    bPrev = B;
  }
  return fPrev;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function resizeRgbNearest(src: Rgb, w: number, h: number): Rgb {
  const dst = rgb(w, h);
  for (let y = 0; y < h; y++) {
    const ys = Math.max(0, Math.min(src.height - 1, Math.floor((y * src.height) / h)));
    for (let x = 0; x < w; x++) {
      const xs = Math.max(0, Math.min(src.width - 1, Math.floor((x * src.width) / w)));
      const s = (ys * src.width + xs) * 3;
      const d = (y * w + x) * 3;
      dst.data[d] = src.data[s];
      dst.data[d + 1] = src.data[s + 1];
      dst.data[d + 2] = src.data[s + 2];
    }
  }
  return dst;
}

function resizePlaneNearest(src: Plane, w: number, h: number): Plane {
  const dst = { data: new Float32Array(w * h), width: w, height: h };
  for (let y = 0; y < h; y++) {
    const ys = Math.max(0, Math.min(src.height - 1, Math.floor((y * src.height) / h)));
    for (let x = 0; x < w; x++) {
      const xs = Math.max(0, Math.min(src.width - 1, Math.floor((x * src.width) / w)));
      dst.data[y * w + x] = src.data[ys * src.width + xs];
    }
  }
  return dst;
}
