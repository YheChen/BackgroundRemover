/**
 * Stage 4 — closed-form alpha matting (Levin et al.), in the browser.
 *
 * The Python side hands this to pymatting, which builds the matting
 * Laplacian as a sparse matrix and factorises a preconditioner. Neither is
 * practical here: for a 1024x1024 image the Laplacian has ~81M non-zeros.
 *
 * Two things make it tractable:
 *
 *   band-only    We solve only for the unknown band, which stage 3 keeps at
 *                roughly 8% of pixels. Known pixels move to the right-hand
 *                side. A 1M-pixel image becomes an ~84k-unknown system.
 *   matrix-free  L is never materialised. Conjugate gradient only needs the
 *                product L@x, and that can be evaluated by sweeping 3x3
 *                windows and accumulating, at O(1) memory in the matrix.
 *
 * The per-window algebra. For a 3x3 window k with mean colour mu_k and
 * covariance Sigma_k, the Laplacian entry is
 *
 *     L[i,j] = sum over windows containing both i and j of
 *              ( delta_ij - (1/9)(1 + d_i^T M d_j) )
 *
 * with d_i = I_i - mu_k and M = (Sigma_k + eps/9 I)^-1. Expanding the
 * product for one window collapses the double sum:
 *
 *     (L_k x)_i = x_i - (1/9)( S + d_i . (M v) )
 *     where S = sum_j x_j  and  v = sum_j d_j x_j
 *
 * so a window costs one pass to build S and v, one 3x3 matvec, and one pass
 * to scatter — not 81 multiply-adds.
 *
 * Windows that contain no unknown pixel are skipped: they cannot contribute
 * to an unknown row, and those are the only rows we keep.
 */

import { TRIMAP_FG, TRIMAP_UNKNOWN, type Plane, type Rgb } from "../types";

const WIN = 9; // pixels in a 3x3 window

export interface MatteOptions {
  /** Covariance regulariser. Larger = smoother, less willing to cut. */
  epsilon?: number;
  maxIterations?: number;
  /** Stop when the residual norm falls below this fraction of the initial. */
  tolerance?: number;
  onProgress?: (fraction: number) => void;
}

/** Precomputed per-window geometry. Built once, reused every CG iteration. */
interface Windows {
  /** WIN pixel indices per window. */
  idx: Int32Array;
  /** WIN * 3 centred colours per window. */
  d: Float32Array;
  /** 6 upper-triangular entries of M per window (M is symmetric). */
  m: Float32Array;
  count: number;
}

export function solve(
  image: Rgb,
  trimap: Uint8Array,
  opts: MatteOptions = {},
): Plane {
  const { epsilon = 1e-7, maxIterations = 200, tolerance = 1e-5, onProgress } = opts;
  const w = image.width;
  const h = image.height;
  const n = w * h;
  if (trimap.length !== n) throw new Error("image and trimap must match");

  // Map each unknown pixel to a slot in the reduced system.
  const slot = new Int32Array(n).fill(-1);
  let nUnknown = 0;
  for (let i = 0; i < n; i++) if (trimap[i] === TRIMAP_UNKNOWN) slot[i] = nUnknown++;

  // known alpha, zero on unknowns — this is what moves to the RHS.
  const known = new Float32Array(n);
  for (let i = 0; i < n; i++) known[i] = trimap[i] === TRIMAP_FG ? 1 : 0;

  if (nUnknown === 0) return { data: known, width: w, height: h };

  const win = buildWindows(image, slot, epsilon);

  // Solving L_uu x = -(L @ known)|_u  gives the unknown alphas directly.
  const rhs = new Float32Array(nUnknown);
  {
    const full = applyL(win, known, n);
    for (let i = 0; i < n; i++) {
      const s = slot[i];
      if (s >= 0) rhs[s] = -full[i];
    }
  }

  const diag = diagonal(win, slot, nUnknown);
  const x = conjugateGradient(
    (p, out) => operator(win, slot, n, p, out),
    rhs,
    diag,
    maxIterations,
    tolerance,
    onProgress,
  );

  const alpha = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = slot[i];
    const v = s >= 0 ? x[s] : known[i];
    alpha[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return { data: alpha, width: w, height: h };
}

/** Precompute centred colours and inverse regularised covariance per window. */
function buildWindows(image: Rgb, slot: Int32Array, epsilon: number): Windows {
  const w = image.width;
  const h = image.height;
  const img = image.data;

  // Only windows touching the band can affect an unknown row.
  const active: number[] = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let touches = false;
      for (let dy = -1; dy <= 1 && !touches; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (slot[(y + dy) * w + (x + dx)] >= 0) { touches = true; break; }
        }
      }
      if (touches) active.push(y * w + x);
    }
  }

  const count = active.length;
  const idx = new Int32Array(count * WIN);
  const d = new Float32Array(count * WIN * 3);
  const m = new Float32Array(count * 6);

  const cols = new Float32Array(WIN * 3);
  for (let k = 0; k < count; k++) {
    const centre = active[k];
    const cy = (centre / w) | 0;
    const cx = centre - cy * w;

    let p = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const pi = (cy + dy) * w + (cx + dx);
        idx[k * WIN + p] = pi;
        cols[p * 3] = img[pi * 3];
        cols[p * 3 + 1] = img[pi * 3 + 1];
        cols[p * 3 + 2] = img[pi * 3 + 2];
        p++;
      }
    }

    let mr = 0, mg = 0, mb = 0;
    for (let i = 0; i < WIN; i++) { mr += cols[i * 3]; mg += cols[i * 3 + 1]; mb += cols[i * 3 + 2]; }
    mr /= WIN; mg /= WIN; mb /= WIN;

    let c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
    for (let i = 0; i < WIN; i++) {
      const r = cols[i * 3] - mr;
      const g = cols[i * 3 + 1] - mg;
      const b = cols[i * 3 + 2] - mb;
      d[(k * WIN + i) * 3] = r;
      d[(k * WIN + i) * 3 + 1] = g;
      d[(k * WIN + i) * 3 + 2] = b;
      c00 += r * r; c01 += r * g; c02 += r * b;
      c11 += g * g; c12 += g * b; c22 += b * b;
    }
    const e = epsilon / WIN;
    c00 = c00 / WIN + e; c01 /= WIN; c02 /= WIN;
    c11 = c11 / WIN + e; c12 /= WIN; c22 = c22 / WIN + e;

    invertSym3(c00, c01, c02, c11, c12, c22, m, k * 6);
  }
  return { idx, d, m, count };
}

/** y = L @ x over active windows. `x` and the result are full-image length. */
function applyL(win: Windows, x: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  accumulateL(win, x, out);
  return out;
}

function accumulateL(win: Windows, x: Float32Array, out: Float32Array): void {
  const { idx, d, m, count } = win;
  for (let k = 0; k < count; k++) {
    const base = k * WIN;
    let S = 0, vr = 0, vg = 0, vb = 0;
    for (let i = 0; i < WIN; i++) {
      const xi = x[idx[base + i]];
      if (xi === 0) continue;
      S += xi;
      const o = (base + i) * 3;
      vr += d[o] * xi; vg += d[o + 1] * xi; vb += d[o + 2] * xi;
    }
    const mo = k * 6;
    // M is symmetric: [m0 m1 m2; m1 m3 m4; m2 m4 m5]
    const ur = m[mo] * vr + m[mo + 1] * vg + m[mo + 2] * vb;
    const ug = m[mo + 1] * vr + m[mo + 3] * vg + m[mo + 4] * vb;
    const ub = m[mo + 2] * vr + m[mo + 4] * vg + m[mo + 5] * vb;

    for (let i = 0; i < WIN; i++) {
      const pi = idx[base + i];
      const o = (base + i) * 3;
      const dot = d[o] * ur + d[o + 1] * ug + d[o + 2] * ub;
      out[pi] += x[pi] - (S + dot) / WIN;
    }
  }
}

/** The reduced operator: extend to full size, apply L, restrict to unknowns. */
function operator(
  win: Windows,
  slot: Int32Array,
  n: number,
  p: Float32Array,
  out: Float32Array,
): void {
  const full = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = slot[i];
    if (s >= 0) full[i] = p[s];
  }
  const acc = new Float32Array(n);
  accumulateL(win, full, acc);
  for (let i = 0; i < n; i++) {
    const s = slot[i];
    if (s >= 0) out[s] = acc[i];
  }
}

/** Diagonal of L on unknown rows, for Jacobi preconditioning. */
function diagonal(win: Windows, slot: Int32Array, nUnknown: number): Float32Array {
  const diag = new Float32Array(nUnknown);
  const { idx, d, m, count } = win;
  for (let k = 0; k < count; k++) {
    const base = k * WIN;
    const mo = k * 6;
    for (let i = 0; i < WIN; i++) {
      const s = slot[idx[base + i]];
      if (s < 0) continue;
      const o = (base + i) * 3;
      const dr = d[o], dg = d[o + 1], db = d[o + 2];
      const ur = m[mo] * dr + m[mo + 1] * dg + m[mo + 2] * db;
      const ug = m[mo + 1] * dr + m[mo + 3] * dg + m[mo + 4] * db;
      const ub = m[mo + 2] * dr + m[mo + 4] * dg + m[mo + 5] * db;
      diag[s] += 1 - (1 + dr * ur + dg * ug + db * ub) / WIN;
    }
  }
  for (let i = 0; i < nUnknown; i++) if (diag[i] <= 1e-12) diag[i] = 1;
  return diag;
}

function conjugateGradient(
  apply: (p: Float32Array, out: Float32Array) => void,
  b: Float32Array,
  diag: Float32Array,
  maxIterations: number,
  tolerance: number,
  onProgress?: (f: number) => void,
): Float32Array {
  const n = b.length;
  const x = new Float32Array(n);
  const r = Float32Array.from(b);
  const z = new Float32Array(n);
  const p = new Float32Array(n);
  const ap = new Float32Array(n);

  for (let i = 0; i < n; i++) { z[i] = r[i] / diag[i]; p[i] = z[i]; }
  let rz = dot(r, z);
  const bNorm = Math.sqrt(dot(b, b)) || 1;

  for (let iter = 0; iter < maxIterations; iter++) {
    apply(p, ap);
    const pAp = dot(p, ap);
    if (!isFinite(pAp) || pAp === 0) break;
    const alpha = rz / pAp;
    for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }

    const rNorm = Math.sqrt(dot(r, r));
    if (onProgress && iter % 8 === 0) onProgress(Math.min(1, iter / maxIterations));
    if (rNorm / bNorm < tolerance) break;

    for (let i = 0; i < n; i++) z[i] = r[i] / diag[i];
    const rzNew = dot(r, z);
    const beta = rzNew / rz;
    rz = rzNew;
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
  }
  onProgress?.(1);
  return x;
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Inverse of a symmetric 3x3, written as 6 upper-triangular entries. */
function invertSym3(
  a: number, b: number, c: number, dd: number, e: number, f: number,
  out: Float32Array, o: number,
): void {
  // [a b c; b d e; c e f]
  const A = dd * f - e * e;
  const B = c * e - b * f;
  const C = b * e - c * dd;
  let det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-20) det = det < 0 ? -1e-20 : 1e-20;
  const inv = 1 / det;
  out[o] = A * inv;
  out[o + 1] = B * inv;
  out[o + 2] = C * inv;
  out[o + 3] = (a * f - c * c) * inv;
  out[o + 4] = (c * b - a * e) * inv;
  out[o + 5] = (a * dd - b * b) * inv;
}
