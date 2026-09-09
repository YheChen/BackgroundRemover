/**
 * The edge tool's engine — an edge-cost field plus live-wire shortest paths.
 *
 * Tracing a subject by hand is mostly the same work over and over: the user
 * knows roughly where the boundary is and the image knows exactly. So instead
 * of asking for a point every few pixels, we make traversal cheap along
 * strong gradients and expensive across flat regions, then let Dijkstra find
 * the cheapest route from the last anchor to the cursor. The path snaps onto
 * the real boundary between clicks. This is Mortensen & Barrett's live wire,
 * reduced to gradient magnitude — direction and Laplacian zero-crossing terms
 * buy little on photographs and cost a pass each.
 *
 * The important structural point: one Dijkstra from the anchor yields the
 * cheapest route to *every* pixel in the window, so the live preview that
 * follows the cursor is a free backtrace through the predecessor map. Running
 * a search per pointer-move instead would be ~100x the work for the same
 * answer.
 */

import type { Rgb } from "../types";

/** Half-width of the searched window, in px. See `seed`. */
export const WINDOW_RADIUS = 256;

/**
 * Added to every step so that a path prefers the shorter of two equally
 * strong edges. Without it the wire happily detours along any contour it can
 * reach for free, which reads as the tool ignoring the cursor.
 */
const LENGTH_BIAS = 0.06;

const SQRT2 = Math.SQRT2;

export class EdgeField {
  readonly width: number;
  readonly height: number;
  /** Gradient magnitude, normalised to [0,1] against a high percentile. */
  readonly magnitude: Float32Array;
  /** Traversal cost per pixel: near 0 on a strong edge, 1 on flat ground. */
  readonly cost: Float32Array;

  constructor(image: Rgb) {
    const w = (this.width = image.width);
    const h = (this.height = image.height);
    const n = w * h;

    // Luminance, lightly blurred. Sobel on raw pixels turns sensor noise into
    // attractive-looking edges and the wire then follows the noise.
    const lum = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      lum[i] =
        0.2126 * image.data[i * 3] +
        0.7152 * image.data[i * 3 + 1] +
        0.0722 * image.data[i * 3 + 2];
    }
    const blur = gaussian3(lum, w, h);

    const mag = new Float32Array(n);
    let peak = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const xm = x > 0 ? x - 1 : 0;
        const xp = x < w - 1 ? x + 1 : w - 1;
        const ym = y > 0 ? y - 1 : 0;
        const yp = y < h - 1 ? y + 1 : h - 1;
        const tl = blur[ym * w + xm], tc = blur[ym * w + x], tr = blur[ym * w + xp];
        const ml = blur[y * w + xm], mr = blur[y * w + xp];
        const bl = blur[yp * w + xm], bc = blur[yp * w + x], br = blur[yp * w + xp];
        const gx = tr + 2 * mr + br - tl - 2 * ml - bl;
        const gy = bl + 2 * bc + br - tl - 2 * tc - tr;
        const m = Math.hypot(gx, gy);
        mag[y * w + x] = m;
        if (m > peak) peak = m;
      }
    }

    // Normalise against the 99th percentile, not the maximum: one specular
    // highlight sets the maximum and flattens every real boundary to nothing.
    const scale = percentile(mag, peak, 0.99) || 1;
    this.magnitude = mag;
    this.cost = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const m = Math.min(1, mag[i] / scale);
      mag[i] = m;
      this.cost[i] = 1 - m;
    }
  }

  /**
   * Pull a point onto the strongest edge nearby, so a click that lands a
   * couple of pixels inside the subject still starts the wire on the boundary.
   */
  snap(x: number, y: number, radius = 4): [number, number] {
    const cx = Math.round(x);
    const cy = Math.round(y);
    let bestX = cx;
    let bestY = cy;
    let best = -1;
    for (let dy = -radius; dy <= radius; dy++) {
      const yy = cy + dy;
      if (yy < 0 || yy >= this.height) continue;
      for (let dx = -radius; dx <= radius; dx++) {
        const xx = cx + dx;
        if (xx < 0 || xx >= this.width) continue;
        // Tie-break towards the click: the user's aim carries information.
        const d = Math.hypot(dx, dy);
        const score = this.magnitude[yy * this.width + xx] - 0.02 * d;
        if (score > best) {
          best = score;
          bestX = xx;
          bestY = yy;
        }
      }
    }
    return [bestX, bestY];
  }
}

/** A completed Dijkstra expansion from one seed point. */
export interface Wire {
  seedX: number;
  seedY: number;
  /** Window bounds in image coords, [x0, y0, x1, y1) exclusive. */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  pred: Int32Array;
}

/**
 * Expand the cheapest-path tree from (sx, sy) over a window of `radius` px.
 *
 * Bounded on purpose: the cost of a full-image expansion is paid on every
 * click, and a wire that runs more than a couple of hundred pixels without a
 * user anchor has usually gone somewhere the user did not intend anyway.
 */
export function seed(field: EdgeField, sx: number, sy: number, radius = WINDOW_RADIUS): Wire {
  const x0 = Math.max(0, Math.round(sx) - radius);
  const y0 = Math.max(0, Math.round(sy) - radius);
  const x1 = Math.min(field.width, Math.round(sx) + radius + 1);
  const y1 = Math.min(field.height, Math.round(sy) + radius + 1);
  const ww = x1 - x0;
  const wh = y1 - y0;
  const n = ww * wh;

  const dist = new Float32Array(n).fill(Infinity);
  const pred = new Int32Array(n).fill(-1);
  const done = new Uint8Array(n);
  const heap = new MinHeap(n);

  const start = (Math.round(sy) - y0) * ww + (Math.round(sx) - x0);
  dist[start] = 0;
  heap.push(start, 0);

  while (heap.size > 0) {
    const u = heap.pop();
    if (done[u]) continue;
    done[u] = 1;
    const ux = u % ww;
    const uy = (u - ux) / ww;
    const du = dist[u];

    for (let dy = -1; dy <= 1; dy++) {
      const vy = uy + dy;
      if (vy < 0 || vy >= wh) continue;
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const vx = ux + dx;
        if (vx < 0 || vx >= ww) continue;
        const v = vy * ww + vx;
        if (done[v]) continue;
        const step = dx !== 0 && dy !== 0 ? SQRT2 : 1;
        const c = (field.cost[(vy + y0) * field.width + (vx + x0)] + LENGTH_BIAS) * step;
        const alt = du + c;
        if (alt < dist[v]) {
          dist[v] = alt;
          pred[v] = u;
          heap.push(v, alt);
        }
      }
    }
  }

  return { seedX: Math.round(sx), seedY: Math.round(sy), x0, y0, x1, y1, pred };
}

/** True while the cursor is inside the window this wire was expanded over. */
export function covers(wire: Wire, x: number, y: number, margin = 8): boolean {
  return (
    x >= wire.x0 + margin &&
    y >= wire.y0 + margin &&
    x < wire.x1 - margin &&
    y < wire.y1 - margin
  );
}

/**
 * Backtrace the cheapest path from the seed to (x, y) as an xy-pair polyline,
 * seed first. Returns a straight two-point line if the target is unreachable,
 * which only happens outside the window.
 */
export function pathTo(wire: Wire, x: number, y: number): number[] {
  const ww = wire.x1 - wire.x0;
  const wh = wire.y1 - wire.y0;
  const tx = Math.min(ww - 1, Math.max(0, Math.round(x) - wire.x0));
  const ty = Math.min(wh - 1, Math.max(0, Math.round(y) - wire.y0));

  const rev: number[] = [];
  let cur = ty * ww + tx;
  let guard = ww * wh + 4;
  while (cur >= 0 && guard-- > 0) {
    rev.push((cur % ww) + wire.x0, ((cur - (cur % ww)) / ww) + wire.y0);
    if (cur === (wire.seedY - wire.y0) * ww + (wire.seedX - wire.x0)) break;
    cur = wire.pred[cur];
  }
  if (cur < 0) return [wire.seedX, wire.seedY, Math.round(x), Math.round(y)];

  const out: number[] = [];
  for (let i = rev.length - 2; i >= 0; i -= 2) out.push(rev[i], rev[i + 1]);
  return out;
}

/** Separable 3-tap [1,2,1] blur. */
function gaussian3(src: Float32Array, w: number, h: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const l = src[row + (x > 0 ? x - 1 : 0)];
      const r = src[row + (x < w - 1 ? x + 1 : w - 1)];
      tmp[row + x] = (l + 2 * src[row + x] + r) / 4;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const u = tmp[(y > 0 ? y - 1 : 0) * w + x];
      const d = tmp[(y < h - 1 ? y + 1 : h - 1) * w + x];
      dst[y * w + x] = (u + 2 * tmp[y * w + x] + d) / 4;
    }
  }
  return dst;
}

/** Percentile of `values` via a 512-bin histogram over [0, peak]. */
function percentile(values: Float32Array, peak: number, q: number): number {
  if (peak <= 0) return 0;
  const BINS = 512;
  const hist = new Int32Array(BINS);
  for (let i = 0; i < values.length; i++) {
    const b = Math.min(BINS - 1, Math.floor((values[i] / peak) * BINS));
    hist[b]++;
  }
  const target = values.length * q;
  let acc = 0;
  for (let b = 0; b < BINS; b++) {
    acc += hist[b];
    if (acc >= target) return ((b + 1) / BINS) * peak;
  }
  return peak;
}

/**
 * Binary min-heap over node indices keyed by f32 distance.
 *
 * Lazy deletion — a node can be pushed several times and stale entries are
 * skipped via the caller's `done` flags. Cheaper than a decrease-key, and the
 * heap is bounded by the edge count either way.
 */
class MinHeap {
  private node: Int32Array;
  private key: Float32Array;
  size = 0;

  constructor(capacity: number) {
    // 8-connected, so each node can be improved a few times; grow if needed.
    const cap = Math.max(16, capacity * 2);
    this.node = new Int32Array(cap);
    this.key = new Float32Array(cap);
  }

  push(n: number, k: number): void {
    if (this.size === this.node.length) this.grow();
    let i = this.size++;
    this.node[i] = n;
    this.key[i] = k;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.key[parent] <= this.key[i]) break;
      this.swap(parent, i);
      i = parent;
    }
  }

  pop(): number {
    const top = this.node[0];
    this.size--;
    if (this.size > 0) {
      this.node[0] = this.node[this.size];
      this.key[0] = this.key[this.size];
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < this.size && this.key[l] < this.key[m]) m = l;
        if (r < this.size && this.key[r] < this.key[m]) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }

  private swap(a: number, b: number): void {
    const n = this.node[a];
    this.node[a] = this.node[b];
    this.node[b] = n;
    const k = this.key[a];
    this.key[a] = this.key[b];
    this.key[b] = k;
  }

  private grow(): void {
    const node = new Int32Array(this.node.length * 2);
    const key = new Float32Array(this.key.length * 2);
    node.set(this.node);
    key.set(this.key);
    this.node = node;
    this.key = key;
  }
}
