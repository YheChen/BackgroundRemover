/**
 * Stage 2 — coarse segmentation via onnxruntime-web.
 *
 * Runs BiRefNet_lite (MIT, Swin-Tiny, 44.4M params). The general model's
 * fp32 graph is 857 MB, which nobody is downloading into a tab; lite is
 * 189 MB and agrees with it to IoU 0.9843 on our test image.
 *
 * Facts carried over from the Python side, each of which cost a debugging
 * cycle there — do not "simplify" them away:
 *   - preprocessing is 1024x1024 bilinear + ImageNet normalisation;
 *   - the graph has ONE output carrying raw LOGITS, not probabilities, so
 *     the sigmoid below is required.
 */

import * as ort from "onnxruntime-web";
import { plane, type Plane, type Rgb } from "../types";

const SIZE = 1024;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export type Backend = "webgpu" | "wasm";

let session: ort.InferenceSession | null = null;
let activeBackend: Backend | null = null;

export function backend(): Backend | null {
  return activeBackend;
}

export async function load(
  modelUrl: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<Backend> {
  if (session) return activeBackend!;

  // Resolve against the DOCUMENT, not this module. A bare "./ort/" is
  // resolved relative to wherever the bundler put this file — in Vite dev
  // that is /node_modules/.vite/deps/, which 404s. document.baseURI also
  // keeps the app working when deployed under a sub-path.
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
  // Multi-threaded WASM needs SharedArrayBuffer, which needs the page to be
  // cross-origin isolated (COOP + COEP). Asking for threads without it makes
  // ORT fail to initialise rather than quietly degrade, so check first.
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;

  const bytes = await fetchWithProgress(modelUrl, onProgress);

  // WebGPU first; it is roughly 20x multi-threaded WASM on this workload.
  // Some graphs fall back to CPU for most nodes, so failure here is normal
  // and not worth surfacing to the user beyond the badge in the UI.
  for (const ep of ["webgpu", "wasm"] as const) {
    try {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      activeBackend = ep;
      return ep;
    } catch (err) {
      if (ep === "wasm") throw err;
      console.warn(`WebGPU unavailable, falling back to WASM:`, err);
    }
  }
  throw new Error("no execution provider available");
}

export async function probabilityMap(image: Rgb): Promise<Plane> {
  if (!session) throw new Error("call load() before probabilityMap()");

  const input = preprocess(image);
  const feeds: Record<string, ort.Tensor> = {
    [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]),
  };
  const output = await session.run(feeds);
  const logits = output[session.outputNames[0]].data as Float32Array;

  const small = plane(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const z = Math.max(-60, Math.min(60, logits[i]));
    small.data[i] = 1 / (1 + Math.exp(-z));
  }
  return resizeBilinear(small, image.width, image.height);
}

function preprocess(image: Rgb): Float32Array {
  const resized = resizeRgbBilinear(image, SIZE, SIZE);
  const out = new Float32Array(3 * SIZE * SIZE);
  const planeSize = SIZE * SIZE;
  for (let i = 0; i < planeSize; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * planeSize + i] = (resized.data[i * 3 + c] - MEAN[c]) / STD[c];
    }
  }
  return out;
}

const MODEL_CACHE = "bgremover-model-v1";

async function fetchWithProgress(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  // Explicit Cache API rather than the HTTP cache: browsers evict large
  // opportunistic responses aggressively, and re-downloading 159 MB is the
  // difference between "instant" and "broken" on a second visit.
  let cache: Cache | null = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) return hit.arrayBuffer();
  } catch {
    // Caches are unavailable in some contexts (private mode, no HTTPS).
    // Not fatal — fall through to a plain fetch.
  }

  const res = await fetch(url);
  if (res.ok && cache) {
    try {
      await cache.put(url, res.clone());
    } catch {
      // Quota exceeded. The download still succeeds, it just won't persist.
    }
  }
  if (!res.ok) throw new Error(`model fetch failed: ${res.status} ${res.statusText}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !total || !onProgress) return res.arrayBuffer();

  // A 189 MB download with no progress reads as "broken". Report it.
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf.buffer;
}

function resizeRgbBilinear(src: Rgb, w: number, h: number): Rgb {
  const dst: Rgb = { data: new Float32Array(w * h * 3), width: w, height: h };
  sample(src.width, src.height, w, h, (di, sa, sb, sc, sd, fx, fy) => {
    for (let c = 0; c < 3; c++) {
      const top = src.data[sa * 3 + c] * (1 - fx) + src.data[sb * 3 + c] * fx;
      const bot = src.data[sc * 3 + c] * (1 - fx) + src.data[sd * 3 + c] * fx;
      dst.data[di * 3 + c] = top * (1 - fy) + bot * fy;
    }
  });
  return dst;
}

export function resizeBilinear(src: Plane, w: number, h: number): Plane {
  if (src.width === w && src.height === h) return src;
  const dst = plane(w, h);
  sample(src.width, src.height, w, h, (di, sa, sb, sc, sd, fx, fy) => {
    const top = src.data[sa] * (1 - fx) + src.data[sb] * fx;
    const bot = src.data[sc] * (1 - fx) + src.data[sd] * fx;
    dst.data[di] = top * (1 - fy) + bot * fy;
  });
  return dst;
}

/** Shared bilinear address arithmetic for the two resamplers above. */
function sample(
  sw: number, sh: number, dw: number, dh: number,
  emit: (di: number, a: number, b: number, c: number, d: number, fx: number, fy: number) => void,
): void {
  const rx = sw / dw;
  const ry = sh / dh;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.max(0, (y + 0.5) * ry - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.max(0, (x + 0.5) * rx - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const fx = sx - x0;
      emit(y * dw + x, y0 * sw + x0, y0 * sw + x1, y1 * sw + x0, y1 * sw + x1, fx, fy);
    }
  }
}
