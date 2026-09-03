/**
 * UI wiring. The pipeline itself lives in pipeline.ts; this file only moves
 * pixels between a file input, a canvas, and a download link.
 */

import { colourPlane, removeBackground, WORKING_PIXELS } from "./pipeline";
import { boundingBox, toImageData, upsampleAlpha } from "./stages/composite";
import * as segment from "./stages/segment";
import type { Cutout, EdgeMode, Rgb } from "./types";

// Where the weights live. Defaults to same-origin for local dev; set
// VITE_MODEL_URL at build time to point at a CDN. On Vercel's Hobby tier the
// per-file upload cap is 100 MB and bandwidth is 100 GB/month, so a 159 MB
// model served from the app's own origin is both impossible and, if it were
// possible, would cap the site at roughly a thousand first-time visitors.
const MODEL_URL = import.meta.env.VITE_MODEL_URL || "./models/birefnet-lite.onnx";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const drop = $<HTMLDivElement>("drop");
const fileInput = $<HTMLInputElement>("file");
const canvas = $<HTMLCanvasElement>("canvas");
const progress = $<HTMLDivElement>("progress");
const pbar = $<HTMLElement>("pbar");
const plabel = $<HTMLElement>("plabel");
const ppct = $<HTMLElement>("ppct");
const work = $<HTMLDivElement>("work");
const errBox = $<HTMLDivElement>("err");
const backendBadge = $<HTMLSpanElement>("backend");
const edgeSel = $<HTMLSelectElement>("edge");
const bgSel = $<HTMLSelectElement>("bg");

let source: Rgb | null = null;
let cutout: Cutout | null = null;
let busy = false;

// --- progress ---------------------------------------------------------------

function showProgress(label: string, fraction: number): void {
  progress.classList.add("show");
  plabel.textContent = label;
  const pct = Math.round(fraction * 100);
  ppct.textContent = `${pct}%`;
  pbar.style.width = `${pct}%`;
}
function hideProgress(): void {
  progress.classList.remove("show");
  pbar.style.width = "0%";
}
function showError(message: string): void {
  errBox.textContent = message;
  errBox.classList.add("show");
  hideProgress();
}
function clearError(): void {
  errBox.classList.remove("show");
}

// --- model ------------------------------------------------------------------

const modelReady = (async () => {
  try {
    // A 189 MB download with no feedback reads as "broken", so report bytes
    // rather than a spinner. The browser cache makes this a one-time cost.
    const ep = await segment.load(MODEL_URL, (loaded, total) => {
      showProgress(
        `Downloading the model — ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`,
        loaded / total,
      );
    });
    backendBadge.textContent = ep === "webgpu" ? "WebGPU" : "WASM (CPU)";
    backendBadge.classList.add("on");
    hideProgress();
  } catch (err) {
    backendBadge.textContent = "unavailable";
    showError(
      `Could not load the model: ${err instanceof Error ? err.message : String(err)}. ` +
        `If you are running from source, the file belongs at web/public/models/birefnet-lite.onnx.`,
    );
    throw err;
  }
})();

// --- input ------------------------------------------------------------------

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) void handleFile(f);
});
for (const type of ["dragenter", "dragover"]) {
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add("over"); });
}
for (const type of ["dragleave", "drop"]) {
  drop.addEventListener(type, () => drop.classList.remove("over"));
}
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = (e as DragEvent).dataTransfer?.files?.[0];
  if (f) void handleFile(f);
});

async function handleFile(file: File): Promise<void> {
  if (busy) return;
  clearError();
  try {
    showProgress("Reading the image", 0.02);
    source = await decode(file);
    await run();
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
  }
}

async function decode(file: File): Promise<Rgb> {
  const bitmap = await createImageBitmap(file);
  // Stages 2-5 run at working resolution; stage 6 reprojects. Matting a
  // 50 MP image directly is a memory problem, not a quality win — and in a
  // tab it is an out-of-memory crash.
  const scale = Math.min(1, Math.sqrt(WORKING_PIXELS / (bitmap.width * bitmap.height)));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));

  const off = new OffscreenCanvas(w, h);
  const ctx = off.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();

  const px = ctx.getImageData(0, 0, w, h).data;
  const out: Rgb = { data: new Float32Array(w * h * 3), width: w, height: h };
  for (let i = 0; i < w * h; i++) {
    out.data[i * 3] = px[i * 4] / 255;
    out.data[i * 3 + 1] = px[i * 4 + 1] / 255;
    out.data[i * 3 + 2] = px[i * 4 + 2] / 255;
  }
  return out;
}

// --- run --------------------------------------------------------------------

async function run(): Promise<void> {
  if (!source || busy) return;
  busy = true;
  setControlsEnabled(false);
  try {
    await modelReady;
    const t0 = performance.now();
    cutout = await removeBackground(source, {
      edgeMode: edgeSel.value as EdgeMode,
      onStage: showProgress,
    });
    console.info(`pipeline: ${(performance.now() - t0).toFixed(0)}ms`);
    render();
    work.classList.add("show");
    hideProgress();
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
  } finally {
    busy = false;
    setControlsEnabled(true);
  }
}

function setControlsEnabled(on: boolean): void {
  for (const id of ["edge", "bg", "rerun", "dl", "dlMatte"]) {
    ($(id) as HTMLButtonElement | HTMLSelectElement).disabled = !on;
  }
}

function render(): void {
  if (!cutout) return;
  const bg = parseBackground(bgSel.value);
  const alpha = upsampleAlpha(cutout.alpha, cutout.image);
  const img = toImageData(colourPlane(cutout), alpha, bg);
  canvas.width = img.width;
  canvas.height = img.height;
  canvas.getContext("2d")!.putImageData(img, 0, 0);
}

function parseBackground(v: string): [number, number, number] | null {
  if (v === "none") return null;
  return [
    parseInt(v.slice(0, 2), 16) / 255,
    parseInt(v.slice(2, 4), 16) / 255,
    parseInt(v.slice(4, 6), 16) / 255,
  ];
}

edgeSel.addEventListener("change", () => void run());
bgSel.addEventListener("change", render);
$("rerun").addEventListener("click", () => void run());

$("dl").addEventListener("click", () => {
  if (!cutout) return;
  // Crop to the subject so the download is not mostly empty pixels.
  const [l, t, r, b] = boundingBox(cutout.alpha);
  download(cropCanvas(l, t, r - l, b - t), "cutout.png");
});

$("dlMatte").addEventListener("click", () => {
  if (!cutout) return;
  // The plane every compositor wants, and almost no free tool hands over.
  const a = upsampleAlpha(cutout.alpha, cutout.image);
  const px = new Uint8ClampedArray(a.width * a.height * 4);
  for (let i = 0; i < a.width * a.height; i++) {
    const v = a.data[i] * 255;
    px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = v;
    px[i * 4 + 3] = 255;
  }
  const c = document.createElement("canvas");
  c.width = a.width; c.height = a.height;
  c.getContext("2d")!.putImageData(new ImageData(px, a.width, a.height), 0, 0);
  download(c, "matte.png");
});

function cropCanvas(x: number, y: number, w: number, h: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d")!.drawImage(canvas, x, y, w, h, 0, 0, w, h);
  return c;
}

function download(c: HTMLCanvasElement, name: string): void {
  c.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }, "image/png");
}
