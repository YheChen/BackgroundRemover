/**
 * UI wiring. The pipeline itself lives in pipeline.ts; this file only moves
 * pixels between a file input, a canvas, and a download link.
 *
 * Two routes to a cutout, and the user picks:
 *
 *   Automatic    stage 2 finds the subject (see stages/segment.ts).
 *   Trace by hand  the user draws the boundary (see trace/editor.ts).
 *
 * They meet at stage 3. Both then get the trimap, matting and decontamination
 * stages, which is the point — a hand-traced edge needs those more than a
 * predicted one, because nobody draws hair by hand.
 */

import {
  colourPlane,
  removeBackground,
  removeBackgroundFromMask,
  WORKING_PIXELS,
} from "./pipeline";
import { boundingBox, toImageData, upsampleAlpha } from "./stages/composite";
import * as segment from "./stages/segment";
import { TraceEditor, type EditorState, type Tool } from "./trace/editor";
import type { Cutout, EdgeMode, Rgb } from "./types";

// Where the weights live. Defaults to same-origin for local dev; set
// VITE_MODEL_URL at build time to point at a CDN. On Vercel's Hobby tier the
// per-file upload cap is 100 MB and bandwidth is 100 GB/month, so a 159 MB
// model served from the app's own origin is both impossible and, if it were
// possible, would cap the site at roughly a thousand first-time visitors.
// ?model= selects a locally-served graph for benchmarking; otherwise the
// deployed build uses VITE_MODEL_URL.
const MODEL_KEY = (new URLSearchParams(location.search).get("model") ||
  "birefnet-lite") as "birefnet-lite" | "isnet-general-use";
const MODEL_URL = new URLSearchParams(location.search).get("model")
  ? `./models/${MODEL_KEY}.onnx`
  : import.meta.env.VITE_MODEL_URL || "./models/birefnet-lite.onnx";

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

const modeAuto = $<HTMLButtonElement>("modeAuto");
const modeManual = $<HTMLButtonElement>("modeManual");
const modeNote = $<HTMLParagraphElement>("modeNote");
const traceView = $<HTMLDivElement>("traceView");
const traceCanvas = $<HTMLCanvasElement>("trace");
const traceCount = $<HTMLSpanElement>("traceCount");
const traceHint = $<HTMLDivElement>("traceHint");
const editTraceBtn = $<HTMLButtonElement>("editTrace");

type Mode = "auto" | "manual";

let mode: Mode = "auto";
let source: Rgb | null = null;
let cutout: Cutout | null = null;
let editor: TraceEditor | null = null;
/** Which route produced what is on screen, so Re-run repeats the right one. */
let lastRoute: Mode = "auto";
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

/**
 * Loaded on first use, not on page load.
 *
 * The hand-trace route never touches the model, and making everyone pay
 * 210 MB before they have chosen a route would be a strange way to ship an
 * offline-capable tool. Automatic users see the download at the moment they
 * drop an image, which is where the progress bar already lives.
 */
let modelPromise: Promise<segment.Backend> | null = null;

function ensureModel(): Promise<segment.Backend> {
  if (modelPromise) return modelPromise;
  modelPromise = (async () => {
    try {
      // A 210 MB download with no feedback reads as "broken", so report bytes
      // rather than a spinner. The Cache API makes this a one-time cost.
      backendBadge.textContent = "loading model…";
      const ep = await segment.load(MODEL_URL, MODEL_KEY, (loaded, total) => {
        showProgress(
          `Downloading the model — ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`,
          loaded / total,
        );
      });
      backendBadge.textContent = ep === "webgpu" ? "WebGPU" : "WASM (CPU)";
      backendBadge.classList.add("on");
      return ep;
    } catch (err) {
      backendBadge.textContent = "unavailable";
      backendBadge.classList.remove("on");
      modelPromise = null; // let a retry work
      throw new Error(
        `Could not load the model: ${err instanceof Error ? err.message : String(err)}. ` +
          `If you are running from source, the file belongs at ` +
          `web/public/models/birefnet-lite.onnx. You can also switch to "Trace by hand", ` +
          `which needs no model at all.`,
      );
    }
  })();
  return modelPromise;
}

// --- mode -------------------------------------------------------------------

const MODE_NOTES: Record<Mode, string> = {
  auto: "A 210&nbsp;MB model finds the subject for you. Downloaded once, then cached.",
  manual:
    "You draw the boundary; matting still does the edges. No model, no download.",
};

function setMode(next: Mode): void {
  mode = next;
  modeAuto.setAttribute("aria-checked", String(next === "auto"));
  modeManual.setAttribute("aria-checked", String(next === "manual"));
  modeNote.innerHTML = MODE_NOTES[next];
  if (!source) return;
  if (next === "manual") openTrace();
  else void runAuto();
}

modeAuto.addEventListener("click", () => setMode("auto"));
modeManual.addEventListener("click", () => setMode("manual"));

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
    hideProgress();
    if (mode === "manual") openTrace();
    else await runAuto();
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

// --- the trace route --------------------------------------------------------

const HINTS: Record<Tool, string> = {
  curve:
    "Click to drop a point; <b>drag</b> as you click to curve the line through it. " +
    "Click the first point to close the loop.",
  edge:
    "Click once to start, then follow the boundary — the line snaps onto the strongest " +
    "edge between your clicks. Click the first point to close the loop.",
  free: "Drag to draw around the subject. The loop closes itself when you let go.",
};

const SHARED_HINT =
  "<kbd>scroll</kbd> zoom · <kbd>space</kbd>-drag or right-drag to pan · " +
  "<kbd>alt</kbd>-click a point to delete it, a line to add one · " +
  "<kbd>⌥</kbd>-drag a handle to break the curve · <kbd>enter</kbd> close · " +
  "<kbd>esc</kbd> drop the loop · <kbd>⌘Z</kbd> undo · <kbd>0</kbd> fit · " +
  "<kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> tools";

function ensureEditor(): TraceEditor {
  if (editor) return editor;
  editor = new TraceEditor(traceCanvas, {
    onState: onEditorState,
    onStatus: (msg) => {
      if (msg) showProgress(msg, 0.5);
      else hideProgress();
    },
  });
  return editor;
}

function onEditorState(state: EditorState): void {
  const loops = state.closedShapes;
  traceCount.textContent =
    `${state.anchors} point${state.anchors === 1 ? "" : "s"} · ` +
    `${loops} loop${loops === 1 ? "" : "s"} · ${Math.round(state.zoom * 100)}%`;
  ($("tUndo") as HTMLButtonElement).disabled = !state.canUndo;
  ($("tClose") as HTMLButtonElement).disabled = !state.drawing;
  ($("tCut") as HTMLButtonElement).disabled = !state.canCut;
  ($("tClear") as HTMLButtonElement).disabled = state.anchors === 0;
  for (const b of document.querySelectorAll<HTMLButtonElement>("[data-tool]")) {
    b.setAttribute("aria-checked", String(b.dataset.tool === state.tool));
  }
  traceHint.innerHTML = `${HINTS[state.tool]}<br><span>${SHARED_HINT}</span>`;
}

function openTrace(): void {
  if (!source) return;
  const ed = ensureEditor();
  work.classList.remove("show");
  traceView.classList.add("show");
  // The editor sizes itself to a laid-out viewport, so image first, then fit.
  ed.setImage(source);
  ed.setTool(ed.currentTool());
  traceView.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

for (const b of document.querySelectorAll<HTMLButtonElement>("[data-tool]")) {
  b.addEventListener("click", () => ensureEditor().setTool(b.dataset.tool as Tool));
}
$("tFit").addEventListener("click", () => editor?.fit());
$("tUndo").addEventListener("click", () => editor?.undo());
$("tClear").addEventListener("click", () => editor?.clear());
$("tClose").addEventListener("click", () => editor?.closeShape());
$("tCut").addEventListener("click", () => void runTrace());
editTraceBtn.addEventListener("click", () => {
  traceView.classList.add("show");
  work.classList.remove("show");
});

// --- runs -------------------------------------------------------------------

async function runAuto(): Promise<void> {
  if (!source || busy) return;
  busy = true;
  setControlsEnabled(false);
  try {
    await ensureModel();
    const t0 = performance.now();
    cutout = await removeBackground(source, {
      edgeMode: edgeSel.value as EdgeMode,
      onStage: showProgress,
    });
    console.info(`pipeline: ${(performance.now() - t0).toFixed(0)}ms`);
    lastRoute = "auto";
    finishRun();
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
  } finally {
    busy = false;
    setControlsEnabled(true);
  }
}

async function runTrace(): Promise<void> {
  if (!source || !editor || busy) return;
  if (!editor.canCut()) {
    showError("Close at least one loop first — an open path does not enclose anything to keep.");
    return;
  }
  busy = true;
  setControlsEnabled(false);
  clearError();
  try {
    showProgress("Rasterising the trace", 0.05);
    // Yield once so the label paints before the synchronous stages start.
    await new Promise((r) => setTimeout(r, 0));
    const mask = editor.mask();
    const t0 = performance.now();
    cutout = await removeBackgroundFromMask(source, mask, {
      edgeMode: edgeSel.value as EdgeMode,
      onStage: showProgress,
    });
    console.info(`trace pipeline: ${(performance.now() - t0).toFixed(0)}ms`);
    lastRoute = "manual";
    traceView.classList.remove("show");
    finishRun();
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
  } finally {
    busy = false;
    setControlsEnabled(true);
  }
}

function rerun(): void {
  if (lastRoute === "manual") void runTrace();
  else void runAuto();
}

function finishRun(): void {
  render();
  editTraceBtn.hidden = lastRoute !== "manual";
  work.classList.add("show");
  hideProgress();
}

function setControlsEnabled(on: boolean): void {
  for (const id of ["edge", "bg", "rerun", "dl", "dlMatte", "editTrace", "tCut"]) {
    const el = document.getElementById(id) as HTMLButtonElement | HTMLSelectElement | null;
    if (el) el.disabled = !on;
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

edgeSel.addEventListener("change", rerun);
bgSel.addEventListener("change", render);
$("rerun").addEventListener("click", rerun);

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

backendBadge.textContent = "model not loaded";
