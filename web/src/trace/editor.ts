/**
 * The hand-trace editor: an image viewport, three ways to lay down a path,
 * and one editable path model underneath all of them.
 *
 *   curve   A pen. Click for a corner, click-drag to pull a Bezier handle.
 *   edge    A live wire. Click a start point and the path snaps onto the
 *           strongest gradient route to the cursor (see livewire.ts).
 *   free    A lasso. Drag freehand; the stroke is simplified and smoothed
 *           into anchors so it can still be edited afterwards.
 *
 * Whatever laid a point down, it is an `Anchor` afterwards, and dragging,
 * splitting and deleting work the same everywhere. Tools only differ in how
 * they *append*.
 *
 * Editing beats tool modes: a pointer-down that lands on an existing anchor
 * or handle always drags it, whichever tool is selected. There is no separate
 * "direct selection" mode to hunt for.
 */

import { plane, type Plane, type Rgb } from "../types";
import {
  anchor,
  cloneShapes,
  flatten,
  hitAnchor,
  hitSegment,
  polyline,
  segment,
  segmentCount,
  simplify,
  smooth,
  splitSegment,
  type AnchorHit,
  type Shape,
} from "./geometry";
import * as livewire from "./livewire";
import { closedCount, maskFromShapes, toPath2D } from "./raster";

export type Tool = "curve" | "edge" | "free";

/** Grab radius for anchors and handles, in *screen* px. */
const HIT_PX = 9;
/** Simplification tolerance for the freehand and wire tools, in screen px. */
const SIMPLIFY_PX = 2.2;
const MAX_HISTORY = 60;

export interface EditorState {
  tool: Tool;
  closedShapes: number;
  anchors: number;
  drawing: boolean;
  canUndo: boolean;
  canCut: boolean;
  zoom: number;
}

export interface EditorOptions {
  onState?: (state: EditorState) => void;
  onStatus?: (message: string) => void;
}

export class TraceEditor {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private opts: EditorOptions;

  private image: Rgb | null = null;
  private bitmap: HTMLCanvasElement | null = null;
  private field: livewire.EdgeField | null = null;

  private shapes: Shape[] = [];
  /** Index of the open shape being drawn, or -1. */
  private active = -1;
  private selected: AnchorHit | null = null;
  private history: Shape[][] = [];

  private tool: Tool = "edge";
  private view = { scale: 1, tx: 0, ty: 0 };

  private wire: livewire.Wire | null = null;
  private wirePreview: number[] | null = null;
  private freehand: number[] | null = null;
  private cursor: { x: number; y: number } | null = null;

  private dragging: AnchorHit | null = null;
  private pulling: { shape: number; anchor: number } | null = null;
  private panning: { x: number; y: number } | null = null;
  private spaceDown = false;
  private raf = 0;
  private observer: ResizeObserver | null = null;

  constructor(canvas: HTMLCanvasElement, opts: EditorOptions = {}) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("could not get a 2D context for the trace editor");
    this.ctx = ctx;
    this.opts = opts;

    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerup", this.onPointerUp);
    canvas.addEventListener("pointerleave", this.onPointerLeave);
    canvas.addEventListener("wheel", this.onWheel, { passive: false });
    canvas.addEventListener("contextmenu", this.onContextMenu);
    canvas.addEventListener("dblclick", this.onDoubleClick);
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);

    if (typeof ResizeObserver !== "undefined") {
      this.observer = new ResizeObserver(() => this.resize());
      this.observer.observe(canvas.parentElement ?? canvas);
    }
  }

  destroy(): void {
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerup", this.onPointerUp);
    this.canvas.removeEventListener("pointerleave", this.onPointerLeave);
    this.canvas.removeEventListener("wheel", this.onWheel);
    this.canvas.removeEventListener("contextmenu", this.onContextMenu);
    this.canvas.removeEventListener("dblclick", this.onDoubleClick);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    this.observer?.disconnect();
    cancelAnimationFrame(this.raf);
  }

  // --- public API -----------------------------------------------------------

  setImage(image: Rgb): void {
    this.image = image;
    this.field = null;
    this.shapes = [];
    this.active = -1;
    this.selected = null;
    this.history = [];
    this.wire = null;
    this.wirePreview = null;

    const c = document.createElement("canvas");
    c.width = image.width;
    c.height = image.height;
    const cctx = c.getContext("2d")!;
    const px = new Uint8ClampedArray(image.width * image.height * 4);
    for (let i = 0; i < image.width * image.height; i++) {
      px[i * 4] = image.data[i * 3] * 255;
      px[i * 4 + 1] = image.data[i * 3 + 1] * 255;
      px[i * 4 + 2] = image.data[i * 3 + 2] * 255;
      px[i * 4 + 3] = 255;
    }
    cctx.putImageData(new ImageData(px, image.width, image.height), 0, 0);
    this.bitmap = c;

    this.resize();
    this.fit();
  }

  setTool(tool: Tool): void {
    this.tool = tool;
    // A half-drawn freehand stroke means nothing to the pen, and a wire
    // seeded for the old tool would fire on the next click.
    this.freehand = null;
    this.wirePreview = null;
    if (tool === "edge") {
      this.ensureField();
      this.reseedWire();
    }
    this.emit();
    this.request();
  }

  currentTool(): Tool {
    return this.tool;
  }

  /** True once at least one loop is closed, so a cutout is possible. */
  canCut(): boolean {
    return closedCount(this.shapes) > 0;
  }

  /** Rasterise the trace at image resolution. */
  mask(): Plane {
    if (!this.image) return plane(1, 1);
    return maskFromShapes(this.shapes, this.image.width, this.image.height);
  }

  traced(): Shape[] {
    return cloneShapes(this.shapes);
  }

  /** Seed the editor with paths from elsewhere — e.g. a model mask contour. */
  setShapes(shapes: Shape[]): void {
    this.pushHistory();
    this.shapes = cloneShapes(shapes);
    this.active = -1;
    this.selected = null;
    this.emit();
    this.request();
  }

  closeShape(): void {
    const s = this.shapes[this.active];
    if (!s || s.anchors.length < 3) return;
    this.pushHistory();
    s.closed = true;
    this.active = -1;
    this.wire = null;
    this.wirePreview = null;
    this.emit();
    this.request();
  }

  undo(): void {
    const previous = this.history.pop();
    if (!previous) return;
    this.shapes = previous;
    this.active = this.shapes.findIndex((s) => !s.closed);
    this.selected = null;
    this.wire = null;
    this.wirePreview = null;
    this.reseedWire();
    this.emit();
    this.request();
  }

  clear(): void {
    if (this.shapes.length === 0) return;
    this.pushHistory();
    this.shapes = [];
    this.active = -1;
    this.selected = null;
    this.wire = null;
    this.wirePreview = null;
    this.emit();
    this.request();
  }

  /** Fit the image to the viewport and centre it. */
  fit(): void {
    if (!this.image) return;
    const { width: vw, height: vh } = this.viewport();
    const scale = Math.min(vw / this.image.width, vh / this.image.height);
    this.view.scale = scale;
    this.view.tx = (vw - this.image.width * scale) / 2;
    this.view.ty = (vh - this.image.height * scale) / 2;
    this.emit();
    this.request();
  }

  zoomBy(factor: number): void {
    const { width: vw, height: vh } = this.viewport();
    this.zoomAt(vw / 2, vh / 2, factor);
  }

  // --- geometry helpers -----------------------------------------------------

  private viewport(): { width: number; height: number } {
    return { width: this.canvas.clientWidth || 1, height: this.canvas.clientHeight || 1 };
  }

  private toImage(sx: number, sy: number): [number, number] {
    return [(sx - this.view.tx) / this.view.scale, (sy - this.view.ty) / this.view.scale];
  }

  private hitRadius(): number {
    return HIT_PX / this.view.scale;
  }

  private clampToImage(x: number, y: number): [number, number] {
    if (!this.image) return [x, y];
    return [
      Math.min(this.image.width - 1, Math.max(0, x)),
      Math.min(this.image.height - 1, Math.max(0, y)),
    ];
  }

  private pointerImage(e: PointerEvent): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    return this.toImage(e.clientX - rect.left, e.clientY - rect.top);
  }

  private ensureField(): void {
    if (this.field || !this.image) return;
    this.opts.onStatus?.("Finding edges…");
    const t0 = performance.now();
    this.field = new livewire.EdgeField(this.image);
    console.info(`[bgremover] edge field: ${(performance.now() - t0).toFixed(0)}ms`);
    this.opts.onStatus?.("");
  }

  private lastAnchor(): { shape: number; index: number } | null {
    const s = this.shapes[this.active];
    if (!s || s.anchors.length === 0) return null;
    return { shape: this.active, index: s.anchors.length - 1 };
  }

  private reseedWire(): void {
    this.wire = null;
    if (this.tool !== "edge" || !this.field) return;
    const last = this.lastAnchor();
    if (!last) return;
    const a = this.shapes[last.shape].anchors[last.index];
    this.wire = livewire.seed(this.field, a.x, a.y);
  }

  private pushHistory(): void {
    this.history.push(cloneShapes(this.shapes));
    if (this.history.length > MAX_HISTORY) this.history.shift();
  }

  private startShape(x: number, y: number): void {
    this.shapes.push({ anchors: [anchor(x, y)], closed: false });
    this.active = this.shapes.length - 1;
  }

  /** Append an xy-pair polyline to the active shape, dropping its first point. */
  private appendPolyline(points: number[], smoothIt: boolean): void {
    const s = this.shapes[this.active];
    if (!s) return;
    const tol = SIMPLIFY_PX / this.view.scale;
    const simple = simplify(points, tol);
    const shape = smoothIt ? smooth(simple, false) : polyline(simple, false);
    // The first point is the anchor we came from; keep its handles.
    for (let i = 1; i < shape.anchors.length; i++) s.anchors.push(shape.anchors[i]);
  }

  // --- pointer handling -----------------------------------------------------

  private onPointerDown = (e: PointerEvent): void => {
    if (!this.image) return;
    this.canvas.focus();

    // Pan: middle button, or space held, or right button.
    if (e.button === 1 || e.button === 2 || this.spaceDown) {
      const rect = this.canvas.getBoundingClientRect();
      this.panning = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      this.canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }
    if (e.button !== 0) return;

    const [ix, iy] = this.pointerImage(e);
    const r = this.hitRadius();

    // Alt-click removes an anchor, or inserts one on a segment. Doing this
    // before the append path means alt is a modifier on every tool.
    if (e.altKey) {
      const onAnchor = hitAnchor(this.shapes, ix, iy, r);
      if (onAnchor && onAnchor.part === "point") {
        this.pushHistory();
        const s = this.shapes[onAnchor.shape];
        s.anchors.splice(onAnchor.anchor, 1);
        if (s.anchors.length === 0) {
          this.shapes.splice(onAnchor.shape, 1);
          if (this.active === onAnchor.shape) this.active = -1;
          else if (this.active > onAnchor.shape) this.active--;
        }
        this.selected = null;
        this.reseedWire();
        this.emit();
        this.request();
        return;
      }
      const onSeg = hitSegment(this.shapes, ix, iy, r);
      if (onSeg) {
        this.pushHistory();
        splitSegment(this.shapes[onSeg.shape], onSeg.segment, onSeg.t);
        this.selected = { shape: onSeg.shape, anchor: onSeg.segment + 1, part: "point" };
        this.emit();
        this.request();
        return;
      }
    }

    // An existing anchor or handle always wins over laying down a new point.
    const hit = hitAnchor(this.shapes, ix, iy, r, this.selected);
    if (hit) {
      const isFirstOfActive = hit.shape === this.active && hit.anchor === 0 && hit.part === "point";
      const canClose = isFirstOfActive && (this.shapes[this.active]?.anchors.length ?? 0) >= 3;
      if (canClose) {
        // Closing on the first anchor. The edge tool runs the wire home
        // first, so the last stretch snaps to the boundary like the rest.
        this.pushHistory();
        if (this.tool === "edge" && this.wire) {
          const first = this.shapes[this.active].anchors[0];
          this.appendPolyline(livewire.pathTo(this.wire, first.x, first.y), false);
          // The wire arrives at the first anchor; drop the duplicate.
          const anchors = this.shapes[this.active].anchors;
          const tail = anchors[anchors.length - 1];
          if (Math.hypot(tail.x - first.x, tail.y - first.y) < 1.5) anchors.pop();
        }
        this.shapes[this.active].closed = true;
        this.active = -1;
        this.wire = null;
        this.wirePreview = null;
        this.selected = null;
        this.emit();
        this.request();
        return;
      }
      this.pushHistory();
      this.selected = hit;
      this.dragging = hit;
      this.canvas.setPointerCapture(e.pointerId);
      this.request();
      return;
    }

    // Nothing under the cursor: the tool appends.
    const [cx, cy] = this.clampToImage(ix, iy);
    this.pushHistory();

    if (this.tool === "free") {
      this.freehand = [cx, cy];
      this.canvas.setPointerCapture(e.pointerId);
      return;
    }

    if (this.tool === "edge") {
      this.ensureField();
      const [sx, sy] = this.field ? this.field.snap(cx, cy) : [cx, cy];
      if (this.active < 0) {
        this.startShape(sx, sy);
      } else if (this.wire) {
        this.appendPolyline(livewire.pathTo(this.wire, sx, sy), false);
      } else {
        this.shapes[this.active].anchors.push(anchor(sx, sy));
      }
      this.reseedWire();
      this.wirePreview = null;
      this.selected = { shape: this.active, anchor: this.shapes[this.active].anchors.length - 1, part: "point" };
      this.emit();
      this.request();
      return;
    }

    // curve
    if (this.active < 0) this.startShape(cx, cy);
    else this.shapes[this.active].anchors.push(anchor(cx, cy));
    const idx = this.shapes[this.active].anchors.length - 1;
    this.selected = { shape: this.active, anchor: idx, part: "point" };
    // Holding the drag pulls a handle out of the point just placed.
    this.pulling = { shape: this.active, anchor: idx };
    this.canvas.setPointerCapture(e.pointerId);
    this.emit();
    this.request();
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.image) return;
    const rect = this.canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    if (this.panning) {
      this.view.tx += sx - this.panning.x;
      this.view.ty += sy - this.panning.y;
      this.panning = { x: sx, y: sy };
      this.request();
      return;
    }

    const [ix, iy] = this.toImage(sx, sy);
    this.cursor = { x: ix, y: iy };

    if (this.pulling) {
      const a = this.shapes[this.pulling.shape]?.anchors[this.pulling.anchor];
      if (a) {
        // Symmetric handles: the smooth case is the common one, and a corner
        // is available by not dragging at all.
        a.outX = ix - a.x;
        a.outY = iy - a.y;
        a.inX = -a.outX;
        a.inY = -a.outY;
      }
      this.request();
      return;
    }

    if (this.dragging) {
      const a = this.shapes[this.dragging.shape]?.anchors[this.dragging.anchor];
      if (a) {
        if (this.dragging.part === "point") {
          const [cx, cy] = this.clampToImage(ix, iy);
          a.x = cx;
          a.y = cy;
        } else if (this.dragging.part === "out") {
          a.outX = ix - a.x;
          a.outY = iy - a.y;
          if (!e.altKey) {
            a.inX = -a.outX;
            a.inY = -a.outY;
          }
        } else {
          a.inX = ix - a.x;
          a.inY = iy - a.y;
          if (!e.altKey) {
            a.outX = -a.inX;
            a.outY = -a.inY;
          }
        }
      }
      this.request();
      return;
    }

    if (this.freehand) {
      const [cx, cy] = this.clampToImage(ix, iy);
      const n = this.freehand.length;
      // Drop samples that land on the previous one; they only cost work.
      if (n < 2 || Math.hypot(cx - this.freehand[n - 2], cy - this.freehand[n - 1]) > 0.5) {
        this.freehand.push(cx, cy);
      }
      this.request();
      return;
    }

    // Live wire preview to the cursor.
    if (this.tool === "edge" && this.active >= 0 && this.field) {
      if (this.wire && !livewire.covers(this.wire, ix, iy)) this.reseedWire();
      this.wirePreview = this.wire ? livewire.pathTo(this.wire, ix, iy) : null;
    } else {
      this.wirePreview = null;
    }
    this.request();
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    this.panning = null;
    this.pulling = null;
    this.dragging = null;

    if (this.freehand) {
      const points = this.freehand;
      this.freehand = null;
      if (points.length >= 6) {
        const tol = SIMPLIFY_PX / this.view.scale;
        if (this.active < 0) {
          // A lasso is a loop: close it on release rather than making the
          // user return to the exact pixel they started from.
          const shape = smooth(simplify(points, tol), true);
          shape.closed = true;
          this.shapes.push(shape);
        } else {
          this.appendPolyline(points, true);
        }
      } else {
        // Too short to be a stroke — treat it as a click and undo the snapshot.
        this.history.pop();
      }
    }

    this.reseedWire();
    this.emit();
    this.request();
  };

  private onPointerLeave = (): void => {
    this.cursor = null;
    this.wirePreview = null;
    this.request();
  };

  private onDoubleClick = (e: MouseEvent): void => {
    // Double-click finishes the shape without hunting for the first anchor.
    e.preventDefault();
    if (this.active >= 0) this.closeShape();
  };

  private onContextMenu = (e: MouseEvent): void => {
    // Right-drag pans, so the menu would fire on every pan.
    e.preventDefault();
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const factor = Math.exp(-e.deltaY * 0.0015);
    this.zoomAt(e.clientX - rect.left, e.clientY - rect.top, factor);
  };

  private zoomAt(sx: number, sy: number, factor: number): void {
    if (!this.image) return;
    const { width: vw, height: vh } = this.viewport();
    const fitScale = Math.min(vw / this.image.width, vh / this.image.height);
    const next = Math.min(24, Math.max(fitScale * 0.5, this.view.scale * factor));
    // Keep the image point under the cursor fixed.
    const [ix, iy] = this.toImage(sx, sy);
    this.view.scale = next;
    this.view.tx = sx - ix * next;
    this.view.ty = sy - iy * next;
    this.emit();
    this.request();
  }

  private onKeyDown = (e: KeyboardEvent): void => {
    const target = e.target as HTMLElement | null;
    if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;

    if (e.code === "Space") {
      this.spaceDown = true;
      this.canvas.style.cursor = "grab";
      e.preventDefault();
      return;
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
      this.undo();
      e.preventDefault();
      return;
    }
    switch (e.key) {
      case "Escape":
        if (this.active >= 0) {
          this.pushHistory();
          this.shapes.splice(this.active, 1);
          this.active = -1;
          this.wire = null;
          this.wirePreview = null;
          this.emit();
          this.request();
        }
        break;
      case "Enter":
        this.closeShape();
        break;
      case "Backspace":
      case "Delete": {
        const sel = this.selected;
        if (!sel) break;
        this.pushHistory();
        const s = this.shapes[sel.shape];
        if (s) {
          s.anchors.splice(sel.anchor, 1);
          if (s.anchors.length === 0) {
            this.shapes.splice(sel.shape, 1);
            if (this.active === sel.shape) this.active = -1;
            else if (this.active > sel.shape) this.active--;
          }
        }
        this.selected = null;
        this.reseedWire();
        this.emit();
        this.request();
        e.preventDefault();
        break;
      }
      case "0":
        this.fit();
        break;
      case "1":
        this.setTool("curve");
        break;
      case "2":
        this.setTool("edge");
        break;
      case "3":
        this.setTool("free");
        break;
      default:
        break;
    }
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    if (e.code === "Space") {
      this.spaceDown = false;
      this.canvas.style.cursor = "";
    }
  };

  // --- rendering ------------------------------------------------------------

  private resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = this.viewport();
    const w = Math.max(1, Math.round(width * dpr));
    const h = Math.max(1, Math.round(height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.request();
  }

  private request(): void {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.draw();
    });
  }

  private emit(): void {
    this.opts.onState?.({
      tool: this.tool,
      closedShapes: closedCount(this.shapes),
      anchors: this.shapes.reduce((n, s) => n + s.anchors.length, 0),
      drawing: this.active >= 0,
      canUndo: this.history.length > 0,
      canCut: this.canCut(),
      zoom: this.view.scale,
    });
  }

  private draw(): void {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const { width: vw, height: vh } = this.viewport();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, vw, vh);
    if (!this.image || !this.bitmap) return;

    const { scale, tx, ty } = this.view;
    // Once past 1:1 the user is tracing individual pixels; smoothing there
    // invents an edge that is not in the data.
    ctx.imageSmoothingEnabled = scale < 2;
    ctx.drawImage(this.bitmap, tx, ty, this.image.width * scale, this.image.height * scale);

    ctx.save();
    ctx.translate(tx, ty);
    ctx.scale(scale, scale);

    // Filled interior of the closed loops, even-odd so holes read as holes.
    if (closedCount(this.shapes) > 0) {
      const path = toPath2D(this.shapes, 1);
      ctx.fillStyle = "rgba(221, 46, 85, 0.18)";
      ctx.fill(path, "evenodd");
    }

    // Outlines. Dark casing under a light stroke so the path stays visible
    // over both a white shirt and a black background.
    for (const shape of this.shapes) {
      const outline = new Path2D();
      const pts = flatten(shape, 16);
      if (pts.length >= 4) {
        outline.moveTo(pts[0], pts[1]);
        const segs = segmentCount(shape);
        if (segs > 0) {
          for (let i = 0; i < segs; i++) {
            const c = segment(shape, i);
            outline.bezierCurveTo(c[2], c[3], c[4], c[5], c[6], c[7]);
          }
        }
      }
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 3 / scale;
      ctx.stroke(outline);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.4 / scale;
      ctx.stroke(outline);
    }

    // The in-flight stretch: wire preview, freehand stroke, or a rubber band.
    const pending = this.pendingPolyline();
    if (pending && pending.length >= 4) {
      ctx.beginPath();
      ctx.moveTo(pending[0], pending[1]);
      for (let i = 2; i < pending.length; i += 2) ctx.lineTo(pending[i], pending[i + 1]);
      ctx.strokeStyle = "rgba(0,0,0,0.5)";
      ctx.lineWidth = 3 / scale;
      ctx.stroke();
      ctx.strokeStyle = "#DD2E55";
      ctx.lineWidth = 1.6 / scale;
      ctx.stroke();
    }

    ctx.restore();

    this.drawHandles(ctx);
  }

  /** The un-committed stretch under the cursor, in image coords. */
  private pendingPolyline(): number[] | null {
    if (this.freehand) return this.freehand;
    if (this.wirePreview) return this.wirePreview;
    if (this.tool === "curve" && this.active >= 0 && this.cursor && !this.dragging && !this.pulling) {
      const anchors = this.shapes[this.active].anchors;
      const last = anchors[anchors.length - 1];
      // Rubber band that respects the handle already pulled out of `last`.
      const c = [last.x, last.y, last.x + last.outX, last.y + last.outY, this.cursor.x, this.cursor.y, this.cursor.x, this.cursor.y];
      const out: number[] = [];
      for (let k = 0; k <= 16; k++) {
        const t = k / 16;
        const u = 1 - t;
        out.push(
          u * u * u * c[0] + 3 * u * u * t * c[2] + 3 * u * t * t * c[4] + t * t * t * c[6],
          u * u * u * c[1] + 3 * u * u * t * c[3] + 3 * u * t * t * c[5] + t * t * t * c[7],
        );
      }
      return out;
    }
    return null;
  }

  /** Anchors and handles, drawn in screen space so they keep a constant size. */
  private drawHandles(ctx: CanvasRenderingContext2D): void {
    const { scale, tx, ty } = this.view;
    const sx = (x: number) => x * scale + tx;
    const sy = (y: number) => y * scale + ty;

    for (let s = 0; s < this.shapes.length; s++) {
      const shape = this.shapes[s];
      for (let i = 0; i < shape.anchors.length; i++) {
        const a = shape.anchors[i];
        const isSelected =
          this.selected?.shape === s && this.selected.anchor === i;
        const isFirstOfActive = s === this.active && i === 0;

        if (isSelected && (a.outX || a.outY || a.inX || a.inY)) {
          ctx.strokeStyle = "rgba(255,255,255,0.9)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(sx(a.x + a.inX), sy(a.y + a.inY));
          ctx.lineTo(sx(a.x), sy(a.y));
          ctx.lineTo(sx(a.x + a.outX), sy(a.y + a.outY));
          ctx.stroke();
          for (const [hx, hy] of [
            [a.x + a.inX, a.y + a.inY],
            [a.x + a.outX, a.y + a.outY],
          ]) {
            ctx.beginPath();
            ctx.arc(sx(hx), sy(hy), 3.5, 0, Math.PI * 2);
            ctx.fillStyle = "#fff";
            ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,0.6)";
            ctx.stroke();
          }
        }

        const r = isFirstOfActive ? 5 : 3.5;
        ctx.beginPath();
        ctx.rect(sx(a.x) - r, sy(a.y) - r, r * 2, r * 2);
        ctx.fillStyle = isSelected ? "#DD2E55" : isFirstOfActive ? "#fff" : "rgba(255,255,255,0.92)";
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = "rgba(0,0,0,0.65)";
        ctx.stroke();
      }
    }
  }
}
