/**
 * Traced shapes -> a coverage mask the rest of the pipeline already
 * understands.
 *
 * The canvas rasteriser does this better than we would, so the cubics go
 * straight to `Path2D` rather than being flattened first. Two details matter:
 *
 * even-odd fill  Nested loops become holes for free. Trace the mug, then
 *                trace the gap in its handle, and the gap is background with
 *                no notion of "subtract" needed anywhere in the model.
 * supersampling  Canvas anti-aliasing is decent but only ~4-bit along a near
 *                horizontal edge, and a hand-cut edge is exactly where that
 *                shows. Rendering at 2x and boxing down gives a smoother
 *                ramp, which then seeds the unknown band in stage 3.
 */

import { plane, type Plane } from "../types";
import { segment, segmentCount, type Shape } from "./geometry";

/** Above this many output pixels, skip supersampling rather than risk the tab. */
const SUPERSAMPLE_PIXEL_LIMIT = 4_000_000;

export function toPath2D(shapes: Shape[], scale = 1): Path2D {
  const path = new Path2D();
  for (const shape of shapes) {
    if (!shape.closed || shape.anchors.length < 3) continue;
    const first = shape.anchors[0];
    path.moveTo(first.x * scale, first.y * scale);
    const segs = segmentCount(shape);
    for (let i = 0; i < segs; i++) {
      const c = segment(shape, i);
      path.bezierCurveTo(
        c[2] * scale, c[3] * scale,
        c[4] * scale, c[5] * scale,
        c[6] * scale, c[7] * scale,
      );
    }
    path.closePath();
  }
  return path;
}

/** Number of shapes that actually enclose area. */
export function closedCount(shapes: Shape[]): number {
  return shapes.filter((s) => s.closed && s.anchors.length >= 3).length;
}

/**
 * Rasterise to a [0,1] coverage plane, 1 inside the trace.
 *
 * The result stands in for stage 2's probability map: soft along the traced
 * edge, hard everywhere else, which is precisely the shape stage 3 expects.
 */
export function maskFromShapes(shapes: Shape[], width: number, height: number): Plane {
  const out = plane(width, height);
  if (closedCount(shapes) === 0) return out;

  const ss = width * height * 4 <= SUPERSAMPLE_PIXEL_LIMIT ? 2 : 1;
  const w = width * ss;
  const h = height * ss;

  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("could not get a 2D context to rasterise the trace");
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fill(toPath2D(shapes, ss), "evenodd");

  const px = ctx.getImageData(0, 0, w, h).data;
  if (ss === 1) {
    for (let i = 0; i < width * height; i++) out.data[i] = px[i * 4 + 3] / 255;
    return out;
  }

  const inv = 1 / (ss * ss * 255);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let dy = 0; dy < ss; dy++) {
        const row = (y * ss + dy) * w;
        for (let dx = 0; dx < ss; dx++) acc += px[(row + x * ss + dx) * 4 + 3];
      }
      out.data[y * width + x] = acc * inv;
    }
  }
  return out;
}
