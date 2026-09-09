/**
 * Trace geometry — the single data model behind all three trace tools.
 *
 * The curve, edge and freehand tools feel different to use but they all
 * produce the same thing: a list of anchors, each carrying two handle
 * offsets. A segment whose two facing handles are zero renders as a straight
 * line, so a polyline is not a special case — it is a cubic whose controls
 * sit on its endpoints. That is why the magnetic tool's dense edge path and
 * the pen tool's two-anchor curve can be dragged, split and deleted by the
 * same code.
 *
 * Coordinates are image pixels throughout. The view transform belongs to the
 * editor, not here.
 */

export interface Anchor {
  x: number;
  y: number;
  /** Handle offsets, relative to the anchor, in image px. */
  inX: number;
  inY: number;
  outX: number;
  outY: number;
}

export interface Shape {
  anchors: Anchor[];
  /** A closed shape encloses area and can be filled; an open one is in progress. */
  closed: boolean;
}

export function anchor(x: number, y: number): Anchor {
  return { x, y, inX: 0, inY: 0, outX: 0, outY: 0 };
}

export function cloneShapes(shapes: Shape[]): Shape[] {
  return shapes.map((s) => ({ closed: s.closed, anchors: s.anchors.map((a) => ({ ...a })) }));
}

/** Segments in a shape: one per gap between consecutive anchors, plus the closing one. */
export function segmentCount(shape: Shape): number {
  const n = shape.anchors.length;
  if (n < 2) return 0;
  return shape.closed ? n : n - 1;
}

/** Cubic control points of segment `i` as [x0,y0, x1,y1, x2,y2, x3,y3]. */
export function segment(shape: Shape, i: number): number[] {
  const n = shape.anchors.length;
  const a = shape.anchors[i];
  const b = shape.anchors[(i + 1) % n];
  return [
    a.x, a.y,
    a.x + a.outX, a.y + a.outY,
    b.x + b.inX, b.y + b.inY,
    b.x, b.y,
  ];
}

function cubicAt(c: number[], t: number): [number, number] {
  const u = 1 - t;
  const w0 = u * u * u;
  const w1 = 3 * u * u * t;
  const w2 = 3 * u * t * t;
  const w3 = t * t * t;
  return [
    w0 * c[0] + w1 * c[2] + w2 * c[4] + w3 * c[6],
    w0 * c[1] + w1 * c[3] + w2 * c[5] + w3 * c[7],
  ];
}

/**
 * Flatten to an xy-pair polyline. Only hit-testing and the on-screen preview
 * need this — rasterisation hands the cubics straight to `Path2D`, which
 * subdivides better than we would.
 */
export function flatten(shape: Shape, stepsPerSegment = 12): number[] {
  const out: number[] = [];
  const segs = segmentCount(shape);
  if (segs === 0) {
    for (const a of shape.anchors) out.push(a.x, a.y);
    return out;
  }
  for (let i = 0; i < segs; i++) {
    const c = segment(shape, i);
    const straight = c[2] === c[0] && c[3] === c[1] && c[4] === c[6] && c[5] === c[7];
    const steps = straight ? 1 : stepsPerSegment;
    for (let s = 0; s < steps; s++) {
      const [x, y] = cubicAt(c, s / steps);
      out.push(x, y);
    }
  }
  if (!shape.closed) {
    const last = shape.anchors[shape.anchors.length - 1];
    out.push(last.x, last.y);
  }
  return out;
}

export interface AnchorHit {
  shape: number;
  anchor: number;
  /** Which part of the anchor was grabbed. */
  part: "point" | "in" | "out";
}

/** Nearest anchor point or handle within `radius` image px, or null. */
export function hitAnchor(
  shapes: Shape[],
  x: number,
  y: number,
  radius: number,
  handlesFor: AnchorHit | null = null,
): AnchorHit | null {
  const r2 = radius * radius;
  let best: AnchorHit | null = null;
  let bestD = r2;

  // Handles of the selected anchor win over other anchors' points: they sit on
  // top of the drawing and are the smaller target.
  if (handlesFor) {
    const a = shapes[handlesFor.shape]?.anchors[handlesFor.anchor];
    if (a) {
      for (const part of ["in", "out"] as const) {
        const hx = part === "in" ? a.x + a.inX : a.x + a.outX;
        const hy = part === "in" ? a.y + a.inY : a.y + a.outY;
        if (hx === a.x && hy === a.y) continue;
        const d = (hx - x) * (hx - x) + (hy - y) * (hy - y);
        if (d < bestD) {
          bestD = d;
          best = { shape: handlesFor.shape, anchor: handlesFor.anchor, part };
        }
      }
      if (best) return best;
    }
  }

  for (let s = 0; s < shapes.length; s++) {
    const anchors = shapes[s].anchors;
    for (let i = 0; i < anchors.length; i++) {
      const d = (anchors[i].x - x) * (anchors[i].x - x) + (anchors[i].y - y) * (anchors[i].y - y);
      if (d < bestD) {
        bestD = d;
        best = { shape: s, anchor: i, part: "point" };
      }
    }
  }
  return best;
}

export interface SegmentHit {
  shape: number;
  segment: number;
  /** Parameter along the segment, for splitting. */
  t: number;
  x: number;
  y: number;
  distance: number;
}

/** Nearest point on any segment within `radius` image px, or null. */
export function hitSegment(
  shapes: Shape[],
  x: number,
  y: number,
  radius: number,
): SegmentHit | null {
  let best: SegmentHit | null = null;
  let bestD = radius * radius;
  const STEPS = 24;

  for (let s = 0; s < shapes.length; s++) {
    const segs = segmentCount(shapes[s]);
    for (let i = 0; i < segs; i++) {
      const c = segment(shapes[s], i);
      for (let k = 0; k <= STEPS; k++) {
        const t = k / STEPS;
        const [px, py] = cubicAt(c, t);
        const d = (px - x) * (px - x) + (py - y) * (py - y);
        if (d < bestD) {
          bestD = d;
          best = { shape: s, segment: i, t, x: px, y: py, distance: Math.sqrt(d) };
        }
      }
    }
  }
  return best;
}

/**
 * Split segment `i` at `t` and insert the new anchor, leaving the curve's
 * shape untouched. De Casteljau gives the two sub-cubics; their controls
 * become the neighbouring handles.
 */
export function splitSegment(shape: Shape, i: number, t: number): void {
  const c = segment(shape, i);
  const n = shape.anchors.length;
  const a = shape.anchors[i];
  const b = shape.anchors[(i + 1) % n];

  const lerp = (ax: number, ay: number, bx: number, by: number): [number, number] => [
    ax + (bx - ax) * t,
    ay + (by - ay) * t,
  ];
  const p01 = lerp(c[0], c[1], c[2], c[3]);
  const p12 = lerp(c[2], c[3], c[4], c[5]);
  const p23 = lerp(c[4], c[5], c[6], c[7]);
  const p012 = lerp(p01[0], p01[1], p12[0], p12[1]);
  const p123 = lerp(p12[0], p12[1], p23[0], p23[1]);
  const mid = lerp(p012[0], p012[1], p123[0], p123[1]);

  a.outX = p01[0] - a.x;
  a.outY = p01[1] - a.y;
  b.inX = p23[0] - b.x;
  b.inY = p23[1] - b.y;

  shape.anchors.splice(i + 1, 0, {
    x: mid[0],
    y: mid[1],
    inX: p012[0] - mid[0],
    inY: p012[1] - mid[1],
    outX: p123[0] - mid[0],
    outY: p123[1] - mid[1],
  });
}

/**
 * Ramer-Douglas-Peucker on an xy-pair polyline.
 *
 * Freehand and magnetic tracing both produce a point per pointer sample or
 * per pixel, which is far more than the curve needs and unusable to edit by
 * hand. Tolerance is in image px, so the editor scales it by the zoom — a
 * wobble the user cannot see should not become an anchor.
 */
export function simplify(points: number[], tolerance: number): number[] {
  const n = points.length / 2;
  if (n < 3) return points.slice();
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const tol2 = tolerance * tolerance;

  // Explicit stack: a hand-drawn stroke can be tens of thousands of points
  // and recursion here has blown the stack in other implementations.
  const stack: [number, number][] = [[0, n - 1]];
  while (stack.length) {
    const [first, last] = stack.pop()!;
    if (last <= first + 1) continue;
    const ax = points[first * 2];
    const ay = points[first * 2 + 1];
    const bx = points[last * 2];
    const by = points[last * 2 + 1];
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy;

    let worst = -1;
    let worstD = -1;
    for (let i = first + 1; i < last; i++) {
      const px = points[i * 2];
      const py = points[i * 2 + 1];
      let d: number;
      if (len2 === 0) {
        d = (px - ax) * (px - ax) + (py - ay) * (py - ay);
      } else {
        let t = ((px - ax) * dx + (py - ay) * dy) / len2;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const cx = ax + t * dx;
        const cy = ay + t * dy;
        d = (px - cx) * (px - cx) + (py - cy) * (py - cy);
      }
      if (d > worstD) {
        worstD = d;
        worst = i;
      }
    }
    if (worstD > tol2 && worst > 0) {
      keep[worst] = 1;
      stack.push([first, worst], [worst, last]);
    }
  }

  const out: number[] = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(points[i * 2], points[i * 2 + 1]);
  return out;
}

/** An xy-pair polyline as corner anchors — straight segments, no handles. */
export function polyline(points: number[], closed: boolean): Shape {
  const anchors: Anchor[] = [];
  for (let i = 0; i < points.length; i += 2) anchors.push(anchor(points[i], points[i + 1]));
  return { anchors, closed };
}

/**
 * An xy-pair polyline as a smooth curve.
 *
 * Catmull-Rom tangents converted to Bezier handles: the handle at p[i] points
 * along (p[i+1] - p[i-1]) scaled by a sixth, which is the standard equivalence
 * and reproduces the polyline's shape without the corners. A freehand stroke
 * traced round a shoulder should not come back looking like a saw.
 */
export function smooth(points: number[], closed: boolean, tension = 1 / 6): Shape {
  const n = points.length / 2;
  if (n < 3) return polyline(points, closed);
  const at = (i: number): [number, number] => {
    const j = closed ? (i + n) % n : i < 0 ? 0 : i > n - 1 ? n - 1 : i;
    return [points[j * 2], points[j * 2 + 1]];
  };

  const anchors: Anchor[] = [];
  for (let i = 0; i < n; i++) {
    const [px, py] = at(i);
    const [prevX, prevY] = at(i - 1);
    const [nextX, nextY] = at(i + 1);
    const tx = (nextX - prevX) * tension;
    const ty = (nextY - prevY) * tension;
    anchors.push({ x: px, y: py, inX: -tx, inY: -ty, outX: tx, outY: ty });
  }
  if (!closed) {
    anchors[0].inX = anchors[0].inY = 0;
    anchors[n - 1].outX = anchors[n - 1].outY = 0;
  }
  return { anchors, closed };
}

/** Signed area via the shoelace formula on the flattened outline. */
export function signedArea(shape: Shape): number {
  const pts = flatten(shape, 8);
  const n = pts.length / 2;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    sum += pts[i * 2] * pts[j * 2 + 1] - pts[j * 2] * pts[i * 2 + 1];
  }
  return sum / 2;
}
