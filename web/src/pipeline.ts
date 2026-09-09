/**
 * The browser pipeline. Mirrors the Python orchestrator stage for stage.
 *
 *   segment -> trimap -> matte -> decontaminate -> composite
 *
 * `edgeMode` decides how far down that chain a call goes. Default is
 * "matte", because "naive" is what makes free background removers look free.
 *
 * There are two ways in. `removeBackground` runs stage 2 and lets the model
 * find the subject. `removeBackgroundFromMask` takes a mask the user traced
 * by hand and starts at stage 3. Everything after stage 2 is shared, and
 * deliberately so: a hand-traced edge needs matting and decontamination more
 * than a model-predicted one does, not less, because a person cannot draw
 * per-pixel hair.
 */

import * as decontaminate from "./stages/decontaminate";
import * as matte from "./stages/matte";
import * as segment from "./stages/segment";
import * as trimapStage from "./stages/trimap";
import { plane, type Cutout, type EdgeMode, type Plane, type Rgb } from "./types";

/** Above this, stages 2-5 run downscaled and stage 6 reprojects. */
export const WORKING_PIXELS = 1600 * 1600;

export interface RunOptions {
  edgeMode?: EdgeMode;
  bandWidth?: number;
  onStage?: (label: string, fraction: number) => void;
}

export async function removeBackground(image: Rgb, opts: RunOptions = {}): Promise<Cutout> {
  opts.onStage?.("Finding the subject", 0.05);
  const prob = await segment.probabilityMap(image);
  return finish(image, prob, opts, 0.45);
}

/**
 * Cut out from a hand-traced coverage mask, skipping the model entirely.
 *
 * `mask` is anti-aliased along the traced boundary, so it enters stage 3
 * looking like a very confident probability map — which is what it is.
 */
export async function removeBackgroundFromMask(
  image: Rgb,
  mask: Plane,
  opts: RunOptions = {},
): Promise<Cutout> {
  if (mask.width !== image.width || mask.height !== image.height) {
    throw new Error(
      `traced mask is ${mask.width}x${mask.height} but the image is ` +
        `${image.width}x${image.height}; rasterise the trace at image size`,
    );
  }
  return finish(image, mask, opts, 0.1);
}

/**
 * Stages 3-5, shared by both entry points.
 *
 * `start` is where this run's progress bar picks up: the model path has
 * already spent most of its time by the time it gets here, the traced path
 * has spent none.
 */
async function finish(
  image: Rgb,
  prob: Plane,
  opts: RunOptions,
  start: number,
): Promise<Cutout> {
  const { edgeMode = "matte", bandWidth = 12, onStage } = opts;
  const span = 1 - start;

  let alpha = plane(image.width, image.height);
  let trimap: Uint8Array | null = null;

  if (edgeMode === "naive") {
    for (let i = 0; i < prob.data.length; i++) alpha.data[i] = prob.data[i] >= 0.5 ? 1 : 0;
  } else {
    onStage?.("Marking the uncertain edge", start);
    await breathe();
    trimap = trimapStage.derive(prob, { bandWidth });

    if (edgeMode === "decontaminate") {
      alpha = prob;
    } else {
      onStage?.("Solving hair and soft edges", start + 0.22 * span);
      await breathe();
      alpha = matte.solve(image, trimap, {
        onProgress: (f) => onStage?.("Solving hair and soft edges", start + span * (0.22 + 0.44 * f)),
      });
    }
  }

  let foreground: Rgb | null = null;
  if (edgeMode !== "naive") {
    onStage?.("Removing colour fringing", start + 0.88 * span);
    await breathe();
    foreground = decontaminate.estimate(image, alpha);
  }

  onStage?.("Done", 1);
  return { image, alpha, foreground, trimap, edgeMode };
}

/**
 * Hand the event loop one turn so the progress label just set actually
 * paints. Not a fix for the main-thread block — the stages themselves are
 * synchronous and a worker is the real answer — but without it the label is
 * always one stage behind what is running.
 */
function breathe(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** Foreground estimate if stage 5 ran, else the raw source pixels. */
export function colourPlane(cutout: Cutout): Rgb {
  return cutout.foreground ?? cutout.image;
}
