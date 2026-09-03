/**
 * The browser pipeline. Mirrors the Python orchestrator stage for stage.
 *
 *   segment -> trimap -> matte -> decontaminate -> composite
 *
 * `edgeMode` decides how far down that chain a call goes. Default is
 * "matte", because "naive" is what makes free background removers look free.
 */

import * as decontaminate from "./stages/decontaminate";
import * as matte from "./stages/matte";
import * as segment from "./stages/segment";
import * as trimapStage from "./stages/trimap";
import { plane, type Cutout, type EdgeMode, type Rgb } from "./types";

/** Above this, stages 2-5 run downscaled and stage 6 reprojects. */
export const WORKING_PIXELS = 1600 * 1600;

export interface RunOptions {
  edgeMode?: EdgeMode;
  bandWidth?: number;
  onStage?: (label: string, fraction: number) => void;
}

export async function removeBackground(image: Rgb, opts: RunOptions = {}): Promise<Cutout> {
  const { edgeMode = "matte", bandWidth = 12, onStage } = opts;

  onStage?.("Finding the subject", 0.05);
  const prob = await segment.probabilityMap(image);

  let alpha = plane(image.width, image.height);
  let trimap: Uint8Array | null = null;

  if (edgeMode === "naive") {
    for (let i = 0; i < prob.data.length; i++) alpha.data[i] = prob.data[i] >= 0.5 ? 1 : 0;
  } else {
    onStage?.("Marking the uncertain edge", 0.45);
    trimap = trimapStage.derive(prob, { bandWidth });

    if (edgeMode === "decontaminate") {
      alpha = prob;
    } else {
      onStage?.("Solving hair and soft edges", 0.55);
      alpha = matte.solve(image, trimap, {
        onProgress: (f) => onStage?.("Solving hair and soft edges", 0.55 + 0.3 * f),
      });
    }
  }

  let foreground: Rgb | null = null;
  if (edgeMode !== "naive") {
    onStage?.("Removing colour fringing", 0.9);
    foreground = decontaminate.estimate(image, alpha);
  }

  onStage?.("Done", 1);
  return { image, alpha, foreground, trimap, edgeMode };
}

/** Foreground estimate if stage 5 ran, else the raw source pixels. */
export function colourPlane(cutout: Cutout): Rgb {
  return cutout.foreground ?? cutout.image;
}
