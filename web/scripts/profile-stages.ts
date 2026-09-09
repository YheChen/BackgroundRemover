/**
 * Where does the wall time actually go?
 *
 * Runs stages 3-6 on a real working-resolution image and reports each. Stage
 * 2 is measured separately (it is a fixed-cost 1024x1024 ONNX forward and
 * does not scale with source size).
 *
 * Usage: tsx scripts/profile-stages.ts <dir-with-cat_rgb.f32>
 */
import { readFileSync } from "node:fs";
import * as decontaminate from "../src/stages/decontaminate";
import * as matte from "../src/stages/matte";
import * as trimapStage from "../src/stages/trimap";
import { upsampleAlpha } from "../src/stages/composite";
import type { Plane, Rgb } from "../src/types";

const dir = process.argv[2];
const W = Number(process.argv[3]);
const H = Number(process.argv[4]);

const rgb: Rgb = {
  data: new Float32Array(readFileSync(`${dir}/cat_rgb.f32`).buffer.slice(0)),
  width: W, height: H,
};
const prob: Plane = {
  data: new Float32Array(readFileSync(`${dir}/cat_prob.f32`).buffer.slice(0)),
  width: W, height: H,
};

const time = <T,>(label: string, fn: () => T): T => {
  const t0 = performance.now();
  const r = fn();
  console.log(`  ${label.padEnd(28)} ${((performance.now() - t0) / 1000).toFixed(2)}s`);
  return r;
};

console.log(`image ${W}x${H} = ${(W * H / 1e6).toFixed(2)} MP\n`);

const tri = time("stage 3  trimap", () => trimapStage.derive(prob, { bandWidth: 12 }));
console.log(`  ${"".padEnd(28)} band = ${(trimapStage.bandFraction(tri) * 100).toFixed(1)}% of pixels\n`);

const alpha = time("stage 4  matte (CG)", () => matte.solve(rgb, tri));
const fg = time("stage 5  decontaminate", () => decontaminate.estimate(rgb, alpha));
time("stage 6  guided upsample", () => upsampleAlpha(alpha, rgb));

void fg;
