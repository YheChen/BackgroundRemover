/**
 * Verify the TypeScript matting solver against the Python reference.
 *
 * The fixture is produced by pymatting's closed-form solver. A matting
 * solver that is subtly wrong is worse than no solver at all: it looks
 * plausible and quietly ruins every edge, so this comparison gates the
 * browser build.
 */
import { readFileSync } from "node:fs";
import { solve } from "../src/stages/matte";

const fx = JSON.parse(readFileSync(new URL("../fixtures/matte-fixture.json", import.meta.url), "utf8"));
const { width, height } = fx;

const image = { data: Float32Array.from(fx.image), width, height };
const trimap = Uint8Array.from(fx.trimap);
const expected = Float32Array.from(fx.expected);

const t0 = performance.now();
const got = solve(image, trimap, { maxIterations: 400, tolerance: 1e-7 });
const ms = performance.now() - t0;

let maxAbs = 0;
let sumAbs = 0;
let softMax = 0;
let softN = 0;
for (let i = 0; i < expected.length; i++) {
  const d = Math.abs(got.data[i] - expected[i]);
  maxAbs = Math.max(maxAbs, d);
  sumAbs += d;
  // Soft pixels are the ones that matter: any solver gets 0 and 1 right.
  if (expected[i] > 0.05 && expected[i] < 0.95) { softMax = Math.max(softMax, d); softN++; }
}
const meanAbs = sumAbs / expected.length;

console.log(`solved ${width}x${height} in ${ms.toFixed(0)}ms`);
console.log(`mean |diff|        ${meanAbs.toFixed(6)}`);
console.log(`max  |diff|        ${maxAbs.toFixed(6)}`);
console.log(`max |diff| on soft ${softMax.toFixed(6)}  (n=${softN})`);

const ok = meanAbs < 0.01 && softMax < 0.12;
console.log(`\nVERDICT: ${ok ? "MATCHES the Python reference" : "MISMATCH"}`);
process.exit(ok ? 0 : 1);
