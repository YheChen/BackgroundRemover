/**
 * Shared types for the browser pipeline.
 *
 * Planes are kept separate for the same reason as the Python side: a
 * semi-transparent pixel's observed colour is contaminated by whatever was
 * behind it, so compositing source pixels onto a new background reproduces
 * the old background in the edge. `foreground` is the decontaminated
 * estimate; `image` is what the camera saw.
 */

export type EdgeMode = "naive" | "decontaminate" | "matte";

/** Trimap sentinels. Same values as the Python side, for debuggability. */
export const TRIMAP_BG = 0;
export const TRIMAP_UNKNOWN = 128;
export const TRIMAP_FG = 255;

/** An RGB image as planar-interleaved f32 in [0,1], length w*h*3. */
export interface Rgb {
  data: Float32Array;
  width: number;
  height: number;
}

/** A single-channel plane in [0,1], length w*h. */
export interface Plane {
  data: Float32Array;
  width: number;
  height: number;
}

export interface Cutout {
  image: Rgb;
  alpha: Plane;
  foreground: Rgb | null;
  trimap: Uint8Array | null;
  edgeMode: EdgeMode;
}

export function plane(width: number, height: number): Plane {
  return { data: new Float32Array(width * height), width, height };
}

export function rgb(width: number, height: number): Rgb {
  return { data: new Float32Array(width * height * 3), width, height };
}
