import { defineConfig } from "vite";

// onnxruntime-web's .wasm and .mjs runtime files are loaded at runtime by
// URL, not imported, so a bundler never sees them. They are copied into
// public/ort by `npm run copy:ort` (wired to predev/prebuild), which is the
// one location Vite serves identically in dev and in a build.
export default defineConfig({
  base: "./",
  // Do not pre-bundle ORT: the optimizer rewrites its runtime dynamic
  // imports into .vite/deps/, which breaks its own wasm loading.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022", assetsInlineLimit: 0 },
  // SharedArrayBuffer is required for multi-threaded WASM. Without these
  // headers ORT silently drops to single-threaded, which is far slower.
  server: {
    port: 5173,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
