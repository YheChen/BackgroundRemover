# Browser build — step 3

Not started. This is the free product, and the reason it can stay free:
inference in the browser has zero marginal cost per image, so "free forever"
is structurally true rather than a promise that breaks when the bill arrives.

## Plan

- `onnxruntime-web` on **WebGPU**, with a WASM fallback. Benchmarks put
  WebGPU at ~20x multi-threaded CPU and ~550x single-threaded, which is
  interactive. WebGPU has shipped by default since Chrome/Edge 113 desktop and
  Chrome 121 on Android.
- Consume the same ONNX graph `scripts/export_onnx.py` produces, quantised.
- Port stages 3–6 to TypeScript. Stage 3 is pure array work; stage 6 is a
  guided filter, which is a good fit for a WebGL/WebGPU shader. Stage 4 is the
  hard port — closed-form matting needs a sparse solver, so either compile
  `pymatting`'s approach to WASM or use a neural matter in the graph.
- No backend, no upload, no account. Static hosting.

## Constraints to design around

- **Cold start is the whole first impression.** A 200 MB–1 GB model download
  on first visit reads as "broken" without real progress reporting and hard
  caching (Cache API, not just HTTP).
- **Mobile memory caps you well below 50 MP.** Detect and degrade: offer the
  reduced-resolution result rather than crashing the tab.
- Full-res stage-6 reprojection of a 50 MP image is not happening in a tab.
  That is what the container in step 4 is for.
