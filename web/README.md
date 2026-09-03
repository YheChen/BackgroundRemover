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

## Blocker found while exporting stage 2 — read this first

BiRefNet's ASPP blocks use **deformable convolutions**. The ONNX standard
schema introduces `DeformConv` at **opset 22**, and the desktop
`onnxruntime` Python package does implement it — but `onnxruntime-web` ships a
different, much smaller kernel set for its WASM and WebGPU backends, and
`DeformConv` is very unlikely to be among them.

**Verify this before writing any browser code**, because it decides the whole
approach:

```js
// smallest possible check: load the exported graph and see if a session builds
const s = await ort.InferenceSession.create('birefnet-general.onnx');
```

If it fails the same way the desktop opset-18 export did
(`No Op registered for DeformConv`), the options are, in order of preference:

1. **Pick a model without deformable convs.** BEN2 is MIT and is already in
   the registry. Check its op set before assuming it is clean.
2. **Replace the 20 DeformConv nodes** with an equivalent subgraph
   (`GridSample`-based) in a post-export graph surgery pass. Doable, and it
   keeps BiRefNet's quality.
3. **Write a custom WebGPU kernel.** Real work; only worth it if 1 and 2 fail.
4. Fall back to a server for stage 2 — which gives up the zero-marginal-cost
   property that makes the free product viable, so treat it as a last resort.

Note the graph is also **857 MB across two files** (`.onnx` + `.onnx.data`) at
fp32. Quantisation is not optional for the browser; it is the difference
between a usable download and an abandoned one.

## Constraints to design around

- **Cold start is the whole first impression.** A 200 MB–1 GB model download
  on first visit reads as "broken" without real progress reporting and hard
  caching (Cache API, not just HTTP).
- **Mobile memory caps you well below 50 MP.** Detect and degrade: offer the
  reduced-resolution result rather than crashing the tab.
- Full-res stage-6 reprojection of a 50 MP image is not happening in a tab.
  That is what the container in step 4 is for.
