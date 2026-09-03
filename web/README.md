# Browser build

Background removal that runs **entirely in the browser**. No upload, no
account, no backend — and it works offline once the model is cached.

This is the free product, and the reason it can stay free: inference in the
browser has zero marginal cost per image, so "free forever" is structurally
true rather than a promise that breaks when the GPU bill arrives.

## Status

Working. Full pipeline (stages 2–6) runs on WebGPU with a WASM fallback.
Measured on an M-series laptop: a 208×242 image completes in ~1.9 s.

## Run it

```bash
npm install
npm run dev          # copies the ORT runtime, then serves on :5173
```

You also need the model at `public/models/birefnet-lite.onnx`. It is **not**
in the repo (210 MB). Build it from the Python side:

```bash
cd .. && python scripts/export_onnx.py --model birefnet-lite --web
python -c "import onnx,pathlib; m=onnx.load('weights/birefnet-lite-web.onnx'); \
onnx.save_model(m, 'web/public/models/birefnet-lite.onnx', save_as_external_data=False)"
```

The `--web` flag matters — see below.

## Why `--web` is not optional

BiRefNet's ASPP blocks use deformable convolutions. Exported normally those
become `DeformConv` nodes, and **onnxruntime-web has no DeformConv kernel on
any provider**, so the session fails to create at all:

```
Could not find an implementation for DeformConv(22) node
```

`scripts/deform_compat.py` rewrites them as the mathematically equivalent
`grid_sample` + 1×1 convolution per kernel tap. Verified against
`torchvision.ops.deform_conv2d` to a relative error of ~1e-15 across
padding, stride, dilation and kernel-size variations, and the resulting graph
matches the native one to `max|diff| = 0.00022`.

It costs ~11% inference time and 21 MB of graph, and it is the difference
between the browser build existing and not.

## Which model, and why

**BiRefNet_lite** (MIT, Swin-Tiny, 44.4M params) rather than the general
model. The general model's fp32 graph is 857 MB, which nobody is downloading
into a tab. Lite is 210 MB and agrees with it to **IoU 0.9843** on our test
image, at roughly twice the speed.

## Architecture

Stages mirror the Python package one-for-one, so a fix in either can be
carried across:

| File | Stage | Notes |
|---|---|---|
| `stages/segment.ts` | 2 | ORT session, ImageNet preprocessing, sigmoid |
| `stages/trimap.ts` | 3 | Separable morphology; band scaled to image size |
| `stages/matte.ts` | 4 | Matrix-free conjugate gradient — see below |
| `stages/decontaminate.ts` | 5 | Faithful port of Germer et al. multi-level |
| `stages/composite.ts` | 6 | Guided-filter upsample, compositing, crop |

### Stage 4 is the interesting one

The Python side hands matting to pymatting, which builds a sparse Laplacian
and factorises a preconditioner. Neither is practical here: for a 1024×1024
image the matting Laplacian has ~81M non-zeros.

Two things make it work:

- **Band-only.** We solve only for the unknown band, which stage 3 keeps at
  roughly 8% of pixels. Known pixels move to the right-hand side, turning a
  1M-pixel problem into an ~84k-unknown one.
- **Matrix-free.** `L` is never built. CG only needs `L@x`, and expanding one
  3×3 window collapses the double sum to `x_i − (S + d_i·Mv)/9` — one pass to
  build `S` and `v`, one 3×3 matvec, one pass to scatter. Not 81 multiply-adds
  per window.

**It is verified against the Python reference**, because a matting solver
that is subtly wrong is worse than none — it looks plausible and quietly
ruins every edge:

```bash
npm run verify:matte
```

That replays a pymatting-generated fixture through the TypeScript solver.
Current agreement: mean |diff| 0.000161, max 0.0039 — the 1/255 floor.

## Known gaps

- **Soft-pixel count runs ~50% higher than Python** (0.045 vs 0.029 on the
  test image) even though `fg` fraction matches and the solver passes its
  fixture. Most likely CG tolerance: this uses Jacobi preconditioning to a
  1e-5 relative residual, pymatting uses incomplete Cholesky and converges
  tighter. Worth tuning against the eval set.
- **`refine` (stage 6b) is not ported.** The Python side re-solves the
  boundary at native resolution in tiles; the browser stops at the guided
  upsample.
- **Cold start is 210 MB.** Cached by the browser after the first visit, and
  the download reports real byte progress, but it is still the first
  impression. Quantising to int8 is the obvious next move.
- **Mobile is untested.** `WORKING_PIXELS` is capped at 1600² to stay inside
  tab memory, but that number is a guess until someone profiles a phone.
