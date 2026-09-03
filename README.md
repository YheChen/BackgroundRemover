# BackgroundRemover

A free, open-source background remover with the **whole** pipeline, not just the
coarse mask — and built on permissively licensed weights only, so anyone can
actually ship it.

remove.bg's API sunsets on **1 December 2026**, moving to Leonardo.Ai. This is a
replacement.

> **Status: stages 2–6 working.** Both stage-2 backends are verified and agree
> (IoU 0.99969 between torch and ONNX on a real image). Export the weights once
> (see [Setup](#setup)) and `bgremover in.jpg out.png` produces a real cutout.
> Stage 1 is a deliberate no-op. See [Build order](#build-order).

## Why another one

Most open-source background removers stop at stage 2 — a coarse binary mask.
That is the easy stage, and it is why their output has hard, halo-ed edges
around hair. The two stages that actually make a cutout look professional are
alpha matting and foreground colour estimation, and almost nothing free ships
both.

The other problem is licensing. The best-known model in this space, BRIA
RMBG-2.0, is **CC BY-NC 4.0** — non-commercial only. It is also the default in
several popular wrappers, which means a lot of "free background remover"
projects are quietly non-compliant the moment anyone uses them commercially.
This repo refuses non-permissive weights at the registry level, and the test
suite enforces it.

## The pipeline

```
  1 classify  ->  2 segment  ->  3 trimap  ->  4 matte  ->  5 decontaminate  ->  6 composite
   subject         coarse         unknown      continuous     true fg            full-res
   routing          mask           band          alpha         colour           reproject
```

| # | Stage | What it does | Status |
|---|-------|--------------|--------|
| 1 | `classify` | Route by subject type (person / product / animal / …) | No-op by design |
| 2 | `segment`  | Coarse foreground probability map, BiRefNet @1024px | Verified |
| 3 | `trimap`   | Derive the unknown band from model confidence | Implemented |
| 4 | `matte`    | Solve continuous α in the band (closed-form) | Implemented |
| 5 | `decontaminate` | Estimate uncontaminated foreground colour | Implemented |
| 6 | `composite` | Guided-filter upsample to full res, crop, encode | Implemented |

Full write-up, including why stages 4 and 5 matter: **[docs/pipeline.md](docs/pipeline.md)**.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
```

That is enough to run the tests. To actually cut out an image you need stage-2
weights, which are **not** bundled — export them once:

```bash
uv pip install -e ".[torch]"                       # heavy, ~2.5 GB
python scripts/export_onnx.py --model birefnet-general
```

Runtime deps stay light on purpose: the shipping path is ONNX, which is
CPU-viable and is the same graph the browser build will consume. `torch` is
only needed for the reference implementation and for this export.

## Usage

```bash
bgremover in.jpg out.png                       # full pipeline
bgremover in.jpg out.png --edge naive          # coarse mask only, for comparison
bgremover in.jpg out.png --matte matte.png     # also write the alpha plane
bgremover in.jpg out.png --bg '#ffffff' --crop # flatten onto white, crop to subject
bgremover batch ./photos ./cutouts             # a whole folder
bgremover models                               # what's registered, and its licence
```

The `--edge` ladder is the point. Run the same image through all four and the
difference between a toy and a product is visible immediately:

| Mode | Stages | Cost | Looks like |
|------|--------|------|-----------|
| `naive` | 2 | free | hard edges, halos on hair |
| `decontaminate` | 2, 5 | negligible | mask edges, but no colour fringe |
| `matte` | 2–5 | slow | real hair, fur, lace **(default)** |
| `refine` | 2–6 | slower | + boundary re-solved at native resolution |

```python
from bgremover import load, remove_background, save, save_alpha

cutout = remove_background(load("in.jpg"), edge_mode="matte")
save(cutout, "out.png")
save_alpha(cutout, "matte.png")   # the plane every compositor wants
```

`Cutout` is deliberately unflattened — source pixels, alpha, and the
decontaminated foreground stay separate planes, because that is what anyone
compositing in Photoshop, Affinity, Blender or Nuke actually needs.

## Evaluate before you optimise

Model swaps are a vibe check without a scored set of hard images. The manifest
of cases that matter — frizzy hair on a busy background, chain-link fence, wine
glass, black cat on a dark sofa, white on white — is in
[`eval/cases.toml`](eval/cases.toml), with the scoring rubric and harness in
[`eval/`](eval/README.md). Populate `eval/images/` with your own photos; they
are gitignored.

Note that newer is not automatically better: BiRefNet-HR is known to
over-correct on some inputs. Score it, don't assume.

## Build order

1. **Baseline + eval set.** Score the existing tools on your own hard images
   first, so there is something to beat. ← *you are here*
2. **The pipeline, MIT-only.** Stages 2–6 as this package. Mostly done; stage 2
   needs its first real run.
3. **Browser build.** Export ONNX, run it through `onnxruntime-web` on WebGPU
   with a WASM fallback. Zero marginal cost per image, and images never leave
   the device. See [`web/`](web/README.md).
4. **High-res container.** Same pipeline behind FastAPI, tiled stage-6
   reprojection, batch endpoint. Publish the image; do **not** run it as a free
   public GPU service.
5. **Routing.** Stage 1, last — once there is an eval set and more than one
   engine to route between. Must ship with a manual override.

## Licence

This project is MIT (see [LICENSE](LICENSE)). **No model weights are bundled.**
Weights are fetched at runtime and carry their own licences — all of them
permissive, all of them recorded in [NOTICE](NOTICE), which is the
authoritative record of what is safe to ship.
