# BackgroundRemover

Free, open-source background removal. A replacement for remove.bg, whose API
sunsets 1 December 2026 (moving to Leonardo.Ai).

## The one hard rule

**No non-permissive model weights, ever.** `src/bgremover/models.py` refuses
anything outside `PERMISSIVE_LICENCES` at construction time, and
`tests/test_registry.py` enforces it.

In particular: **do not add BRIA RMBG-2.0 or RMBG-1.4.** They are CC BY-NC 4.0
(non-commercial), they are the default in several popular wrappers, and adding
them would make every downstream user non-compliant. [NOTICE](NOTICE) is the
authoritative record — update it whenever a model is added.

## Architecture

Six stages, in `src/bgremover/stages/`. `pipeline.py` orchestrates; `EdgeMode`
decides how far down the chain a call goes.

```
1 classify -> 2 segment -> 3 trimap -> 4 matte -> 5 decontaminate -> 6 composite
```

Read [docs/pipeline.md](docs/pipeline.md) before changing any stage. The short
version of what matters:

- Stage 2 alone is what makes free background removers look free. Never let
  `naive` become the default.
- Stages 4 and 5 are where the quality is. Stage 5 (decontamination) is the
  invisible half — users say "the edges look wrong" without knowing why.
- `Cutout` stays unflattened (source / alpha / foreground as separate planes).
  Do not collapse it to RGBA internally; compositors want the matte.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"   # enough to run tests
.venv/bin/python -m pytest -q
.venv/bin/ruff check .

uv pip install -e ".[torch]"            # heavy, only for the export below
python scripts/export_onnx.py --model birefnet-general
```

Runtime deps stay light on purpose: ONNX is the shipping path and the browser
build's input. torch is reference-only. Don't move torch into
`[project.dependencies]`.

## Status

- Stages 2–6: working and verified. The torch and ONNX backends agree to
  IoU 0.99969 on a real image. `bgremover in.jpg out.png` produces a cutout.
- Stage 1: intentional no-op. It is step 5 of the build order, not step 1 —
  routing between one engine is a coin flip.
- `EdgeMode.REFINE` is still an alias for `MATTE`; stage-6 local refinement
  is not implemented.

### Stage 2 facts you must not re-litigate

Each of these cost a debugging cycle. They are load-bearing:

- HF weights are **fp16**; CPU conv2d needs `.float()`.
- The model returns a **list of one** tensor of raw **logits**. Sigmoid is
  applied by us, unconditionally, for both backends.
- ONNX export needs **opset 22** — BiRefNet uses deformable convs and
  `DeformConv` only enters the standard schema at 22.
- **CoreML EP fails** on this graph (228 partitions, then an execution
  error). CPU is the default and the reliable path.
- Stage 3's `band_width` is quoted at a 1024px reference edge and **rescaled**
  to the actual image. Unscaled, it put 41% of a small image in the unknown
  band and haloed every edge.

## Conventions

- Comments explain *why*, especially where a choice looks arbitrary but
  encodes something about matting. Don't strip them.
- British spelling in prose and identifiers (`licence`, `colour`,
  `normalised`) — stay consistent with what's there.
- Errors name the likely cause and the fix, not just the symptom. See
  `matte.solve`'s band-fraction guard for the tone.
- Tests use the synthetic red-disc-on-green fixture in `tests/conftest.py`.
  Green on purpose: it gives stage 5 real contamination to fix.
- Never commit weights or eval images. Both are gitignored.
