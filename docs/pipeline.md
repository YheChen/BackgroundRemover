# The six-stage cutout pipeline

Notes on what a commercial background remover actually does between upload and
output, and where the quality really comes from.

## How we know remove.bg's shape

remove.bg never published its architecture, but its API surface leaks it:

- **A `type` parameter** accepting `person`, `product`, `car`, `animal`,
  `graphic`, `transportation`. You only expose that if a classifier routes to
  specialised weights, and only make it overridable because the classifier is
  sometimes wrong.
- **A ZIP output** containing `color.jpg` plus `alpha.png`, documented as the
  *fastest* format. That is a premultiplied colour plane and a separate 8-bit
  matte — the native internal representation, handed over before compositing.
- **A `semitransparency` flag** plus documented "algorithms for improving fine
  details and preventing colour contamination". Those are two distinct
  post-passes: alpha matting, then foreground colour estimation.

Every commercial cutout service is some variation of the same six stages.

## Stage 1 — subject classification

A cheap classifier decides what is in the frame, then picks weights tuned for
it. A portrait model that has seen 10,000 hairlines beats a general model on
hair; a product model biased toward closed convex shapes beats it on a shoe.

Deliberately the *last* thing we build. Routing between one engine is a coin
flip dressed up as intelligence.

## Stage 2 — coarse segmentation

Dichotomous image segmentation: an encoder–decoder emits a foreground
probability map at a fixed working resolution. Historically U²-Net; now
Swin- or ViT-backboned networks like BiRefNet.

This is the stage everyone means by "AI background removal", and it is the
*easy* one. It gets the silhouette right and the boundary wrong. Ship only
this and you have the hard-edged, halo-ed look of every free tool.

## Stage 3 — uncertainty band derivation

Classical matting demanded a hand-painted trimap; we derive one from two
signals, unioned:

1. **Low model confidence.** Where the probability is neither near 0 nor near
   1, the model is telling us it doesn't know. This is the better signal, and
   it is what BEN2 calls Confidence Guided Matting.
2. **A guaranteed minimum band** around the mask boundary. A very confident
   model returns a razor-thin ambiguous region, but matting needs room to
   solve in — without this, hair the model called "definitely background"
   never gets a chance to come back.

Only the band goes to stage 4, so band width is the main quality/speed dial
in the whole pipeline. A band over ~35% of the image means stage 2 failed and
matting is about to be asked to segment from scratch, which it cannot do.

## Stage 4 — alpha matting

Solve for continuous α ∈ [0,1] inside the band. The compositing equation is

```
I = α·F + (1−α)·B
```

and the whole game is recovering α (here) and F (stage 5) from the observed
pixel I. This is what makes hair, fur, lace, bicycle spokes, motion blur,
glass and smoke work: those pixels genuinely *are* part-foreground, and no
binary mask can represent them.

Our default solver is closed-form matting (Levin et al.) via `pymatting` —
MIT, CPU-only, no weights. A neural solver (ViTMatte) is better on hard cases
and belongs behind the same interface.

**Skip this stage and the tool is a toy.** It is the single largest quality
difference between a weekend wrapper and a service people pay for.

## Stage 5 — foreground colour estimation

A semi-transparent pixel's observed colour is contaminated by whatever was
behind it. Cut a person out of a green room and their hair edges stay green.
So the observed colour must be *replaced* with an estimate of the
uncontaminated foreground, for every pixel where α < 1.

Photoshop calls this "Decontaminate Colors"; Nuke calls it decontamination.
Method: Fast Multi-Level Foreground Estimation (Germer et al., 2020), which
ships in `pymatting` and is cheap enough to run unconditionally.

This is the invisible half of cutout quality. Users describe an
un-decontaminated result as "the edges look wrong" without being able to say
why — so get this in before polishing any UI.

It is also why remove.bg's fastest output is `color.jpg` + `alpha.png`: those
are exactly the two planes coming out of stage 5.

## Stage 6 — reprojection and compositing

The matte was computed at ~1–2 MP; the source may be up to 50 MP. Upsampling
α with plain bilinear throws away the edge detail stages 3–5 just paid for, so
α is upsampled *guided by* the full-resolution image — a guided filter (He et
al.) with full-res luminance as the guide, or a local refinement pass in the
style of CascadePSP.

Everything after is plumbing: auto-crop to the subject box, margins, rescale,
background colour or image, contact shadow, encode. But it is the plumbing
that separates a demo from a product — and above ~4 MP it becomes tiling,
seam handling and memory management, which is most of that distance.

## What stays hard

- **Semi-transparency is unsolved.** Glass, veils, smoke, motion blur and
  out-of-focus edges defeat every model including remove.bg. Decide
  explicitly whether you claim to handle it.
- **High-res is a memory problem, not a model problem.**
- **Cold start dominates perceived speed.** A 455 MB–1 GB model load on first
  use reads as "broken" without real progress reporting and hard caching.

## References

- BiRefNet — <https://github.com/ZhengPeng7/BiRefNet> (MIT)
- BEN2 — <https://huggingface.co/PramaLLC/BEN2> (MIT)
- PyMatting — <https://github.com/pymatting/pymatting> (MIT)
- Germer et al., *Fast Multi-Level Foreground Estimation* — <https://arxiv.org/abs/2006.14970>
- CascadePSP — <https://github.com/hkchengrex/CascadePSP> (MIT)
- withoutbg — <https://github.com/withoutbg/withoutbg> (Apache-2.0 + DINOv3)
- remove.bg API reference — <https://www.remove.bg/api>
