"""Stage 4 — alpha matting.

Solve for continuous alpha in [0, 1] inside the unknown band. This is what
makes hair, fur, lace, bicycle spokes, motion blur, glass and smoke work:
those pixels genuinely *are* part-foreground, and no binary mask can
represent them.

The compositing equation is  I = a*F + (1-a)*B  and the whole game is
recovering a (here) and F (stage 5) from the observed pixel I.

Default solver is closed-form matting (Levin et al.) via pymatting: MIT,
CPU-only, no weights, no licence questions. A neural solver (ViTMatte) is
better on hard cases and is left as a pluggable backend.
"""

from __future__ import annotations

import numpy as np

from ..types import TRIMAP_FG
from . import trimap as trimap_mod


def solve(
    image: np.ndarray,
    trimap: np.ndarray,
    *,
    max_band_fraction: float = 0.5,
) -> np.ndarray:
    """Estimate alpha for the unknown band of `trimap`.

    Args:
        image: (H, W, 3) uint8 or float — the source pixels at working res.
        trimap: (H, W) uint8 trimap from stage 3.
        max_band_fraction: refuse to solve if the band is larger than this.
            Closed-form matting on a huge band is slow and produces mush; it
            means stage 2 failed and should be fixed instead.

    Returns:
        (H, W) float32 alpha in [0, 1].
    """
    import functools

    from pymatting import estimate_alpha_cf, ichol

    if image.shape[:2] != trimap.shape[:2]:
        raise ValueError(
            f"image {image.shape[:2]} and trimap {trimap.shape[:2]} must match"
        )

    # Stage 2 found nothing. pymatting would raise "Trimap did not contain
    # foreground values", which reads like a bug in our trimap rather than
    # what it actually is: no subject in the picture. Say the real thing.
    if not np.any(trimap == TRIMAP_FG):
        raise ValueError(
            "no foreground found — stage 2 did not identify a subject in this "
            "image, so there is nothing to matte. Inspect the coarse mask "
            "(--edge naive) before touching matting parameters."
        )

    frac = trimap_mod.band_fraction(trimap)
    if frac > max_band_fraction:
        raise ValueError(
            f"unknown band covers {frac:.0%} of the image (limit {max_band_fraction:.0%}). "
            "Stage 2 probably failed — check the coarse mask before blaming the matte."
        )

    img = _as_float01(image)
    # pymatting's default preconditioner starts at shift=0, where the matting
    # Laplacian is routinely not positive-definite enough for an incomplete
    # Cholesky. It recovers by retrying with larger shifts, but prints a
    # PERFORMANCE WARNING and wastes the first factorisation attempt. Start at
    # a shift that actually works and the warning goes away with it.
    preconditioner = functools.partial(
        ichol, discard_threshold=1e-4, shifts=[1e-3, 1e-2, 1e-1, 0.5, 1.0]
    )
    alpha = estimate_alpha_cf(
        img, trimap_mod.to_pymatting(trimap), preconditioner=preconditioner
    )
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _as_float01(image: np.ndarray) -> np.ndarray:
    """pymatting wants float64 RGB in [0, 1]."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got shape {image.shape}")
    img = image.astype(np.float64)
    return img / 255.0 if img.max() > 1.0 else img
