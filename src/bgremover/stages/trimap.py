"""Stage 3 — uncertainty band derivation (pseudo-trimap).

Classical matting demanded a hand-painted trimap. We derive one.

Two sources of "unknown", unioned:

  a) Low model confidence. Wherever the segmentation probability is neither
     near 0 nor near 1, the model is telling us it doesn't know. This is the
     better signal and it is what BEN2 calls Confidence Guided Matting.
  b) A guaranteed minimum band around the mask boundary. A very confident
     model returns a razor-thin ambiguous region, but matting needs room to
     solve in. Without this, hair that the model called "definitely
     background" never gets a chance to come back.

Only the unknown band goes to stage 4, which is the expensive one, so band
width is the main quality/speed dial in the whole pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..types import TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN

REFERENCE_EDGE = 1024
"""Short edge, in px, that `band_width` is quoted against. See `derive`."""


def derive(
    prob: np.ndarray,
    *,
    fg_threshold: float = 0.95,
    bg_threshold: float = 0.05,
    band_width: int = 12,
    absolute_band: bool = False,
) -> np.ndarray:
    """Build a trimap from a segmentation probability map.

    Args:
        prob: (H, W) float in [0, 1] — stage 2 output at full working res.
        fg_threshold: at or above this, treat as definite foreground.
        bg_threshold: at or below this, treat as definite background.
        band_width: minimum unknown band either side of the boundary, in px
            **at a REFERENCE_EDGE short edge**, rescaled to the actual image.
            0 disables the morphological band and trusts confidence alone.
        absolute_band: take `band_width` as literal pixels, skipping the
            rescale. For callers that already know their working resolution.

    A fixed pixel band is wrong, and wrong in both directions: 12 px is a
    hairline on a 4000 px product shot and 12% of the frame on a 208 px
    thumbnail. Measured on a 208x242 image, an unscaled 12 px band put 41% of
    all pixels in the unknown region — stage 4 was being asked to re-segment
    the picture, and it answered by softening every edge into a halo. Scaled,
    the same call yields 8%.

    Returns:
        (H, W) uint8 trimap using TRIMAP_BG / TRIMAP_UNKNOWN / TRIMAP_FG.
    """
    if prob.ndim != 2:
        raise ValueError(f"prob must be 2-D (H, W), got shape {prob.shape}")
    if not 0.0 <= bg_threshold < fg_threshold <= 1.0:
        raise ValueError(
            f"need 0 <= bg_threshold ({bg_threshold}) < fg_threshold ({fg_threshold}) <= 1"
        )
    if band_width < 0:
        raise ValueError(f"band_width must be >= 0, got {band_width}")

    prob = np.clip(prob.astype(np.float32), 0.0, 1.0)

    definite_fg = prob >= fg_threshold
    definite_bg = prob <= bg_threshold

    # (a) confidence-driven unknown
    unknown = ~(definite_fg | definite_bg)

    # (b) morphological band around the boundary of the thresholded mask
    iterations = _scaled_band(band_width, prob.shape, absolute_band)
    if iterations > 0:
        solid = prob >= 0.5
        outer = ndimage.binary_dilation(solid, iterations=iterations)
        inner = ndimage.binary_erosion(solid, iterations=iterations)
        unknown |= outer & ~inner

    trimap = np.full(prob.shape, TRIMAP_BG, dtype=np.uint8)
    trimap[definite_fg] = TRIMAP_FG
    trimap[unknown] = TRIMAP_UNKNOWN
    return trimap


def _scaled_band(band_width: int, shape: tuple[int, ...], absolute: bool) -> int:
    """Convert a REFERENCE_EDGE-relative band width to pixels for this image."""
    if band_width <= 0:
        return 0
    if absolute:
        return band_width
    short_edge = min(shape[:2])
    return max(1, round(band_width * short_edge / REFERENCE_EDGE))


def band_fraction(trimap: np.ndarray) -> float:
    """Share of pixels in the unknown band. The stage-4 cost driver.

    Useful as a guard: a band over ~0.35 usually means stage 2 failed and
    matting is about to be asked to segment the image from scratch, which it
    cannot do.
    """
    return float(np.count_nonzero(trimap == TRIMAP_UNKNOWN) / trimap.size)


def to_pymatting(trimap: np.ndarray) -> np.ndarray:
    """Convert to pymatting's convention: float64, 0 = bg, 1 = fg, else unknown."""
    out = np.zeros(trimap.shape, dtype=np.float64)
    out[trimap == TRIMAP_FG] = 1.0
    out[trimap == TRIMAP_UNKNOWN] = 0.5
    return out
