"""Stage 6b — local high-resolution refinement.

Stages 2-5 run on a downscaled copy (see pipeline.WORKING_PIXELS) and stage 6
guided-upsamples the resulting matte back to full size. That upsample is
edge-aware, but it cannot invent detail the matte never had: a 50 MP photo
matted at 2048x2048 has its hair solved at roughly 1/25th of the available
resolution, and no filter recovers that afterwards.

This pass re-solves the matte at *native* resolution, in tiles, along the
boundary only. It is the difference the docs mean by "above ~4 MP it becomes
tiling, seam handling and memory management".

Why it is affordable: almost every tile in a large image is entirely inside
the subject or entirely outside it. Only tiles straddling the boundary carry
unknown-band pixels, and only those get solved. On a typical portrait that is
a few dozen tiles out of a few hundred.

Two details that matter:

  seams     Tiles overlap and are blended with a feathered window, so no
            tile edge appears in the output. A hard tile boundary in an
            alpha channel is glaringly visible once composited.
  context   A tile with no definite-foreground or no definite-background
            pixels is unsolvable -- matting needs both to propagate from.
            Those tiles are skipped and keep the upsampled alpha.
"""

from __future__ import annotations

import numpy as np

from ..types import TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN
from . import matte as matte_stage
from . import trimap as trimap_mod

DEFAULT_TILE = 512
DEFAULT_OVERLAP = 64


def refine(
    image: np.ndarray,
    alpha: np.ndarray,
    *,
    tile: int = DEFAULT_TILE,
    overlap: int = DEFAULT_OVERLAP,
    band_width: int = 12,
    max_tiles: int = 400,
) -> np.ndarray:
    """Re-solve `alpha` at full resolution along the subject boundary.

    Args:
        image: (H, W, 3) uint8 source at native resolution.
        alpha: (H, W) float in [0, 1] — the upsampled matte from stage 6.
        tile: tile side in px. Larger gives matting more context and costs
            more; 512 is a reasonable balance.
        overlap: feather margin between neighbouring tiles, in px.
        band_width: unknown band for the per-tile trimap, at the stage-3
            reference edge. Applied with `absolute_band` because the tile is
            already at native resolution.
        max_tiles: refuse beyond this many solvable tiles. A boundary that
            fragmented usually means stage 2 produced noise, and refining
            noise is expensive and pointless.

    Returns:
        (H, W) float32 refined alpha. Pixels in skipped tiles keep their
        input value, so the result is never worse-covered than the input.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got shape {image.shape}")
    if image.shape[:2] != alpha.shape[:2]:
        raise ValueError(
            f"image {image.shape[:2]} and alpha {alpha.shape[:2]} must match"
        )
    if overlap * 2 >= tile:
        raise ValueError(f"overlap ({overlap}) must be under half the tile ({tile})")

    H, W = alpha.shape[:2]
    alpha = alpha.astype(np.float32)

    # Full-resolution trimap. absolute_band: we are already at native scale,
    # so stage 3's reference-edge rescaling would inflate the band by the
    # image's own size, which is exactly what we do not want here.
    full_trimap = trimap_mod.derive(
        alpha, band_width=band_width, absolute_band=True
    )
    if not np.any(full_trimap == TRIMAP_UNKNOWN):
        return alpha  # nothing ambiguous left to solve

    stride = tile - overlap
    jobs = [
        (y0, x0)
        for y0 in _starts(H, tile, stride)
        for x0 in _starts(W, tile, stride)
        if _solvable(full_trimap[y0 : y0 + tile, x0 : x0 + tile])
    ]
    if not jobs:
        return alpha
    if len(jobs) > max_tiles:
        raise ValueError(
            f"{len(jobs)} boundary tiles exceeds max_tiles={max_tiles}. The "
            "coarse mask is probably fragmented — inspect it with --edge naive "
            "rather than paying to refine noise."
        )

    numer = np.zeros((H, W), np.float32)
    denom = np.zeros((H, W), np.float32)

    for y0, x0 in jobs:
        y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
        sub_img = image[y0:y1, x0:x1]
        sub_tri = full_trimap[y0:y1, x0:x1]

        # max_band_fraction is a whole-image check on whether stage 2 failed.
        # Per tile it is meaningless: these tiles are chosen *because* they
        # straddle the boundary, so a high band fraction is the normal case.
        solved = matte_stage.solve(sub_img, sub_tri, max_band_fraction=1.0)

        w = _window(y1 - y0, x1 - x0, overlap)
        numer[y0:y1, x0:x1] += solved * w
        denom[y0:y1, x0:x1] += w

    covered = denom > 0
    out = alpha.copy()
    out[covered] = numer[covered] / denom[covered]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _starts(total: int, tile: int, stride: int) -> list[int]:
    """Tile origins along one axis, with the last tile flush to the edge."""
    if total <= tile:
        return [0]
    xs = list(range(0, total - tile + 1, stride))
    if xs[-1] + tile < total:
        xs.append(total - tile)
    return xs


def _solvable(tile_trimap: np.ndarray) -> bool:
    """A tile is worth solving only if it is ambiguous AND has both anchors.

    Matting propagates from known foreground and known background. A tile
    holding only unknown pixels has nothing to propagate from and pymatting
    would either refuse it or return mush.
    """
    return bool(
        np.any(tile_trimap == TRIMAP_UNKNOWN)
        and np.any(tile_trimap == TRIMAP_FG)
        and np.any(tile_trimap == TRIMAP_BG)
    )


def _window(h: int, w: int, overlap: int) -> np.ndarray:
    """Separable feather window, strictly positive so denom never divides by 0."""
    return np.outer(_ramp(h, overlap), _ramp(w, overlap)).astype(np.float32)


def _ramp(n: int, overlap: int) -> np.ndarray:
    r = np.ones(n, np.float32)
    m = min(overlap, n // 2)
    if m > 0:
        # Exclude the endpoints so the window never actually reaches zero.
        edge = np.linspace(0.0, 1.0, m + 2, dtype=np.float32)[1:-1]
        r[:m] = edge
        r[-m:] = edge[::-1]
    return r
