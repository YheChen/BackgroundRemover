"""Stage 6 — reprojection and compositing.

The matte was computed at ~1-2 MP; the source may be up to 50 MP. Upsampling
alpha with plain bilinear throws away the edge detail stages 3-5 just paid
for, so the alpha plane is upsampled *guided by* the full-resolution image:
a joint-bilateral / guided-filter step that snaps the matte back onto the real
edges.

Everything after that is plumbing — crop, margin, background, encode — but it
is the plumbing that makes the difference between a demo and a product.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..types import Cutout


def upsample_alpha(
    alpha: np.ndarray,
    image: np.ndarray,
    *,
    radius: int = 4,
    eps: float = 1e-4,
) -> np.ndarray:
    """Edge-aware upsample of `alpha` to `image`'s resolution.

    Guided filter (He et al.) with the full-res luminance as the guide. Falls
    back to bilinear when the sizes already match.

    Args:
        alpha: (h, w) float in [0, 1] at working resolution.
        image: (H, W, 3) uint8 source at full resolution.
        radius: box-filter radius, in full-res px.
        eps: regularisation. Larger = smoother, blurrier edges.

    Returns:
        (H, W) float32 alpha at full resolution.
    """
    H, W = image.shape[:2]
    coarse = _resize(alpha, (W, H))
    if alpha.shape[:2] == (H, W):
        return coarse

    guide = _luminance(image)
    mean_g = _box(guide, radius)
    mean_a = _box(coarse, radius)
    corr_gg = _box(guide * guide, radius)
    corr_ga = _box(guide * coarse, radius)

    var_g = corr_gg - mean_g * mean_g
    cov_ga = corr_ga - mean_g * mean_a

    a = cov_ga / (var_g + eps)
    b = mean_a - a * mean_g

    out = _box(a, radius) * guide + _box(b, radius)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def composite(
    cutout: Cutout,
    background: tuple[int, int, int] | np.ndarray | None = None,
) -> np.ndarray:
    """Flatten a cutout onto a background, or to straight RGBA if None."""
    if background is None:
        return cutout.to_rgba()

    fg = cutout.colour_plane().astype(np.float32)
    a = (cutout.alpha.astype(np.float32) / 255.0)[..., None]

    if isinstance(background, tuple):
        bg = np.empty_like(fg)
        bg[:] = np.asarray(background, dtype=np.float32)
    else:
        bg = _resize_rgb(background, fg.shape[1], fg.shape[0]).astype(np.float32)

    return (fg * a + bg * (1.0 - a)).round().clip(0, 255).astype(np.uint8)


def bounding_box(alpha: np.ndarray, threshold: int = 8) -> tuple[int, int, int, int]:
    """Tight box around the subject as (left, top, right, bottom), exclusive.

    Returns the full frame when the matte is empty, so callers never have to
    special-case a failed cutout.
    """
    solid = alpha > threshold
    rows = np.flatnonzero(solid.any(axis=1))
    cols = np.flatnonzero(solid.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return 0, 0, alpha.shape[1], alpha.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def crop_to_subject(
    cutout: Cutout, *, margin: float = 0.0, threshold: int = 8
) -> Cutout:
    """Crop every plane to the subject, with an optional fractional margin."""
    left, top, right, bottom = bounding_box(cutout.alpha, threshold)
    if margin > 0:
        pad_x = int(round((right - left) * margin))
        pad_y = int(round((bottom - top) * margin))
        h, w = cutout.alpha.shape[:2]
        left, top = max(0, left - pad_x), max(0, top - pad_y)
        right, bottom = min(w, right + pad_x), min(h, bottom + pad_y)

    box = (slice(top, bottom), slice(left, right))
    return Cutout(
        image=cutout.image[box],
        alpha=cutout.alpha[box],
        foreground=None if cutout.foreground is None else cutout.foreground[box],
        trimap=None if cutout.trimap is None else cutout.trimap[box],
        subject=cutout.subject,
        edge_mode=cutout.edge_mode,
    )


# --- helpers ---------------------------------------------------------------

def _luminance(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _box(x: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter over a (2r+1) square, via a summed-area table."""
    from scipy.ndimage import uniform_filter

    return uniform_filter(x, size=2 * radius + 1, mode="reflect")


def _resize(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if plane.shape[::-1] == size:
        return plane.astype(np.float32)
    img = Image.fromarray((np.clip(plane, 0, 1) * 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize(size, Image.Resampling.BICUBIC), np.float32) / 255.0


def _resize_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    if rgb.shape[:2] == (h, w):
        return rgb
    return np.asarray(
        Image.fromarray(rgb).resize((w, h), Image.Resampling.LANCZOS)
    )
