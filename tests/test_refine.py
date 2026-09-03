"""Stage 6b — local high-resolution refinement.

The failure mode this guards is a visible tile seam in the alpha channel,
which is glaring once composited and invisible in a shape-only assertion.
"""

import numpy as np
import pytest

from bgremover.stages import refine


@pytest.fixture
def big_scene():
    """A 900x900 red disc on green with a soft edge — bigger than one tile."""
    n = 900
    yy, xx = np.mgrid[0:n, 0:n]
    alpha = np.clip((300.0 - np.hypot(yy - n / 2, xx - n / 2)) / 6.0, 0.0, 1.0)
    fg = np.zeros((n, n, 3), np.float32)
    fg[..., 0] = 220.0
    bg = np.zeros((n, n, 3), np.float32)
    bg[..., 1] = 200.0
    image = (fg * alpha[..., None] + bg * (1 - alpha[..., None])).astype(np.uint8)
    return image, alpha.astype(np.float32)


def test_refine_preserves_shape_and_range(big_scene):
    image, alpha = big_scene
    out = refine.refine(image, alpha, tile=256, overlap=32)
    assert out.shape == alpha.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_refine_leaves_no_tile_seams(big_scene):
    """A hard tile boundary shows up as a row/column of large gradient jumps."""
    image, alpha = big_scene
    out = refine.refine(image, alpha, tile=256, overlap=32)

    # Column-wise mean absolute horizontal gradient. A seam at a tile stride
    # would spike far above its neighbours.
    grad = np.abs(np.diff(out, axis=1)).mean(axis=0)
    interior = grad[10:-10]
    assert interior.max() < interior.mean() + 12 * interior.std() + 0.05, (
        f"suspected seam: max {interior.max():.4f} vs mean {interior.mean():.4f}"
    )


def test_refine_recovers_the_soft_edge(big_scene):
    image, alpha = big_scene
    # Degrade the matte the way a low-res upsample would: blur it out.
    from scipy.ndimage import uniform_filter

    degraded = uniform_filter(alpha, size=21).astype(np.float32)
    out = refine.refine(image, degraded, tile=256, overlap=32, band_width=16)

    assert np.abs(out - alpha).mean() < np.abs(degraded - alpha).mean()


def test_uniform_alpha_is_returned_untouched():
    """No unknown band means nothing to solve — return the input, not mush."""
    image = np.zeros((300, 300, 3), np.uint8)
    solid = np.ones((300, 300), np.float32)
    assert np.array_equal(refine.refine(image, solid), solid)


def test_skipped_tiles_keep_their_input_alpha(big_scene):
    """Coverage is never worse than the input: untouched pixels are preserved."""
    image, alpha = big_scene
    out = refine.refine(image, alpha, tile=256, overlap=32)
    # The disc centre is far from any boundary tile's unknown band.
    assert out[450, 450] == pytest.approx(alpha[450, 450], abs=1e-6)


def test_rejects_overlap_at_or_above_half_the_tile(big_scene):
    image, alpha = big_scene
    with pytest.raises(ValueError, match="under half the tile"):
        refine.refine(image, alpha, tile=128, overlap=64)


def test_max_tiles_guard_names_the_real_cause(big_scene):
    image, alpha = big_scene
    with pytest.raises(ValueError, match="fragmented"):
        refine.refine(image, alpha, tile=64, overlap=8, max_tiles=4)


def test_window_is_strictly_positive():
    """Zero anywhere in the window would divide by zero in the blend."""
    w = refine._window(200, 200, 32)
    assert w.min() > 0.0
    assert w.max() == pytest.approx(1.0)


def test_tile_starts_cover_the_full_axis():
    starts = refine._starts(900, 256, 224)
    assert starts[0] == 0
    assert starts[-1] + 256 == 900, "last tile must sit flush with the edge"


def test_single_tile_axis_needs_no_tiling():
    assert refine._starts(100, 256, 224) == [0]
