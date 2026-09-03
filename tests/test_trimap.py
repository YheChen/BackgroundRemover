import numpy as np
import pytest

from bgremover.stages import trimap as trimap_mod
from bgremover.types import TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN


def test_partitions_into_three_regions(prob):
    tri = trimap_mod.derive(prob)
    assert set(np.unique(tri)) <= {TRIMAP_BG, TRIMAP_UNKNOWN, TRIMAP_FG}
    for label in (TRIMAP_BG, TRIMAP_UNKNOWN, TRIMAP_FG):
        assert np.any(tri == label), f"no {label} pixels"


def test_confident_pixels_are_not_unknown(prob):
    tri = trimap_mod.derive(prob, fg_threshold=0.95, bg_threshold=0.05, band_width=0)
    assert np.all(tri[prob >= 0.95] == TRIMAP_FG)
    assert np.all(tri[prob <= 0.05] == TRIMAP_BG)


def test_band_width_widens_the_unknown_region(prob):
    narrow = trimap_mod.band_fraction(trimap_mod.derive(prob, band_width=2))
    wide = trimap_mod.band_fraction(trimap_mod.derive(prob, band_width=16))
    assert wide > narrow


def test_band_width_zero_trusts_confidence_alone(prob):
    tri = trimap_mod.derive(prob, band_width=0)
    unknown = (prob > 0.05) & (prob < 0.95)
    assert np.array_equal(tri == TRIMAP_UNKNOWN, unknown)


def test_pymatting_conversion_uses_expected_sentinels(prob):
    conv = trimap_mod.to_pymatting(trimap_mod.derive(prob))
    assert conv.dtype == np.float64
    assert set(np.unique(conv)) <= {0.0, 0.5, 1.0}


def test_rejects_bad_thresholds(prob):
    with pytest.raises(ValueError, match="bg_threshold"):
        trimap_mod.derive(prob, fg_threshold=0.1, bg_threshold=0.9)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        trimap_mod.derive(np.zeros((8, 8, 3), np.float32))


def test_band_width_scales_with_image_size():
    """A fixed pixel band is 12% of a thumbnail and a hairline on a 4K photo."""
    small = np.zeros((208, 242), np.float32)
    small[60:150, 70:170] = 1.0
    large = np.zeros((2080, 2420), np.float32)
    large[600:1500, 700:1700] = 1.0

    small_band = trimap_mod.band_fraction(trimap_mod.derive(small, band_width=12))
    large_band = trimap_mod.band_fraction(trimap_mod.derive(large, band_width=12))

    # Same call, same geometry, 10x the pixels -> comparable band fraction.
    assert abs(small_band - large_band) < 0.05, (small_band, large_band)
    # And the regression this guards: an unscaled 12px band put 41% of a
    # 208x242 image in the unknown region.
    assert small_band < 0.20, small_band


def test_absolute_band_opts_out_of_scaling():
    small = np.zeros((208, 242), np.float32)
    small[60:150, 70:170] = 1.0
    scaled = trimap_mod.band_fraction(trimap_mod.derive(small, band_width=12))
    literal = trimap_mod.band_fraction(
        trimap_mod.derive(small, band_width=12, absolute_band=True)
    )
    assert literal > scaled


def test_scaled_band_never_rounds_a_requested_band_to_zero():
    """A tiny image must still get a 1px band, not silently lose stage 4."""
    tiny = np.zeros((32, 32), np.float32)
    tiny[10:22, 10:22] = 1.0
    assert trimap_mod._scaled_band(12, tiny.shape, False) == 1
