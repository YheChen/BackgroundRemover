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
