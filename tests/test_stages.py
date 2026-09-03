"""Stages 4, 5 and 6 on a synthetic scene.

Stage 2 is not covered here — it needs ~900 MB of weights. See
tests/test_pipeline_integration.py, which is skipped unless they are present.
"""

import numpy as np

from bgremover.stages import composite as composite_stage
from bgremover.stages import decontaminate, matte
from bgremover.stages import trimap as trimap_mod
from bgremover.types import Cutout


def test_matte_recovers_a_soft_edge(synthetic, prob):
    image, truth = synthetic
    solved = matte.solve(image, trimap_mod.derive(prob, band_width=8))

    assert solved.shape == truth.shape
    assert solved.min() >= 0.0 and solved.max() <= 1.0
    # A soft edge means intermediate alpha actually exists.
    assert np.any((solved > 0.2) & (solved < 0.8))
    assert np.abs(solved - truth).mean() < 0.1


def test_matte_refuses_an_implausible_band(synthetic):
    image, _ = synthetic
    everything_unknown = np.full(image.shape[:2], 128, np.uint8)
    try:
        matte.solve(image, everything_unknown, max_band_fraction=0.5)
    except ValueError as exc:
        assert "Stage 2" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a 100% unknown band")


def test_decontamination_removes_the_green_fringe(synthetic):
    image, alpha = synthetic
    fg = decontaminate.estimate(image, alpha)

    edge = (alpha > 0.25) & (alpha < 0.75)
    green_before = image[..., 1][edge].mean()
    green_after = fg[..., 1][edge].mean()
    # The fringe is green only because of the background it was blended with.
    assert green_after < green_before


def test_guided_upsample_matches_target_resolution(synthetic):
    image, alpha = synthetic
    big = np.asarray(
        np.repeat(np.repeat(image, 3, axis=0), 3, axis=1), dtype=np.uint8
    )
    coarse = alpha[::2, ::2]
    out = composite_stage.upsample_alpha(coarse, big)

    assert out.shape == big.shape[:2]
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_bounding_box_is_tight(synthetic):
    _, alpha = synthetic
    a8 = (alpha * 255).astype(np.uint8)
    left, top, right, bottom = composite_stage.bounding_box(a8)

    assert 0 < left < right < alpha.shape[1]
    assert 0 < top < bottom < alpha.shape[0]
    assert not a8[:top].any() and not a8[bottom:].any()


def test_bounding_box_of_empty_matte_is_the_full_frame():
    empty = np.zeros((40, 60), np.uint8)
    assert composite_stage.bounding_box(empty) == (0, 0, 60, 40)


def test_composite_onto_solid_colour(synthetic):
    image, alpha = synthetic
    cutout = Cutout(image=image, alpha=(alpha * 255).astype(np.uint8))
    flat = composite_stage.composite(cutout, (0, 0, 255))

    assert flat.shape == image.shape
    # Fully transparent corners take the new background exactly.
    assert tuple(flat[0, 0]) == (0, 0, 255)


def test_composite_none_gives_rgba(synthetic):
    image, alpha = synthetic
    cutout = Cutout(image=image, alpha=(alpha * 255).astype(np.uint8))
    out = composite_stage.composite(cutout, None)
    assert out.shape[2] == 4


def test_crop_to_subject_shrinks_the_frame(synthetic):
    image, alpha = synthetic
    cutout = Cutout(image=image, alpha=(alpha * 255).astype(np.uint8))
    cropped = composite_stage.crop_to_subject(cutout)
    assert cropped.alpha.shape[0] < alpha.shape[0]
    assert cropped.image.shape[:2] == cropped.alpha.shape[:2]
