"""End-to-end, skipped unless stage-2 weights are actually present.

Run after `python scripts/export_onnx.py --model birefnet-general`.
"""

import numpy as np
import pytest

from bgremover import models

pytestmark = pytest.mark.skipif(
    not models.onnx_path(models.DEFAULT_MODEL).exists(),
    reason="stage-2 weights not exported; see scripts/export_onnx.py",
)


def test_pipeline_end_to_end(synthetic):
    from bgremover import remove_background

    image, _ = synthetic
    cutout = remove_background(image, edge_mode="matte")

    assert cutout.alpha.shape == image.shape[:2]
    assert cutout.foreground is not None
    assert cutout.to_rgba().shape == (*image.shape[:2], 4)


def test_naive_mode_skips_stage_five(synthetic):
    from bgremover import remove_background

    image, _ = synthetic
    cutout = remove_background(image, edge_mode="naive")
    assert cutout.foreground is None
    assert set(np.unique(cutout.alpha)) <= {0, 255}
