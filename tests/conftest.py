import numpy as np
import pytest


@pytest.fixture
def synthetic():
    """A 128x128 scene: red disc on a green field, with a soft edge.

    Green on purpose. It is the classic contamination case, so stage 5 has
    something real to fix and the tests can assert it did.
    """
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(yy - h / 2, xx - w / 2)

    alpha = np.clip((34.0 - dist) / 4.0, 0.0, 1.0).astype(np.float32)
    fg = np.zeros((h, w, 3), np.float32)
    fg[..., 0] = 220.0
    bg = np.zeros((h, w, 3), np.float32)
    bg[..., 1] = 200.0

    image = (fg * alpha[..., None] + bg * (1 - alpha[..., None])).astype(np.uint8)
    return image, alpha


@pytest.fixture
def prob(synthetic):
    """Stage-2-like probability map for the synthetic scene."""
    return synthetic[1]
