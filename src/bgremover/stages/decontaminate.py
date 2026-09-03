"""Stage 5 — foreground colour estimation ("decontamination").

A semi-transparent pixel's observed colour is a blend of foreground and
whatever was behind it. Cut a person out of a green room and their hair edges
stay green. So the observed colour must be *replaced* with an estimate of the
uncontaminated foreground colour, for every pixel where alpha < 1.

This is what Photoshop calls "Decontaminate Colors" and Nuke calls
decontamination. It is the invisible half of cutout quality: users describe an
un-decontaminated result as "the edges look wrong" without being able to say
why.

Method: Fast Multi-Level Foreground Estimation (Germer et al., 2020), which
ships in pymatting. Cheap enough to run unconditionally.
"""

from __future__ import annotations

import numpy as np


def estimate(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Estimate the true foreground colour given the source and its matte.

    Args:
        image: (H, W, 3) uint8 or float — source pixels.
        alpha: (H, W) float in [0, 1] — the matte from stage 4 (or a
            thresholded mask, for EdgeMode.DECONTAMINATE).

    Returns:
        (H, W, 3) uint8 foreground colour estimate.
    """
    from pymatting import estimate_foreground_ml

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got shape {image.shape}")
    if image.shape[:2] != alpha.shape[:2]:
        raise ValueError(
            f"image {image.shape[:2]} and alpha {alpha.shape[:2]} must match"
        )

    img = image.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    fg = estimate_foreground_ml(img, alpha.astype(np.float64), return_background=False)
    return (np.clip(fg, 0.0, 1.0) * 255.0).round().astype(np.uint8)
