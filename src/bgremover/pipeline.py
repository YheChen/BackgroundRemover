"""The pipeline orchestrator.

    classify -> segment -> trimap -> matte -> decontaminate -> composite

`EdgeMode` decides how far down that chain a call goes. The default is
MATTE, because NAIVE is what makes free background removers look free.
"""

from __future__ import annotations

import numpy as np

from . import models
from .stages import classify as classify_stage
from .stages import composite as composite_stage
from .stages import decontaminate as decontaminate_stage
from .stages import matte as matte_stage
from .stages import segment as segment_stage
from .stages import trimap as trimap_stage
from .types import Cutout, EdgeMode, Subject

# Above this many pixels, stages 2-5 run on a downscaled copy and stage 6
# reprojects. Matting a 50 MP image directly is a memory problem, not a
# quality win.
WORKING_PIXELS = 2_048 * 2_048


def remove_background(
    image: np.ndarray,
    *,
    edge_mode: EdgeMode | str = EdgeMode.MATTE,
    model: str = models.DEFAULT_MODEL,
    backend: str = "onnx",
    subject: Subject | str = Subject.AUTO,
    band_width: int = 12,
    keep_trimap: bool = False,
) -> Cutout:
    """Run the pipeline and return an unflattened Cutout.

    Args:
        image: (H, W, 3) uint8 RGB.
        edge_mode: how far past the coarse mask to go. See EdgeMode.
        model: registry key, e.g. "birefnet-general".
        backend: "onnx" (shipping) or "torch" (reference).
        subject: routing hint; AUTO defers to stage 1.
        band_width: stage-3 unknown band in px. The main quality/speed dial.
        keep_trimap: attach the trimap to the result for debugging.
    """
    edge_mode = EdgeMode(edge_mode)
    subject = Subject(subject)
    _validate_rgb(image)

    if subject is Subject.AUTO:
        subject = classify_stage.classify(image)

    full_h, full_w = image.shape[:2]
    working = _downscale_for_working(image)

    # --- stage 2 -----------------------------------------------------------
    prob = segment_stage.probability_map(
        working, model=model, backend=backend, subject=subject
    )

    if edge_mode is EdgeMode.NAIVE:
        alpha_w = (prob >= 0.5).astype(np.float32)
        trimap = None
    else:
        # --- stage 3 -------------------------------------------------------
        trimap = trimap_stage.derive(prob, band_width=band_width)
        if edge_mode is EdgeMode.DECONTAMINATE:
            # Keep the mask's own edges, just fix the fringe colour. Cheap.
            alpha_w = prob.astype(np.float32)
        else:
            # --- stage 4 ---------------------------------------------------
            alpha_w = matte_stage.solve(working, trimap)

    # --- stage 6a: reproject alpha back to full resolution -----------------
    alpha = (
        alpha_w
        if working.shape[:2] == (full_h, full_w)
        else composite_stage.upsample_alpha(alpha_w, image)
    )

    # --- stage 5 -----------------------------------------------------------
    foreground = None
    if edge_mode is not EdgeMode.NAIVE:
        foreground = decontaminate_stage.estimate(image, alpha)

    return Cutout(
        image=image,
        alpha=(alpha * 255.0).round().clip(0, 255).astype(np.uint8),
        foreground=foreground,
        trimap=_upsample_trimap(trimap, full_w, full_h) if keep_trimap else None,
        subject=subject,
        edge_mode=edge_mode,
    )


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"expected (H, W, 3) RGB uint8, got shape {image.shape}. "
            "Drop the alpha channel or convert from greyscale first."
        )
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {image.dtype}")


def _downscale_for_working(image: np.ndarray) -> np.ndarray:
    """Shrink to WORKING_PIXELS, preserving aspect. Stage 6 undoes this."""
    from PIL import Image

    h, w = image.shape[:2]
    if h * w <= WORKING_PIXELS:
        return image
    scale = (WORKING_PIXELS / (h * w)) ** 0.5
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))


def _upsample_trimap(
    trimap: np.ndarray | None, w: int, h: int
) -> np.ndarray | None:
    if trimap is None:
        return None
    from PIL import Image

    if trimap.shape[::-1] == (w, h):
        return trimap
    return np.asarray(
        Image.fromarray(trimap, mode="L").resize((w, h), Image.Resampling.NEAREST)
    )
