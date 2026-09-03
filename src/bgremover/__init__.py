"""bgremover — a full six-stage background removal pipeline, MIT weights only.

    from bgremover import remove_background, load, save
    cutout = remove_background(load("in.jpg"))
    save(cutout, "out.png")

The six stages, and why the last two matter, are documented in docs/pipeline.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .models import DEFAULT_MODEL, available
from .pipeline import remove_background
from .types import Cutout, EdgeMode, Subject

__version__ = "0.0.1"

__all__ = [
    "Cutout",
    "DEFAULT_MODEL",
    "EdgeMode",
    "Subject",
    "available",
    "load",
    "remove_background",
    "save",
    "save_alpha",
]


def load(path: str | Path) -> np.ndarray:
    """Read an image as (H, W, 3) uint8 RGB, dropping any existing alpha."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def save(cutout: Cutout, path: str | Path) -> None:
    """Write straight RGBA. Use .png or .webp — .jpg cannot hold alpha."""
    Image.fromarray(cutout.to_rgba(), mode="RGBA").save(path)


def save_alpha(cutout: Cutout, path: str | Path) -> None:
    """Write the matte on its own, as 8-bit greyscale.

    The output almost no free tool gives you, and the one anyone compositing
    in Photoshop, Affinity, Blender or Nuke actually wants.
    """
    Image.fromarray(cutout.alpha, mode="L").save(path)
