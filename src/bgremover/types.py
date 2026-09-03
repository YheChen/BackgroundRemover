"""Shared types for the cutout pipeline.

The pipeline moves three things between stages: the source image, a
single-channel alpha plane, and (from stage 5 onward) an estimated
foreground colour plane that is *not* the same as the source pixels.

Keeping foreground colour separate from the source is the whole point of
stage 5. A semi-transparent pixel's observed colour is contaminated by
whatever was behind it, so compositing the source pixels against a new
background reproduces the old background's colour in the edge. That is the
green fringe you see on cheap cutouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EdgeMode(str, Enum):
    """How much of the pipeline to run past the coarse mask.

    The quality ladder, cheapest first. `NAIVE` is what most open-source
    demos ship and is the source of hard, halo-ed hair edges.
    """

    NAIVE = "naive"                  # stage 2 only: threshold the mask
    DECONTAMINATE = "decontaminate"  # + stage 5: fix fringe colour, keep mask edges
    MATTE = "matte"                  # + stages 3-5: real continuous alpha
    REFINE = "refine"                # + stage 6 local refinement pass


class Subject(str, Enum):
    """Stage 1 routing labels.

    Mirrors remove.bg's `type` parameter, which exists because their
    classifier is sometimes wrong and callers need an override. Ours is
    AUTO-only until step 5 of the build order; the enum is here so stage 2
    can already accept and thread it through.
    """

    AUTO = "auto"
    PERSON = "person"
    PRODUCT = "product"
    ANIMAL = "animal"
    CAR = "car"
    GRAPHIC = "graphic"


# Trimap encoding. uint8 so it round-trips through PNG for debugging.
TRIMAP_BG = 0
TRIMAP_UNKNOWN = 128
TRIMAP_FG = 255


@dataclass(slots=True)
class Cutout:
    """The pipeline's output, kept unflattened.

    `image` is the original RGB. `alpha` is the matte. `foreground` is the
    decontaminated colour estimate, or None if stage 5 did not run.

    Deliberately not a flattened RGBA PNG: anyone compositing in Photoshop,
    Affinity, Blender or Nuke wants the matte as a separate plane, and almost
    no free tool hands it over. `to_rgba()` is available when you do want the
    flattened thing.
    """

    image: np.ndarray                 # (H, W, 3) uint8 — source pixels
    alpha: np.ndarray                 # (H, W)    uint8 — the matte
    foreground: np.ndarray | None = None   # (H, W, 3) uint8 — stage 5 output
    trimap: np.ndarray | None = None       # (H, W)    uint8 — stage 3, for debugging
    subject: Subject = Subject.AUTO
    edge_mode: EdgeMode = EdgeMode.MATTE

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.alpha.shape[:2]
        return w, h

    def colour_plane(self) -> np.ndarray:
        """Foreground estimate if stage 5 ran, else the raw source pixels."""
        return self.image if self.foreground is None else self.foreground

    def to_rgba(self) -> np.ndarray:
        """Flatten to straight (non-premultiplied) RGBA."""
        return np.dstack([self.colour_plane(), self.alpha]).astype(np.uint8)
