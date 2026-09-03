"""Stage 1 — subject classification and routing.

Deliberately last in the build order, not first. Routing is only worth adding
once there is an eval set and more than one engine to route between;
otherwise it is a coin flip dressed up as intelligence.

When it lands, it must ship with a manual override. remove.bg exposes a
`type` parameter for exactly this reason: their classifier is sometimes wrong
and callers need a way to say so.
"""

from __future__ import annotations

import numpy as np

from ..types import Subject


def classify(image: np.ndarray) -> Subject:  # noqa: ARG001
    """Guess the subject type. Currently a no-op that always defers.

    Returning AUTO means "use the general model", which is the correct
    behaviour until a real classifier is trained and scored.
    """
    return Subject.AUTO


def route(subject: Subject) -> str:
    """Map a subject to a model key. One entry until step 5 of the build order."""
    from .. import models

    return models.DEFAULT_MODEL
