"""Model registry.

Two rules this file enforces:

1. No weights ship with the repo. They are fetched on first use and cached.
2. Nothing lands here without a permissive licence recorded in NOTICE.

Rule 2 is load-bearing. The best-known model in this space (BRIA RMBG-2.0)
is CC BY-NC 4.0, it is the default in several popular wrappers, and adding it
would quietly make every downstream user non-compliant. `register()` refuses
anything not on the allow-list of licences.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PERMISSIVE_LICENCES = frozenset({"MIT", "Apache-2.0", "BSD-3-Clause", "CC0-1.0"})


class LicenceError(RuntimeError):
    """Raised when a model's licence is not shippable."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    licence: str
    hf_repo: str | None            # source of truth for the torch reference path
    onnx_filename: str | None      # produced by scripts/export_onnx.py
    input_size: int                # square inference resolution, px
    notes: str = ""

    def __post_init__(self) -> None:
        if self.licence not in PERMISSIVE_LICENCES:
            raise LicenceError(
                f"{self.key!r} is {self.licence}, which is not in the allow-list "
                f"{sorted(PERMISSIVE_LICENCES)}. See NOTICE for why this project "
                f"refuses non-permissive weights."
            )


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    _REGISTRY[spec.key] = spec
    return spec


# --- stage 2 segmentation models -------------------------------------------

register(
    ModelSpec(
        key="birefnet-general",
        licence="MIT",
        hf_repo="ZhengPeng7/BiRefNet",
        onnx_filename="birefnet-general.onnx",
        input_size=1024,
        notes="Default. Swin-L backbone, ~0.2B params. Trained on open datasets only.",
    )
)

register(
    ModelSpec(
        key="birefnet-lite",
        licence="MIT",
        hf_repo="ZhengPeng7/BiRefNet_lite",
        onnx_filename="birefnet-lite.onnx",
        input_size=1024,
        notes="Swin-Tiny, 44.4M params (~1/5 of general). The browser model.",
    )
)

register(
    ModelSpec(
        key="birefnet-hr",
        licence="MIT",
        hf_repo="ZhengPeng7/BiRefNet_HR",
        onnx_filename="birefnet-hr.onnx",
        input_size=2048,
        notes="2048px. Known to over-correct on some inputs — score it, don't assume.",
    )
)

register(
    ModelSpec(
        key="isnet-general-use",
        licence="Apache-2.0",
        hf_repo=None,
        onnx_filename="isnet-general-use.onnx",
        input_size=1024,
        notes="DIS IS-Net. No deformable convs -> 0.86s on WebGPU vs BiRefNet's 22.4s.",
    )
)

register(
    ModelSpec(
        key="ben2",
        licence="MIT",
        hf_repo="PramaLLC/BEN2",
        onnx_filename="ben2.onnx",
        input_size=1024,
        notes="Folds stages 3-4 in via Confidence Guided Matting. Second engine for hair.",
    )
)

DEFAULT_MODEL = "birefnet-general"


def get(key: str = DEFAULT_MODEL) -> ModelSpec:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown model {key!r}; available: {sorted(_REGISTRY)}"
        ) from None


def available() -> list[str]:
    return sorted(_REGISTRY)


def weights_dir() -> Path:
    """Where exported ONNX graphs live. Override with BGREMOVER_WEIGHTS."""
    env = os.environ.get("BGREMOVER_WEIGHTS")
    root = Path(env) if env else Path(__file__).resolve().parents[2] / "weights"
    root.mkdir(parents=True, exist_ok=True)
    return root


def onnx_path(key: str = DEFAULT_MODEL) -> Path:
    spec = get(key)
    if spec.onnx_filename is None:
        raise ValueError(f"{key!r} has no ONNX export configured")
    return weights_dir() / spec.onnx_filename
