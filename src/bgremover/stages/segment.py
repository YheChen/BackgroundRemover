"""Stage 2 — coarse segmentation (dichotomous image segmentation).

An encoder-decoder produces a foreground probability map at a fixed working
resolution (1024x1024 for BiRefNet general). This is the stage everyone means
when they say "AI background removal", and it is the *easy* stage: it gets the
silhouette right and the boundary wrong.

Two backends:

  onnx   the shipping path. CPU-viable, and the same graph the browser build
         consumes via onnxruntime-web. Requires scripts/export_onnx.py to
         have been run once.
  torch  the reference path. Loads straight from Hugging Face. Heavier, needs
         the [torch] extra, and is what you compare against when the ONNX
         export looks wrong.

NOTE: neither backend has been executed yet — the scaffold does not download
the ~900 MB of weights. Expect to fix the pre/post-processing details on
first run, particularly the normalisation constants and output tensor layout.
"""

from __future__ import annotations

import functools
from typing import Literal

import numpy as np
from PIL import Image

from .. import models
from ..types import Subject

Backend = Literal["onnx", "torch"]

# ImageNet normalisation — what BiRefNet's Swin backbone was trained with.
# Verify against the upstream repo's inference notebook on first run.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def probability_map(
    image: np.ndarray,
    *,
    model: str = models.DEFAULT_MODEL,
    backend: Backend = "onnx",
    subject: Subject = Subject.AUTO,
) -> np.ndarray:
    """Run stage 2 and return a (H, W) float32 probability map at input size.

    The returned map is at the *source* resolution — resampled up from the
    model's working resolution. Stage 3 consumes it directly; stage 6 handles
    the edge-aware reprojection separately.
    """
    spec = models.get(model)
    h, w = image.shape[:2]

    tensor = _preprocess(image, spec.input_size)
    if backend == "onnx":
        logits = _run_onnx(tensor, model)
    elif backend == "torch":
        logits = _run_torch(tensor, spec.hf_repo)
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'onnx' or 'torch'")

    prob = _sigmoid(logits)
    return _resize_map(prob, (w, h))


def _preprocess(image: np.ndarray, size: int) -> np.ndarray:
    """RGB uint8 (H, W, 3) -> normalised NCHW float32 (1, 3, size, size)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got shape {image.shape}")
    resized = np.asarray(
        Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    normalised = (resized - _MEAN) / _STD
    return normalised.transpose(2, 0, 1)[None].astype(np.float32)


@functools.lru_cache(maxsize=2)
def _onnx_session(model: str):
    """Cached ORT session. Two resident models max — they are ~1 GB each."""
    import onnxruntime as ort

    path = models.onnx_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"no ONNX graph at {path}.\n"
            f"Export it once with:  python scripts/export_onnx.py --model {model}\n"
            f"or run with --backend torch to use the reference path."
        )
    providers = [
        p
        for p in ("CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        if p in ort.get_available_providers()
    ]
    return ort.InferenceSession(str(path), providers=providers)


def _run_onnx(tensor: np.ndarray, model: str) -> np.ndarray:
    session = _onnx_session(model)
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: tensor})
    # BiRefNet emits a list of multi-scale side outputs; the last is the
    # finest. Verify this ordering when the export lands.
    return np.asarray(outputs[-1]).squeeze()


@functools.lru_cache(maxsize=1)
def _torch_model(hf_repo: str):
    import torch
    from transformers import AutoModelForImageSegmentation

    # BiRefNet ships custom modelling code, hence trust_remote_code.
    model = AutoModelForImageSegmentation.from_pretrained(
        hf_repo, trust_remote_code=True
    )
    model.eval()
    if torch.backends.mps.is_available():
        model.to("mps")
    elif torch.cuda.is_available():
        model.to("cuda").half()
    return model


def _run_torch(tensor: np.ndarray, hf_repo: str | None) -> np.ndarray:
    import torch

    if hf_repo is None:
        raise ValueError("this model has no Hugging Face repo for the torch backend")
    model = _torch_model(hf_repo)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        batch = torch.from_numpy(tensor).to(device=device, dtype=dtype)
        out = model(batch)
    preds = out[-1] if isinstance(out, (list, tuple)) else out
    if hasattr(preds, "logits"):
        preds = preds.logits
    if isinstance(preds, (list, tuple)):
        preds = preds[-1]
    return preds.float().cpu().numpy().squeeze()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid. Skipped if the graph already applied it."""
    if x.min() >= 0.0 and x.max() <= 1.0:
        return x.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))).astype(np.float32)


def _resize_map(prob: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resample a probability map to (w, h)."""
    if prob.shape[::-1] == size:
        return prob.astype(np.float32)
    img = Image.fromarray((prob * 255.0).astype(np.uint8), mode="L")
    resized = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0
