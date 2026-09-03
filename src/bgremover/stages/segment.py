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

Verified against BiRefNet (ZhengPeng7/BiRefNet, transformers 5.16, torch 2.14):

  - Preprocessing is 1024x1024 bilinear + ImageNet normalisation. Confirmed
    against upstream's own inference snippet.
  - The HF checkpoint ships **fp16 weights**. CPU conv2d refuses to mix fp16
    bias with fp32 input, so the model is cast to fp32 unless it lands on
    CUDA. This is the first thing that breaks if you touch _torch_model.
  - `model(x)` returns a *list of one* tensor, shape (1, 1, 1024, 1024),
    carrying raw **logits** (observed range roughly -28..19). Sigmoid is
    required; it is not baked into the graph.
"""

from __future__ import annotations

import functools
import os
from typing import Literal

import numpy as np
from PIL import Image

from .. import models
from ..types import Subject

Backend = Literal["onnx", "torch"]

# ImageNet normalisation — what BiRefNet's Swin backbone was trained with.
# Confirmed against upstream's own inference snippet; see the module docstring.
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
    # CoreML is deliberately NOT in the default list. On this graph it claims
    # 873 of 2869 nodes but fragments them into 228 partitions, then fails at
    # execution ("Unable to compute the prediction using a neural network
    # model"). Even working, 228 partition boundaries would erase any speedup.
    # CUDA is fine. CPU is the reliable path, and slow: one 1024x1024 pass
    # measured 7-13 s on a 10-core M-series laptop, run to run, with the
    # variance dominated by scheduling rather than anything we control. Treat
    # single-image CPU latency as "seconds, not sub-second" and do not quote a
    # tighter figure than that without re-measuring.
    # Override with BGREMOVER_PROVIDERS as a comma-separated list.
    override = os.environ.get("BGREMOVER_PROVIDERS")
    wanted = (
        [p.strip() for p in override.split(",") if p.strip()]
        if override
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    providers = [p for p in wanted if p in ort.get_available_providers()]
    if not providers:
        raise RuntimeError(
            f"none of {wanted} are available; onnxruntime offers "
            f"{ort.get_available_providers()}"
        )
    return ort.InferenceSession(str(path), providers=providers)


def _run_onnx(tensor: np.ndarray, model: str) -> np.ndarray:
    session = _onnx_session(model)
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: tensor})
    # scripts/export_onnx.py wraps the model so the graph has exactly one
    # output — the finest prediction, pre-sigmoid. outputs[-1] is therefore
    # that tensor, shaped (1, 1, size, size).
    return np.asarray(outputs[-1]).squeeze()


@functools.lru_cache(maxsize=1)
def _torch_model(hf_repo: str):
    import torch
    from transformers import AutoModelForImageSegmentation

    # BiRefNet ships custom modelling code, hence trust_remote_code.
    model = AutoModelForImageSegmentation.from_pretrained(
        hf_repo, trust_remote_code=True
    )
    # The HF weights are stored in float16. CPU conv2d refuses to mix a
    # float32 input with half bias, so normalise to float32 and only go back
    # to half on CUDA, where it is both supported and faster.
    model = model.float().eval()
    if torch.cuda.is_available():
        model = model.to("cuda").half()
    elif torch.backends.mps.is_available():
        model = model.to("mps")
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
    """Stable sigmoid, applied unconditionally.

    Both backends return raw logits: upstream's own inference is
    `birefnet(x)[-1].sigmoid()`, and scripts/export_onnx.py exports that same
    pre-sigmoid graph on purpose. Do not make this conditional on the value
    range — a degenerate input whose logits happen to land inside [0, 1]
    would silently skip activation and return a near-uniform 0.5 mask.
    """
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32)


def _resize_map(prob: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resample a probability map to (w, h)."""
    if prob.shape[::-1] == size:
        return prob.astype(np.float32)
    img = Image.fromarray((prob * 255.0).astype(np.uint8), mode="L")
    resized = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0
