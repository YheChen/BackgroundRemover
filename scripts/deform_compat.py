"""ONNX-compatible modulated deformable convolution.

Why this exists: BiRefNet's ASPP blocks use torchvision's `deform_conv2d`.
Exported directly it becomes a `DeformConv` node, which

  - only enters the ONNX standard schema at opset 22, and
  - is NOT implemented by onnxruntime-web at all, on any provider.

So the browser build fails at session creation with "Could not find an
implementation for DeformConv(22)". Native onnxruntime does implement it, so
this only matters for the web target — but the web target is the product.

The fix is to express the same arithmetic with `grid_sample`, which ORT-web
does implement. Modulated deformable convolution is

    y[b,co,oh,ow] = bias[co]
                  + sum over ci,kh,kw of
                      w[co,ci,kh,kw] * m[b,k,oh,ow] * x[b,ci, py, px]

    py = oh*stride_h - pad_h + kh*dil_h + offset[b, 2k,   oh, ow]
    px = ow*stride_w - pad_w + kw*dil_w + offset[b, 2k+1, oh, ow]

with x bilinearly sampled at (py, px) and zero outside. That is exactly a
`grid_sample` per kernel tap, followed by a 1x1 convolution with that tap's
slice of the weight — so the whole thing is K grid_samples and K 1x1 convs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def deform_conv2d_onnx(
    inp: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for torchvision.ops.deform_conv2d, ONNX-friendly."""
    b, _, h_in, w_in = inp.shape
    out_ch, _, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    k = kh * kw

    h_out = (h_in + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    w_out = (w_in + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1

    device, dtype = inp.device, inp.dtype

    # Base sampling location of every output pixel, before offsets.
    oy = torch.arange(h_out, device=device, dtype=dtype) * sh - ph
    ox = torch.arange(w_out, device=device, dtype=dtype) * sw - pw
    base_y = oy.view(1, h_out, 1)
    base_x = ox.view(1, 1, w_out)

    # Kernel tap displacements. Plain Python scalars on purpose: kh/kw are
    # known at trace time, and torch.repeat_interleave with an implicit dim
    # is not convertible by the dynamo ONNX exporter.

    # offset is laid out (b, 2*k, h_out, w_out) as y0,x0,y1,x1,...
    off = offset.view(b, k, 2, h_out, w_out)
    off_y = off[:, :, 0]
    off_x = off[:, :, 1]

    out = inp.new_zeros((b, out_ch, h_out, w_out))
    # 2/(N-1) converts a pixel index to align_corners=True normalised space.
    ny = 2.0 / max(h_in - 1, 1)
    nx = 2.0 / max(w_in - 1, 1)

    for i in range(k):
        tap_y = float((i // kw) * dh)
        tap_x = float((i % kw) * dw)
        py = base_y + tap_y + off_y[:, i]
        px = base_x + tap_x + off_x[:, i]

        grid = torch.stack((px * nx - 1.0, py * ny - 1.0), dim=-1)
        sampled = F.grid_sample(
            inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        if mask is not None:
            sampled = sampled * mask[:, i : i + 1]

        # weight[:, :, kh_i, kw_i] as a 1x1 conv over the sampled plane.
        w_tap = weight[:, :, i // kw, i % kw].unsqueeze(-1).unsqueeze(-1)
        out = out + F.conv2d(sampled, w_tap)

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def patch_birefnet(model: torch.nn.Module) -> int:
    """Rewrite every DeformableConv2d.forward in `model` to the ONNX version.

    Returns the number of modules patched, so the caller can fail loudly if
    the model's internals move and this silently patches nothing.
    """
    patched = 0
    for module in model.modules():
        if type(module).__name__ != "DeformableConv2d":
            continue

        def forward(x, _m=module):
            offset = _m.offset_conv(x)
            modulator = 2.0 * torch.sigmoid(_m.modulator_conv(x))
            pad = _m.padding
            pad = (pad, pad) if isinstance(pad, int) else tuple(pad)
            return deform_conv2d_onnx(
                x,
                offset,
                _m.regular_conv.weight,
                _m.regular_conv.bias,
                stride=tuple(_m.stride),
                padding=pad,
                dilation=tuple(_m.regular_conv.dilation),
                mask=modulator,
            )

        module.forward = forward
        patched += 1
    return patched
