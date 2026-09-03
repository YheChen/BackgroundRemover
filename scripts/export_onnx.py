#!/usr/bin/env python3
"""Export a stage-2 segmentation model to ONNX.

Run once per model. The resulting graph is what both the Python shipping path
and the browser build consume, so this script is the boundary between the
heavy reference environment and everything we actually distribute.

    uv pip install -e ".[torch]"
    python scripts/export_onnx.py --model birefnet-general

Weights land in ./weights/ (gitignored) or $BGREMOVER_WEIGHTS.

UNVERIFIED: this has not been run yet. BiRefNet emits a list of multi-scale
side outputs and the export needs `output_names` to match what
stages/segment.py::_run_onnx expects (it takes the last output). If the mask
comes back inverted, blank, or at the wrong scale, check that first, then the
normalisation constants in segment.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgremover import models  # noqa: E402

# BiRefNet's ASPP blocks use deformable convolutions, and the ONNX standard
# schema introduces DeformConv at opset 22. Export below that and the graph
# loads fine in onnx but ORT rejects it at session creation with
# "No Op registered for DeformConv with domain_version of 18". ORT itself does
# implement the op -- the opset is the whole problem.
OPSET = 22


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=models.DEFAULT_MODEL, choices=models.available())
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument(
        "--dynamic", action="store_true",
        help="allow variable input size (larger graph, slower; needed for HR tiling)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing export")
    args = parser.parse_args()

    spec = models.get(args.model)
    out = models.onnx_path(args.model)

    if out.exists() and not args.force:
        print(f"{out} already exists; pass --force to overwrite")
        return 0

    print(f"model    {spec.key}  ({spec.licence})")
    print(f"weights  {spec.hf_repo}")
    print(f"size     {spec.input_size}x{spec.input_size}")
    print(f"output   {out}\n")

    try:
        import torch
        from transformers import AutoModelForImageSegmentation
    except ImportError:
        print(
            "error: the torch extra is not installed.\n"
            '  uv pip install -e ".[torch]"',
            file=sys.stderr,
        )
        return 1

    print("loading weights (this downloads ~900 MB on first run)...")
    model = AutoModelForImageSegmentation.from_pretrained(
        spec.hf_repo, trust_remote_code=True
    )
    # Weights are float16 on the Hub; CPU conv2d will not mix a float32 input
    # with a half bias, and the export traces on CPU.
    model = model.float().eval()

    # BiRefNet returns a LIST of multi-scale side outputs, finest last.
    # Exporting that directly yields a multi-output graph whose ordering we
    # would then have to trust at inference time. Wrap it so the graph has
    # exactly one output: the finest prediction, still pre-sigmoid.
    # segment.py applies sigmoid itself, for both backends.
    class FinestOnly(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.inner(x)
            while isinstance(out, (list, tuple)):
                out = out[-1]
            if hasattr(out, "logits"):
                out = out.logits
            return out

    model = FinestOnly(model)

    dummy = torch.randn(1, 3, spec.input_size, spec.input_size)
    dynamic_axes = (
        {"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch"}}
        if args.dynamic
        else {"input": {0: "batch"}, "output": {0: "batch"}}
    )

    print("exporting...")
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out),
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    # The exporter puts weights in a sibling `<name>.onnx.data` rather than
    # inlining them, so the .onnx file alone is ~10 MB of pure graph. Report
    # both, or the number looks impossibly small.
    parts = [out, *sorted(out.parent.glob(out.name + ".data*"))]
    total_mb = sum(p.stat().st_size for p in parts) / 1e6
    print()
    for p in parts:
        print(f"wrote {p.name:<34} {p.stat().st_size / 1e6:>8.0f} MB")
    print(f"{'total':<40} {total_mb:>8.0f} MB")
    if len(parts) > 1:
        print(
            "\nnote: the .onnx and .onnx.data files must travel together — "
            "onnxruntime resolves the external data relative to the graph."
        )
    print("\nverify with:  python -m pytest tests/test_pipeline_integration.py -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
