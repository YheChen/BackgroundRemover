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

OPSET = 17


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
    model.eval()

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

    size_mb = out.stat().st_size / 1e6
    print(f"\nwrote {out}  ({size_mb:.0f} MB)")
    print("verify with:  python -m pytest tests/test_pipeline_integration.py -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
