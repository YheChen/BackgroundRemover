#!/usr/bin/env python3
"""Strip ceil_mode from MaxPool nodes so a graph loads in onnxruntime-web.

ORT-web refuses any graph containing a MaxPool with ceil_mode=1:

    using ceil() in shape computation is not yet supported for MaxPool

For a 2x2 / stride-2 pool over an even-sized feature map, ceil and floor give
the identical output size, so the attribute is decorative and dropping it is a
no-op. That holds for ISNet at 1024x1024, where every feature map stays even
all the way down (1024, 512, 256, 128, 64, 32, 16).

It does NOT hold in general. This script therefore verifies the patched graph
against the original on random input and refuses to write a graph that differs.

    python scripts/patch_maxpool_ceil.py weights/isnet-general-use.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--size", type=int, default=1024, help="input side for verification")
    args = ap.parse_args()

    out = args.out or args.model.with_name(args.model.stem + "-web" + args.model.suffix)

    m = onnx.load(str(args.model))
    changed = 0
    for node in m.graph.node:
        if node.op_type != "MaxPool":
            continue
        for a in node.attribute:
            if a.name == "ceil_mode" and a.i == 1:
                a.i = 0
                changed += 1

    if changed == 0:
        print("no MaxPool with ceil_mode=1; nothing to do")
        return 0
    print(f"stripped ceil_mode from {changed} MaxPool nodes")

    onnx.save_model(m, str(out), save_as_external_data=False)

    # The attribute is only decorative when every pooled map is even. Prove it
    # for this graph rather than assuming.
    so = ort.SessionOptions()
    so.log_severity_level = 3
    x = np.random.RandomState(0).randn(1, 3, args.size, args.size).astype(np.float32)
    ref = _run(str(args.model), x, so)
    got = _run(str(out), x, so)
    delta = float(np.abs(ref - got).max())
    print(f"max|diff| vs original: {delta:.3e}")

    if delta > 1e-6:
        out.unlink(missing_ok=True)
        print(
            "REFUSING to write: the patched graph differs from the original, so "
            "ceil_mode was load-bearing at this input size.",
            file=sys.stderr,
        )
        return 1

    print(f"verified identical -> {out}")
    return 0


def _run(path: str, x: np.ndarray, so: ort.SessionOptions) -> np.ndarray:
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return np.asarray(s.run(None, {s.get_inputs()[0].name: x})[0]).squeeze()


if __name__ == "__main__":
    raise SystemExit(main())
