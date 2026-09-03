"""Command line interface.

    bgremover in.jpg out.png                    # full pipeline
    bgremover in.jpg out.png --edge naive       # coarse mask only, for comparison
    bgremover in.jpg out.png --matte matte.png  # also write the alpha plane
    bgremover batch ./photos ./cutouts          # whole folder
    bgremover models                            # what's registered, and its licence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bgremover",
        description="Remove image backgrounds. Six-stage pipeline, MIT weights only.",
    )
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--edge",
            default="matte",
            choices=["naive", "decontaminate", "matte", "refine"],
            help="how far past the coarse mask to go (default: matte)",
        )
        p.add_argument("--model", default=None, help="registry key (default: birefnet-general)")
        p.add_argument(
            "--backend", default="onnx", choices=["onnx", "torch"],
            help="onnx = shipping path, torch = reference path (default: onnx)",
        )
        p.add_argument(
            "--band", type=int, default=12,
            help="stage-3 unknown band width in px; the quality/speed dial (default: 12)",
        )
        p.add_argument(
            "--refine-tile", type=int, default=512,
            help="tile side for --edge refine, in px (default: 512)",
        )
        p.add_argument(
            "--bg", default=None,
            help="composite onto a solid colour, e.g. '#ffffff' (default: transparent)",
        )
        p.add_argument("--crop", action="store_true", help="crop to the subject")
        p.add_argument(
            "--margin", type=float, default=0.0,
            help="fractional margin when cropping, e.g. 0.05 (default: 0)",
        )

    single = sub.add_parser("i", help="one image (default command)")
    single.add_argument("input", type=Path)
    single.add_argument("output", type=Path)
    single.add_argument("--matte", type=Path, default=None, help="also write the alpha plane here")
    single.add_argument("--trimap", type=Path, default=None, help="also write the trimap (debug)")
    add_common(single)

    batch = sub.add_parser("batch", help="a folder of images")
    batch.add_argument("input_dir", type=Path)
    batch.add_argument("output_dir", type=Path)
    add_common(batch)

    sub.add_parser("models", help="list registered models and their licences")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Allow `bgremover in.jpg out.png` without the `i` subcommand.
    if argv and argv[0] not in {"i", "batch", "models", "-h", "--help"}:
        argv.insert(0, "i")

    args = build_parser().parse_args(argv)
    try:
        if args.command == "models":
            return _cmd_models()
        if args.command == "batch":
            return _cmd_batch(args)
        if args.command == "i":
            return _cmd_single(args)
    except FileNotFoundError as exc:
        # Missing weights on a fresh clone is an expected state, not a crash.
        # The exception message already names the fix; don't bury it in a
        # traceback the user has to read past.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    build_parser().print_help()
    return 2


def _cmd_models() -> int:
    from . import models

    print(f"{'KEY':<22} {'LICENCE':<12} {'PX':<6} NOTES")
    for key in models.available():
        spec = models.get(key)
        default = "  (default)" if key == models.DEFAULT_MODEL else ""
        print(f"{key:<22} {spec.licence:<12} {spec.input_size:<6} {spec.notes}{default}")
    print("\nWeights carry their own licences — see NOTICE.")
    return 0


def _cmd_single(args: argparse.Namespace) -> int:
    from . import load, save_alpha
    from .models import DEFAULT_MODEL

    if not args.input.exists():
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 1

    cutout = _run(load(args.input), args, args.model or DEFAULT_MODEL)
    _write(cutout, args.output, args)
    if args.matte:
        save_alpha(cutout, args.matte)
    if args.trimap and cutout.trimap is not None:
        from PIL import Image

        Image.fromarray(cutout.trimap, mode="L").save(args.trimap)
    print(f"{args.input} -> {args.output}  [{args.edge}]")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from . import load
    from .models import DEFAULT_MODEL

    if not args.input_dir.is_dir():
        print(f"error: not a directory: {args.input_dir}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        print(f"error: no images in {args.input_dir}", file=sys.stderr)
        return 1

    model = args.model or DEFAULT_MODEL
    failures = 0
    for n, path in enumerate(paths, 1):
        out = args.output_dir / f"{path.stem}.png"
        try:
            _write(_run(load(path), args, model), out, args)
            print(f"[{n}/{len(paths)}] {path.name} -> {out.name}")
        except Exception as exc:  # keep going; report at the end
            failures += 1
            print(f"[{n}/{len(paths)}] {path.name} FAILED: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{failures}/{len(paths)} failed", file=sys.stderr)
    return 1 if failures else 0


def _run(image, args: argparse.Namespace, model: str):
    from .pipeline import remove_background

    return remove_background(
        image,
        edge_mode=args.edge,
        model=model,
        backend=args.backend,
        band_width=args.band,
        refine_tile=args.refine_tile,
        keep_trimap=getattr(args, "trimap", None) is not None,
    )


def _write(cutout, path: Path, args: argparse.Namespace) -> None:
    from PIL import Image

    from .stages import composite as composite_stage

    if args.crop or args.margin:
        cutout = composite_stage.crop_to_subject(cutout, margin=args.margin)

    if args.bg:
        flat = composite_stage.composite(cutout, _parse_colour(args.bg))
        Image.fromarray(flat, mode="RGB").save(path)
    else:
        Image.fromarray(cutout.to_rgba(), mode="RGBA").save(path)


def _parse_colour(text: str) -> tuple[int, int, int]:
    s = text.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"cannot parse colour {text!r}; use '#rrggbb' or '#rgb'")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


if __name__ == "__main__":
    raise SystemExit(main())
