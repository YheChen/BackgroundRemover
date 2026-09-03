#!/usr/bin/env python3
"""Eval harness: run every case through one or more configs, then score by hand.

Two modes.

    python eval/score.py --run                 generate cutouts for every case
    python eval/score.py --score               walk the outputs and record 0-3

Scores go to eval/scores.json, keyed by (case, config), so re-running --score
only asks about combinations you haven't judged yet. That matters: hand
scoring 20 cases x 4 configs is 80 judgements and you will not redo it for
fun.

Deliberately manual. There is no automatic metric for "does this cutout look
right" that correlates with what users notice, and IoU in particular rewards
exactly the wrong thing — it barely moves when a 3px hair fringe is wrong,
which is the thing people actually complain about.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tomllib

EVAL_DIR = Path(__file__).resolve().parent
CASES = EVAL_DIR / "cases.toml"
IMAGES = EVAL_DIR / "images"
OUT = EVAL_DIR / "out"
SCORES = EVAL_DIR / "scores.json"

# The configs to compare. Add a row when you add an engine.
CONFIGS: dict[str, dict] = {
    "naive":         {"edge_mode": "naive"},
    "decontaminate": {"edge_mode": "decontaminate"},
    "matte":         {"edge_mode": "matte"},
    "matte-wide":    {"edge_mode": "matte", "band_width": 24},
}

RUBRIC = """
  0  unusable      subject damaged, or background left behind
  1  visible       a viewer would notice unprompted
  2  good          flaws only visible at 100% zoom
  3  clean         indistinguishable from a hand-cut mask
  s  skip          come back to this one
  q  quit          save and exit
"""


def load_cases() -> list[dict]:
    with CASES.open("rb") as fh:
        return tomllib.load(fh)["case"]


def load_scores() -> dict:
    return json.loads(SCORES.read_text()) if SCORES.exists() else {}


def save_scores(scores: dict) -> None:
    SCORES.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")


def cmd_run(configs: list[str]) -> int:
    from PIL import Image

    from bgremover import load, remove_background
    from bgremover.stages import composite as composite_stage

    cases = load_cases()
    missing = [c["file"] for c in cases if not (IMAGES / c["file"]).exists()]
    if missing:
        print(f"note: {len(missing)}/{len(cases)} case images not present in {IMAGES}")
        print("      populate them with your own photos; they are gitignored.\n")

    present = [c for c in cases if (IMAGES / c["file"]).exists()]
    if not present:
        print(f"error: no case images found in {IMAGES}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case in present:
        image = load(IMAGES / case["file"])
        stem = Path(case["file"]).stem
        for name in configs:
            dest = OUT / f"{stem}__{name}.png"
            try:
                cutout = remove_background(image, **CONFIGS[name])
                # Flatten onto mid-grey: transparent-on-white hides white
                # fringing and transparent-on-black hides dark fringing.
                flat = composite_stage.composite(cutout, (128, 128, 128))
                Image.fromarray(flat, mode="RGB").save(dest)
                print(f"  {dest.name}")
            except Exception as exc:
                failures += 1
                print(f"  {dest.name} FAILED: {exc}", file=sys.stderr)

    print(f"\n{len(present)} cases x {len(configs)} configs -> {OUT}")
    if failures:
        print(f"{failures} failed", file=sys.stderr)
    print("now:  python eval/score.py --score")
    return 1 if failures else 0


def cmd_score(configs: list[str]) -> int:
    scores = load_scores()
    cases = load_cases()
    todo = [
        (case, name)
        for case in cases
        for name in configs
        if (OUT / f"{Path(case['file']).stem}__{name}.png").exists()
        and f"{case['file']}|{name}" not in scores
    ]

    if not todo:
        print("nothing new to score.")
        return report(configs)

    print(f"{len(todo)} to score.{RUBRIC}")
    for case, name in todo:
        key = f"{case['file']}|{name}"
        path = OUT / f"{Path(case['file']).stem}__{name}.png"
        print(f"\n{case['file']}  [{name}]")
        print(f"  probes: {', '.join(case.get('probes', []))}")
        print(f"  why:    {case.get('why', '')}")
        print(f"  open:   {path}")
        try:
            answer = input("  score 0-3 / s / q > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\ninterrupted")
            break
        if answer == "q":
            break
        if answer == "s" or answer not in {"0", "1", "2", "3"}:
            continue
        scores[key] = int(answer)

    save_scores(scores)
    print(f"\nsaved {len(scores)} scores to {SCORES}")
    return report(configs)


def report(configs: list[str]) -> int:
    scores = load_scores()
    if not scores:
        print("no scores yet.")
        return 0

    print(f"\n{'CONFIG':<16} {'N':>4} {'MEAN':>6} {'UNUSABLE':>9}")
    for name in configs:
        vals = [v for k, v in scores.items() if k.endswith(f"|{name}")]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        unusable = sum(1 for v in vals if v == 0)
        print(f"{name:<16} {len(vals):>4} {mean:>6.2f} {unusable:>9}")

    print("\nBy probe:")
    cases = {c["file"]: c for c in load_cases()}
    probes: dict[str, list[int]] = {}
    for key, val in scores.items():
        file = key.split("|")[0]
        for probe in cases.get(file, {}).get("probes", []):
            probes.setdefault(probe, []).append(val)
    for probe, vals in sorted(probes.items()):
        print(f"  {probe:<20} {sum(vals) / len(vals):.2f}  (n={len(vals)})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="generate cutouts")
    parser.add_argument("--score", action="store_true", help="hand-score the outputs")
    parser.add_argument("--report", action="store_true", help="print the table only")
    parser.add_argument(
        "--configs", default=",".join(CONFIGS),
        help=f"comma-separated subset of: {','.join(CONFIGS)}",
    )
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        print(f"error: unknown configs {sorted(unknown)}", file=sys.stderr)
        return 1

    if args.run:
        return cmd_run(configs)
    if args.score:
        return cmd_score(configs)
    if args.report:
        return report(configs)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
