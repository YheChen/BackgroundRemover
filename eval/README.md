# Eval

You need this on day one. Without a scored set of hard images, every model
swap is a vibe check.

## Setup

Put your own photos in `eval/images/`, named to match the `file` field in
[`cases.toml`](cases.toml). They are gitignored — this repo does not ship or
redistribute anyone's images, and you should not commit photos you don't hold
the rights to.

Twenty cases are listed. Aim for 30–50 eventually. Every case records what it
`probes`, so a regression tells you which stage broke rather than just moving
a number.

## Run

```bash
python eval/score.py --run                 # generate cutouts for every case
python eval/score.py --score               # hand-score 0-3
python eval/score.py --report              # just the table
python eval/score.py --run --configs naive,matte    # compare two configs
```

Outputs land in `eval/out/`, flattened onto **mid-grey** on purpose:
transparent-on-white hides white fringing and transparent-on-black hides dark
fringing. Grey hides neither.

## Why scoring is manual

There is no automatic metric for "does this cutout look right" that tracks
what users actually notice. IoU is actively misleading here — it barely moves
when a 3px hair fringe is wrong, which is precisely the thing people
complain about. Eighty hand judgements is a couple of hours, once, and it is
the difference between engineering and guessing.

Scores persist in `eval/scores.json` keyed by `(case, config)`, so `--score`
only asks about combinations you haven't judged yet.
