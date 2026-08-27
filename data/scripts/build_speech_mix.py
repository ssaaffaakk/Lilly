#!/usr/bin/env python3
"""Mix Bosnian speech with its Croatian and Serbian neighbours, without drowning it.

Lilly has 3,091 Bosnian clips and its word error rate is 35.5%. The nearest
useful data is Croatian and Serbian — close languages, and there is far more of
it. Concatenating the two is the obvious move and the wrong one: at ten thousand
Croatian clips against three thousand Bosnian, the model hears mostly Croatian
and drifts towards it. The overall error rate can fall while Bosnian gets worse,
which is exactly the thing this project is supposed to be good at.

So Bosnian is repeated until it holds a stated share of the examples. That share
is an argument rather than a constant because the right value is not knowable
from first principles — it has to be measured by training at two or three
settings and scoring each on Bosnian specifically.

    python3 data/scripts/build_speech_mix.py --share 0.35

Writes data/speech/train-mix.tsv. Never touches valid.tsv or test.tsv: a change
there invalidates every speech number this project has published.
"""
import argparse
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOSNIAN = REPO_ROOT / "data" / "speech" / "train.tsv"
EXTRA_DIR = REPO_ROOT / "data" / "speech-extra"
OUT = REPO_ROOT / "data" / "speech" / "train-mix.tsv"
HELD_OUT = (REPO_ROOT / "data" / "speech" / "valid.tsv",
            REPO_ROOT / "data" / "speech" / "test.tsv")


def read(path: Path) -> list:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            rows.append(parts)
    return rows


def transcripts(paths) -> set:
    """Normalised, because a leak that differs by an accent is still a leak."""
    seen = set()
    for path in paths:
        for parts in read(path):
            seen.add(unicodedata.normalize("NFC", parts[1].strip().lower()))
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=0.35,
                    help="fraction of examples that must be Bosnian")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bosnian = read(BOSNIAN)
    if not bosnian:
        print(f"no Bosnian clips at {BOSNIAN}", file=sys.stderr)
        return 1
    extra = []
    for path in sorted(EXTRA_DIR.glob("*.tsv")) if EXTRA_DIR.exists() else []:
        rows = read(path)
        extra += rows
        print(f"{path.name}: {len(rows):,} clips")
    if not extra:
        print(f"nothing in {EXTRA_DIR} yet — nothing to mix", file=sys.stderr)
        return 1

    # The held-out sets decide every published number, so a clip whose
    # transcript is in them cannot be trained on. Checked here rather than
    # trusted from whoever produced the extra corpus.
    forbidden = transcripts(HELD_OUT)
    before = len(extra)
    extra = [r for r in extra
             if unicodedata.normalize("NFC", r[1].strip().lower()) not in forbidden]
    leaked = before - len(extra)
    print(f"\n{leaked:,} extra clips dropped for appearing in valid or test")
    if leaked:
        print("  (that they existed is worth knowing — the source overlaps our "
              "held-out sets, so any score measured without this check was inflated)")

    # Repeat Bosnian until it holds the share. Repetition rather than discarding
    # the neighbours: throwing away Croatian to reach a ratio wastes the data we
    # went and got, and the model sees each Bosnian clip several times per epoch
    # either way.
    if not 0 < args.share < 1:
        print("--share must be between 0 and 1", file=sys.stderr)
        return 1
    copies = max(1, round(args.share * len(extra) / ((1 - args.share) * len(bosnian))))
    mixed = bosnian * copies + extra
    realised = copies * len(bosnian) / len(mixed)

    print(f"\nBosnian {len(bosnian):,} x{copies} = {copies * len(bosnian):,}")
    print(f"neighbours {len(extra):,}")
    print(f"total {len(mixed):,}, Bosnian share {realised:.1%} (asked {args.share:.0%})")
    corpora = Counter(r[2] if len(r) > 2 else "?" for r in extra)
    print("  neighbours by source:", dict(corpora.most_common(6)))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    random.Random(args.seed).shuffle(mixed)
    OUT.write_text("\n".join("\t".join(r) for r in mixed) + "\n", encoding="utf-8")
    print(f"\nwritten to {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
