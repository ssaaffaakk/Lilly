#!/usr/bin/env python3
"""Mix Croatian speech into the Bosnian training set without drowning it.

Lilly hears 35.5% of words wrong, trained on 3,091 Bosnian clips, and the
nearest useful data is Croatian. Concatenating the two is the obvious move and
the wrong one: at forty hours of Croatian against four of Bosnian the model
hears mostly Croatian and drifts towards it. The overall error rate can fall
while Bosnian gets worse, which is exactly the thing this project is supposed to
be good at, and a single word-error figure cannot see it happening.

So the Bosnian clips are repeated until they hold a stated share of the examples.
The share is an argument rather than a constant because the right value is not
knowable from first principles -- it gets measured by training at two settings
and scoring each on Bosnian specifically.

    python3 data/scripts/build_speech_mix.py --share 0.35
    python3 data/scripts/build_speech_mix.py --share 0.35 --dry-run

Every path in the output is absolute. That is not tidiness: training/train_speech.py
resolves a relative clip path against the directory of the TSV it was read from,
so combining two corpora into one relative-path file silently repoints every row
of one of them at a directory it was never in. The rows do not vanish with an
error -- they turn into missing files, or worse, into whatever happens to share
that name.
"""
import argparse
import random
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOSNIAN = REPO_ROOT / "data" / "speech" / "train.tsv"
EXTRA_DIR = REPO_ROOT / "data" / "speech-extra"
OUT = REPO_ROOT / "data" / "speech-extra" / "train-mix.tsv"
HELD_OUT = (REPO_ROOT / "data" / "speech" / "valid.tsv",
            REPO_ROOT / "data" / "speech" / "test.tsv")


def read(path: Path) -> list:
    """Rows as (absolute clip path, transcript).

    Relative paths resolve against the TSV's own directory, which is the rule
    training/train_speech.py uses. Doing it here, once, is what lets corpora
    from different directories sit in one file.
    """
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
            continue
        clip = Path(parts[0].strip())
        rows.append((clip if clip.is_absolute() else path.parent / clip,
                     parts[1].strip()))
    return rows


def transcripts(paths) -> set:
    """Normalised, because a leak that differs by an accent is still a leak."""
    seen = set()
    for path in paths:
        for _clip, text in read(path):
            seen.add(unicodedata.normalize("NFC", text.lower()))
    return seen


def neighbour_rows() -> list:
    """Every extra corpus the downloader has finished writing.

    Per-source files rather than the combined one the downloader also writes:
    that combined file already has our Bosnian folded into it, and this needs
    the two sides separable to set the ratio between them. Reading the parts is
    also what lets this run while the download is still going.
    """
    rows = []
    for path in sorted(EXTRA_DIR.glob("*/*.tsv")):
        if path.name.startswith("train-mix"):
            continue
        found = read(path)
        if found:
            rows += found
            print(f"{path.parent.name}: {len(found):,} clips")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=0.35,
                    help="fraction of the mixed set that should be Bosnian")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not 0 < args.share < 1:
        print("--share must be between 0 and 1", file=sys.stderr)
        return 1

    bosnian = read(BOSNIAN)
    if not bosnian:
        print(f"no Bosnian clips at {BOSNIAN}", file=sys.stderr)
        return 1
    print(f"Bosnian: {len(bosnian):,} clips")

    extra = neighbour_rows()
    if not extra:
        print(f"nothing under {EXTRA_DIR}/*/ yet — nothing to mix", file=sys.stderr)
        return 1

    # The held-out sets decide every published number, so a clip whose
    # transcript is in them cannot be trained on. Checked here rather than
    # trusted from whoever produced the extra corpus.
    forbidden = transcripts(HELD_OUT)
    before = len(extra)
    extra = [r for r in extra
             if unicodedata.normalize("NFC", r[1].lower()) not in forbidden]
    leaked = before - len(extra)
    print(f"\n{leaked:,} extra clips dropped for appearing in valid or test")
    if leaked:
        print("  (that they existed is worth knowing — the source overlaps our "
              "held-out sets, so any score measured without this check was inflated)")

    # Repeat Bosnian until it holds the share. Repetition rather than discarding
    # the neighbours: throwing away Croatian to reach a ratio wastes the data we
    # went and got, and the model sees each Bosnian clip several times per epoch
    # either way.
    copies = max(1, round(args.share * len(extra) / ((1 - args.share) * len(bosnian))))
    mixed = bosnian * copies + extra
    random.Random(args.seed).shuffle(mixed)
    actual = len(bosnian) * copies / len(mixed)

    print(f"\nBosnian repeated {copies}x -> {len(bosnian) * copies:,} rows")
    print(f"neighbours                  {len(extra):,} rows")
    print(f"mixed                       {len(mixed):,} rows, {actual:.0%} Bosnian"
          f"  (asked for {args.share:.0%})")

    missing = sum(1 for clip, _ in mixed[:400] if not clip.exists())
    if missing:
        print(f"\n{missing} of the first 400 clips do not exist on disk — the "
              f"paths are wrong, not the ratio", file=sys.stderr)
        return 1
    print("first 400 clip paths all resolve")

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(f"{clip}\t{text}\n" for clip, text in mixed), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO_ROOT)}")
    print(f"  train with: python3 training/train_speech.py "
          f"--data {args.out.relative_to(REPO_ROOT)} --valid data/speech/valid.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
