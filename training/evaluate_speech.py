#!/usr/bin/env python3
"""Score Lilly's listening against clips it never trained on.

Word error rate: of every word in the reference, how many did the model get
wrong — a wrong word, a missing one, or one it invented. 0% is perfect, and
above 100% is possible when it invents more than it hears. Lower is better.

Run it once before training to get a baseline, once after to see whether the
fine-tune actually helped. A fine-tune that does not move this number did not
work, however cleanly it ran.

    python3 training/evaluate_speech.py --data data/speech/test.tsv
    python3 training/evaluate_speech.py --data data/speech/test.tsv \
        --model models/lilly/listen-previous     # the baseline you kept

Same TSV as training: audio path, tab, what is actually said.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from training.train_speech import read_tsv  # noqa: E402  (same TSV format)

# Refuse to start if the machine has no room. Five of these ran at once on
# 27 August and the kernel panicked: 100% of the compressor limit, fifteen
# swapfiles, watchdog silent for 94 seconds. Each job is reasonable alone and
# none of them knew the others existed.
#
# Claimed in main(), not at import: `from training.evaluate_speech import edits`
# is a word-distance function and must not cost 1.2 GB. See the same note in
# training/train_speech.py.
from scripts.guard import claim  # noqa: E402


def edits(hyp: list, ref: list) -> int:
    """Levenshtein distance in words — substitutions, deletions, insertions."""
    prev = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        cur = [i]
        for j, r in enumerate(ref, 1):
            cur.append(min(prev[j] + 1,           # deletion
                           cur[j - 1] + 1,        # insertion
                           prev[j - 1] + (h != r)))  # substitution
        prev = cur
    return prev[-1]


def normalise(text: str) -> list:
    keep = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return keep.split()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path,
                    default=REPO_ROOT / "models" / "lilly" / "listen")
    ap.add_argument("--language", default="bs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", type=int, default=5, help="worst clips to print")
    args = ap.parse_args()

    rows = read_tsv(args.data)[: args.limit]
    if not rows:
        print(f"no usable rows in {args.data}", file=sys.stderr)
        return 1

    claim(1.2, "speech scoring")
    # Through the app's own listener, not a faster-whisper call built here. The
    # decode settings a user gets are then the decode settings that are scored,
    # by construction rather than by two copies staying in step.
    from app.speech import transcribe
    print(f"model: {args.model}\nclips: {len(rows)}")

    total_edits = total_words = 0
    worst = []
    for clip, reference in rows:
        heard = transcribe(str(clip), language=args.language, build=args.model)
        ref_words, hyp_words = normalise(reference), normalise(heard)
        wrong = edits(hyp_words, ref_words)
        total_edits += wrong
        total_words += len(ref_words)
        if ref_words:
            worst.append((wrong / len(ref_words), clip.name, reference, heard))

    wer = 100 * total_edits / max(total_words, 1)
    print(f"\nword error rate: {wer:.1f}%  ({total_edits} wrong of {total_words} words)")

    worst.sort(reverse=True)
    if args.show and worst:
        print(f"\nworst {min(args.show, len(worst))}:")
        for rate, name, reference, heard in worst[: args.show]:
            print(f"  {name} — {100 * rate:.0f}%\n    said:  {reference}\n    heard: {heard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
