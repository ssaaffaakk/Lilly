#!/usr/bin/env python3
"""Measure what ALL-CAPS costs the served translator, and what truecasing gives back.

docs/REPORT-what-would-raise-the-numbers-2026-09-13.md, section 1, measured
this on uppercased FLORES sentences: 25.5 chrF2 lost to capitals and 23.1 of it
recovered by a crude restorer. That measurement was a diagnostic. This one runs
the same three variants through app.translate.Engine -- the served int8 build,
the app's own sentence splitter and target label -- so the number describes the
product's path rather than a PyTorch model around it.

Per direction, three variants of the same source against the same reference:

    A  the source as written            (the chat path, no recasing)
    B  the source uppercased            (what a sign looks like to the translator)
    C  the source uppercased, then restored by app.truecase.restore_sentence_case

C calls the restorer directly. It used to go through LILLY_TRUECASE inside
Engine.translate, but the shipped wiring moved to the photograph path
(app.lilly.translate_photo, line by line) and no longer sits in translate at
all -- so a whole-sentence FLORES source has no flag to toggle. That is fine
here: FLORES is single sentences in one language, and this is a diagnostic of
the restorer's effect on translation, not of the photograph path. The number
is unchanged, because the flag only ever called this same restorer on the whole
string.

    .venv/bin/python3 scripts/measure_truecase.py --limit 200

Scoring is sacrebleu, called exactly as training/evaluate.py's score() calls it
(corpus_bleu, and corpus_chrf with word_order=0, which is chrF2 -- word_order=2
is chrF++ and is a different metric). Four worker processes each hold one or
both engines; a sentence is one decode, so the wall clock is roughly
sentences x 2 seconds and --limit is the lever.

Nothing here changes the repository. It prints the table and writes the raw
hypotheses to a JSON file whose path it prints, so a number can be re-scored
without paying for the translations again.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The app package lives at the repo root, not next to this file.
sys.path.insert(0, str(REPO))

FLORES = REPO / "data" / "flores"

# Each worker keeps the engines it has loaded, so a worker that touches both
# directions pays two loads rather than one per task.
_ENGINES = {}


def _engine(direction: str):
    if direction not in _ENGINES:
        from app.translate import Engine
        _ENGINES[direction] = Engine(direction=direction)
    return _ENGINES[direction]


def _translate_task(task: dict) -> tuple:
    """One (direction, variant, block of rows); returns hypotheses by row index.

    Variant C uppercases the source and then restores it with the same restorer
    the photograph path runs, applied to the whole sentence (FLORES rows are one
    sentence in one language, so the line-wise photo restorer would do the same
    thing here). A and B feed the source straight through.
    """
    from app.truecase import restore_sentence_case
    engine = _engine(task["direction"])
    hyps = {}
    for i, line in zip(task["idxs"], task["lines"]):
        source = line.upper() if task["variant"] in ("B", "C") else line
        if task["variant"] == "C":
            source = restore_sentence_case(source)[0]
        hyps[i] = engine.translate(source)
    return task["direction"], task["variant"], hyps


def score(hyps: list, refs: list) -> tuple:
    """BLEU and chrF2, the same calls as training/evaluate.py's score()."""
    import sacrebleu
    return (sacrebleu.corpus_bleu(hyps, [refs]).score,
            sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score)


def rows_for(direction: str, limit: int) -> tuple:
    """(sources, references) for a direction, in file order."""
    if direction == "en-bs":
        src, ref = FLORES / "devtest.en", FLORES / "devtest.bs"
    else:
        src, ref = FLORES / "devtest.bs", FLORES / "devtest.en"
    sources = src.read_text(encoding="utf-8").splitlines()[:limit]
    refs = ref.read_text(encoding="utf-8").splitlines()[:limit]
    if len(sources) != len(refs):
        raise SystemExit(f"{src.name} and {ref.name} disagree on length; cannot pair them")
    return sources, refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200,
                    help="FLORES devtest pairs per direction (default 200)")
    ap.add_argument("--workers", type=int, default=4,
                    help="worker processes (default 4; measured fastest on this 6-core Mac)")
    ap.add_argument("--directions", nargs="+", default=["en-bs", "bs-en"],
                    choices=["en-bs", "bs-en"])
    ap.add_argument("--out", default="/tmp/lilly_truecase_measure.json")
    args = ap.parse_args()

    rows = {d: rows_for(d, args.limit) for d in args.directions}
    # Two blocks per (direction, variant) so four workers stay busy while one
    # direction's engine is loading.
    tasks = []
    for direction in args.directions:
        sources, _ = rows[direction]
        n = len(sources)
        halves = [(0, n // 2), (n // 2, n)]
        for variant in ("A", "B", "C"):
            for start, end in halves:
                if start == end:
                    continue
                tasks.append({"direction": direction, "variant": variant,
                              "idxs": list(range(start, end)),
                              "lines": sources[start:end]})

    total_sentences = sum(len(t["idxs"]) for t in tasks)
    print(f"{args.limit} pairs x 3 variants x {len(args.directions)} directions "
          f"= {total_sentences} sentences to translate, {args.workers} workers",
          flush=True)

    t0 = time.time()
    results = {d: {v: {} for v in "ABC"} for d in args.directions}
    done = 0
    with Pool(args.workers) as pool:
        for direction, variant, hyps in pool.imap_unordered(_translate_task, tasks):
            results[direction][variant].update(hyps)
            done += len(hyps)
            print(f"  {done}/{total_sentences} sentences  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    elapsed = time.time() - t0

    table = {}
    for direction in args.directions:
        _, refs = rows[direction]
        table[direction] = {}
        for variant in "ABC":
            hyps = [results[direction][variant][i] for i in range(len(refs))]
            table[direction][variant] = score(hyps, refs)

    print()
    print(f"pairs per direction: {args.limit}   "
          f"wall clock: {elapsed / 60:.1f} min   workers: {args.workers}")
    print()
    print(f"{'direction':<9} {'variant':<28} {'BLEU':>7} {'chrF2':>7}")
    labels = {"A": "A as written",
              "B": "B uppercased",
              "C": "C uppercased, truecased"}
    for direction in args.directions:
        for variant in "ABC":
            bleu, chrf = table[direction][variant]
            print(f"{direction:<9} {labels[variant]:<28} {bleu:>7.2f} {chrf:>7.2f}")
        print()
    print(f"{'direction':<9} {'gap':<28} {'BLEU':>7} {'chrF2':>7}")
    for direction in args.directions:
        a_bleu, a_chrf = table[direction]["A"]
        for variant in ("B", "C"):
            bleu, chrf = table[direction][variant]
            print(f"{direction:<9} {f'{variant} minus A':<28} "
                  f"{bleu - a_bleu:>+7.2f} {chrf - a_chrf:>+7.2f}")
        print()

    Path(args.out).write_text(json.dumps({
        "limit": args.limit, "workers": args.workers,
        "seconds": round(elapsed, 1),
        "directions": args.directions,
        "scores": table,
        "hypotheses": {d: {v: [results[d][v][i] for i in range(args.limit)]
                           for v in "ABC"} for d in args.directions},
        "references": {d: rows[d][1] for d in args.directions},
        "sources": {d: rows[d][0] for d in args.directions},
    }, ensure_ascii=False), encoding="utf-8")
    print(f"raw hypotheses, references and scores written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
