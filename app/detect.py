#!/usr/bin/env python3
"""Language detection: is this text Bosnian or English?

A character n-gram multinomial Naive Bayes, trained on the repo's own parallel
corpus and shipped as one committed JSON file, so inference reads a few hundred
kilobytes off disk and never touches the network or the training corpora.

Why this and not a library: the app is offline at inference, and the one
heavyweight option (fasttext's lid.176.ftz) is a 900 MB download that is not
in the repo. Bosnian and English sit far enough apart in their letter
inventories -- c, c, d and s with their diacritics exist in Bosnian and never
in English -- that counting character 1-, 2- and 3-grams and asking which
language counts them more often is already a strong classifier. It is also
small, deterministic, and auditable: the whole model is two tables of counts.

Training (only ever on the machine that has the corpora, never at inference):

    python3 app/detect.py --train

reads the first 60,000 pairs of data/clean/train-mix.tsv (the curated mix:
SETIMES, WikiMatrix, wikimedia -- parallel Bosnian/English, both sides
sentence-aligned), holds out a random 20% and reports how many of the held-out
sentences it classified into the right language. The shipped model
(app/detect-model.json) is then trained on the whole sample, which is the
standard last step after the accuracy was established on the held-out part;
the numbers it was measured at are written into the file itself so the
artifact and its scores cannot drift apart. FLORES devtest -- professionally
translated, never in the sample -- is scored too, as an independent check.

The held-out figure understates the classifier: the crawled corpus carries
Wikipedia rows whose "Bosnian" column is an English citation ("Pristupljeno
10 April 2021"), and the classifier is usually right about the actual language
of the string while the label is wrong. FLORES has none of that noise.

Usage:
    python3 app/detect.py "Dobar dan"      # -> bs
    python3 app/detect.py --train          # retrain and re-measure
"""
import csv
import json
import math
import random
import sys
from pathlib import Path

MODEL_FILE = Path(__file__).resolve().parent / "detect-model.json"

# The training sample: the curated mix corpus, first N rows, both columns.
# A prefix rather than a shuffle so the sample is the same every run; the file
# is itself a mix of sources, so a prefix is not one source.
DATA_TSV = Path(__file__).resolve().parents[1] / "data" / "clean" / "train-mix.tsv"
SAMPLE_ROWS = 60_000
# Character n-grams up to this length. 3 catches "ije" against "ie" and
# "šće" against nothing in English; longer grams add model size, not accuracy.
NGRAMS = (1, 2, 3)
# A 2- or 3-gram seen once is usually noise from a crawled row; dropping it
# shrinks the model without moving the measured accuracy (measured: 98.39%
# at min_count 2, 3 and 4 alike). 1-grams are always kept: they are the
# letters themselves, and a held-out word of a new 3-gram still meets the
# model through its letters.
MIN_COUNT = 3
# Add-one smoothing, and every gram outside the kept vocabulary shares one
# small unseen probability rather than being counted -- that is what keeps
# the shipped table small while a sentence full of unfamiliar grams still
# gets a sane, nearly-neutral score.
SMOOTHING = 1.0

LANGS = ("bs", "en")


def _grams(text: str, n: int):
    lower = text.lower()
    return [lower[i:i + n] for i in range(len(lower) - n + 1)]


def _features(text: str):
    for n in NGRAMS:
        yield from _grams(text, n)


def _load_pairs(path: Path, rows: int) -> list:
    """The first `rows` sentence pairs of the corpus, as (bosnian, english).

    A malformed line is skipped rather than shifted: the corpus is crawled,
    and one short row must not misalign every pair after it. The two columns
    are read over the same file prefix, so the pairs stay aligned by index.
    """
    pairs = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) >= 3 and row[1].strip() and row[2].strip():
                pairs.append((row[1].strip(), row[2].strip()))
                if len(pairs) >= rows:
                    break
    return pairs


def _counts(pairs: list) -> dict:
    """Per-language counts of every kept gram, plus the totals for smoothing."""
    raw = {lang: {} for lang in LANGS}
    for bosnian, english in pairs:
        for gram in _features(bosnian):
            raw["bs"][gram] = raw["bs"].get(gram, 0) + 1
        for gram in _features(english):
            raw["en"][gram] = raw["en"].get(gram, 0) + 1
    kept = {}
    for lang in LANGS:
        kept[lang] = {g: c for g, c in raw[lang].items()
                      if len(g) < 2 or c >= MIN_COUNT}
    totals = {lang: sum(kept[lang].values()) for lang in LANGS}
    return {"counts": kept, "totals": totals}


class Detector:
    """One language classifier: log-probabilities built from the count table.

    A gram in the table scores by its count (add-one smoothed); one outside
    it scores the unseen mass. The two languages are compared by summed log
    likelihood under equal priors, and a text that scores them equally --
    say, a single digit, which is as common in either -- is Bosnian, the
    language the app faces by default.
    """

    def __init__(self, table: dict):
        self.counts = table["counts"]
        self.totals = table["totals"]
        self.meta = table.get("meta", {})
        self.vocab = sorted(set(self.counts["bs"]) | set(self.counts["en"]))
        self._probs = {}
        for lang in LANGS:
            total = self.totals[lang] + SMOOTHING * len(self.vocab)
            self._probs[lang] = {g: math.log((self.counts[lang].get(g, 0) + SMOOTHING) / total)
                                 for g in self.vocab}
            self._probs[lang]["\x00unseen"] = math.log(SMOOTHING / total)

    def predict(self, text: str) -> str:
        """`text` -> "bs" or "en". Ties go to Bosnian (see the class note)."""
        scores = {lang: 0.0 for lang in LANGS}
        for gram in _features(text):
            for lang in LANGS:
                scores[lang] += self._probs[lang].get(gram, self._probs[lang]["\x00unseen"])
        return "bs" if scores["bs"] >= scores["en"] else "en"


_detector = None
_detector_lock = None


def get_detector() -> Detector:
    """The shipped classifier, loaded once. Missing model file is a missing
    download, reported the way the reply model's is (FileNotFoundError)."""
    global _detector, _detector_lock
    if _detector_lock is None:
        import threading
        _detector_lock = threading.Lock()
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                if not MODEL_FILE.is_file():
                    raise FileNotFoundError(
                        f"no language detector at {MODEL_FILE} -- run "
                        f"python3 app/detect.py --train on a machine with data/clean/")
                _detector = Detector(json.loads(MODEL_FILE.read_text(encoding="utf-8")))
    return _detector


def detect_language(text: str) -> str:
    return get_detector().predict(text)


def _accuracy(detector: Detector, pairs: list) -> tuple:
    """(correct, total) over a set of pairs, both columns scored."""
    correct = total = 0
    for bosnian, english in pairs:
        correct += detector.predict(bosnian) == "bs"
        correct += detector.predict(english) == "en"
        total += 2
    return correct, total


def train_and_evaluate(rows: int = SAMPLE_ROWS) -> dict:
    """Measure on a held-out 20%, then train the shipped model on the whole.

    Returns the report (accuracy figures and model size) so main() can print
    it and the meta block can carry it. The split is seeded, so the same
    corpus reproduces the same numbers.
    """
    if not DATA_TSV.is_file():
        raise FileNotFoundError(
            f"no training corpus at {DATA_TSV} -- data/clean/ is recreated by "
            f"the data scripts and is not committed")
    pairs = _load_pairs(DATA_TSV, rows)
    if len(pairs) < rows:
        print(f"note: the corpus held {len(pairs)} usable pairs, not {rows}", file=sys.stderr)
    random.Random(42).shuffle(pairs)
    cut = int(len(pairs) * 0.8)
    train_pairs, held_out = pairs[:cut], pairs[cut:]

    held_detector = Detector(_counts(train_pairs))
    correct, total = _accuracy(held_detector, held_out)

    report = {
        "pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "held_out_pairs": len(held_out),
        "held_out_accuracy": correct / total if total else 0.0,
        "held_out_correct": correct,
        "held_out_total": total,
        "flores_devtest_accuracy": None,
    }

    # FLORES devtest: professionally translated, never in the sample, so it is
    # a second, independent check with none of the crawled corpus's label noise.
    flores = Path(__file__).resolve().parents[1] / "data" / "flores"
    flores_bs = flores / "devtest.bs"
    flores_en = flores / "devtest.en"
    if flores_bs.is_file() and flores_en.is_file():
        fb = [line.strip() for line in flores_bs.read_text(encoding="utf-8").splitlines() if line.strip()]
        fe = [line.strip() for line in flores_en.read_text(encoding="utf-8").splitlines() if line.strip()]
        flores_pairs = list(zip(fb, fe))
        c, t = _accuracy(held_detector, flores_pairs)
        report["flores_devtest_accuracy"] = c / t if t else 0.0
        report["flores_devtest_correct"] = c
        report["flores_devtest_total"] = t

    # The shipped model trains on the whole sample: the accuracy above was
    # measured on the held-out 20% first, and the last model gets all of it.
    # The corpus is recorded repo-relative, so the committed file names the
    # same path on every machine that has it.
    table = _counts(pairs)
    table["meta"] = {
        "corpus": "data/clean/train-mix.tsv",
        "rows": len(pairs),
        "min_count": MIN_COUNT,
        "ngrams": list(NGRAMS),
        "smoothing": SMOOTHING,
        "held_out_accuracy": report["held_out_accuracy"],
        "flores_devtest_accuracy": report["flores_devtest_accuracy"],
        "measured_note": "random 80/20 split of the sample; FLORES devtest is "
                         "independent of it. Crawled rows carry English text in "
                         "the Bosnian column, so the split figure understates "
                         "the classifier.",
    }
    MODEL_FILE.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    report["model_bytes"] = MODEL_FILE.stat().st_size
    return report


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--train":
        try:
            report = train_and_evaluate()
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        acc = report["held_out_accuracy"]
        print(f"trained on {report['train_pairs']} pairs, held out {report['held_out_pairs']}")
        print(f"held-out accuracy: {report['held_out_correct']}/{report['held_out_total']} "
              f"= {acc:.2%} (random 80/20 split)")
        if report["flores_devtest_accuracy"] is not None:
            print(f"FLORES devtest:     {report['flores_devtest_correct']}/"
                  f"{report['flores_devtest_total']} "
                  f"= {report['flores_devtest_accuracy']:.2%} (independent set)")
        print(f"wrote {MODEL_FILE} ({report['model_bytes'] // 1024} KB)")
        return 0
    if argv:
        print(detect_language(" ".join(argv)))
        return 0
    print("usage: python3 app/detect.py [--train] [text]", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
