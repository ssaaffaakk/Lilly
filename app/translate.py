#!/usr/bin/env python3
"""Lilly's translation engine — Bosnian -> English, and English -> Bosnian back.

Serves the quantised models in models/lilly/translator/ (bs-en) and
models/lilly/translator-en-bs/ (en-bs), built by scripts/build_translator.py.
That build is where our fine-tuning gets folded in, so what runs here is Lilly
rather than the untuned base once the adapter exists.

The two directions are separate models, separate builds and separate loads.
en-bs additionally needs a target label on every sentence -- see DIRECTIONS.

Quantised on purpose: the same weights served as float32 PyTorch cost about
1.8 GB resident and run roughly half as fast, for a chrF2 that matches to two
decimal places.

Usage:
    python3 app/translate.py "Dobar dan, kako ste?"
    python3 app/translate.py --direction en-bs "Where is the bus station?"
    python3 app/translate.py            # interactive: type a line, get the translation
"""
import json
import re
import sys
import threading

from app.lilly import BadInput, TRANSLATOR_DIR, TRANSLATOR_EN_BS_DIR

# Cost is driven by sentences x longest sentence, not by how many characters
# arrived, so the limits are in tokens. Every sentence in a batch is padded to
# the longest one, so a single wide batch is what turns a modest paste into
# gigabytes.
MAX_INPUT_TOKENS = 2048      # one request may not ask for more work than this
BATCH_TOKEN_BUDGET = 256     # sentences x padded length per batch
MAX_SENTENCE_TOKENS = 256

# Split on line breaks as well as on sentence-ending punctuation. Text off a
# photograph rarely has punctuation — a sign reads "ZABRANJEN ULAZ / Radovi na
# mostu / Hvala na razumijevanju", three separate notices on three lines — and
# joined into one run-on the model translates it as a single sentence and drops
# the opening clause entirely. Measured on exactly that sign: the prohibition
# vanished from the translation, so the reader never saw that entry was
# forbidden. The lines are the sentence boundaries the punctuation is missing.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\s*\n+\s*")

# Each direction is its own build, its own tokenizer and its own target label.
# The label is not decoration: this base decodes five South Slavic languages and
# picks between them from the front of the source sentence, so a missing label
# returns fluent Croatian to somebody who asked for Bosnian.
DIRECTIONS = {
    "bs-en": {"dir": TRANSLATOR_DIR, "tag": None,
              "reads": "Bosnian", "writes": "English"},
    "en-bs": {"dir": TRANSLATOR_EN_BS_DIR, "tag": ">>bos_Latn<<",
              "reads": "English", "writes": "Bosnian"},
}

_engines = {}
_engine_lock = threading.Lock()
# Translation is CPU-bound and the server hands requests to a wide threadpool.
# Without this, ten callers each get their own copy of the peak allocation.
_translate_lock = threading.Lock()


class TextTooLong(BadInput):
    """Raised when a request asks for more work than one translation may do."""


class Engine:
    def __init__(self, directory=None, direction="bs-en"):
        """directory lets a measurement load a different build of the same model.

        Scoring has to go through this class, not around it: the sentence
        splitting below is part of what Lilly does, and a measurement that skips
        it scores a failure mode the app does not have. Measured on FLORES,
        feeding rows in whole: single-sentence rows lost 0.33 chrF2 to the
        fine-tuning, multi-sentence rows lost 3.36 — the model drops a clause
        when it is handed several at once, which is the very thing the splitter
        exists to prevent.
        """
        import ctranslate2
        from transformers import AutoTokenizer

        spec = DIRECTIONS[direction]
        directory = directory or spec["dir"]
        if not (directory / "model.bin").exists():
            raise FileNotFoundError(
                f"no {direction} translator at {directory} — run "
                f"scripts/build_translator.py --direction {direction}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory))
        self.device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        self.translator = ctranslate2.Translator(
            str(directory), device=self.device, compute_type="int8")
        self.direction = direction
        self.tag = spec["tag"]
        if self.tag and self.tokenizer.convert_tokens_to_ids(
                [self.tag])[0] in (None, self.tokenizer.unk_token_id):
            # Refuse at load rather than answer in the wrong language all day.
            raise FileNotFoundError(
                f"the build at {directory} does not know {self.tag}, so it "
                f"cannot be held to Bosnian — rebuild it from the {direction} base")
        built = directory / "built.json"
        tuned = json.loads(built.read_text())["fine_tuned"] if built.exists() else False
        self.name = "Lilly (fine-tuned)" if tuned else "base model (not fine-tuned yet)"

    def translate(self, text: str, truncate: bool = False) -> str:
        """Source language in, target language out, per this engine's direction.

        The model silently drops sentences when fed several at once, so the text
        is split and translated sentence by sentence. Those sentences go out in
        small groups rather than one wide batch: peak memory is set by the widest
        batch, and an unbounded one is how a paste becomes an out-of-memory kill.

        truncate=True quietly drops the overflow instead of refusing. That is for
        text the caller never typed — a photo of a dense page, a long recording —
        where a refusal would be baffling.
        """
        sentences = [s for s in SENTENCE_BREAK.split(text.strip()) if s.strip()] or [text]
        # Every sentence carries the label, not just the first one: the splitter
        # above means each sentence is its own decode, and a label on the first
        # of five steers only the first of five.
        if self.tag:
            sentences = [f"{self.tag} {s}" for s in sentences]
        tokenised = [self.tokenizer.convert_ids_to_tokens(
                        self.tokenizer.encode(s, truncation=True,
                                              max_length=MAX_SENTENCE_TOKENS))
                     for s in sentences]

        total = 0
        for i, tokens in enumerate(tokenised):
            if total + len(tokens) > MAX_INPUT_TOKENS:
                if not truncate:
                    raise TextTooLong(
                        f"that is about {sum(len(t) for t in tokenised)} tokens of "
                        f"text; this translates up to {MAX_INPUT_TOKENS} at a time")
                tokenised = tokenised[:i]
                break
            total += len(tokens)
        if not tokenised:
            return ""

        out = []
        for group in self._grouped(tokenised):
            with _translate_lock:
                results = self.translator.translate_batch(
                    group, beam_size=4, max_decoding_length=MAX_SENTENCE_TOKENS)
            for result in results:
                ids = self.tokenizer.convert_tokens_to_ids(result.hypotheses[0])
                out.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return " ".join(out)

    @staticmethod
    def _grouped(tokenised: list):
        """Batches whose cost — count x padded width — stays under the budget."""
        group, widest = [], 0
        for tokens in tokenised:
            candidate = max(widest, len(tokens))
            if group and (len(group) + 1) * candidate > BATCH_TOKEN_BUDGET:
                yield group
                group, widest = [tokens], len(tokens)
            else:
                group.append(tokens)
                widest = candidate
        if group:
            yield group


def get_engine(direction: str = "bs-en") -> Engine:
    """Shared lazy singleton per direction, so the web server loads each once.

    Locked, because the server answers requests from a threadpool: without it a
    first burst of traffic starts several loads at once. Per direction rather
    than one global, because the reply side is a different model and loading it
    should not cost anybody who only ever reads Bosnian.
    """
    if direction not in _engines:
        with _engine_lock:
            if direction not in _engines:
                _engines[direction] = Engine(direction=direction)
    return _engines[direction]


def main() -> int:
    argv = sys.argv[1:]
    direction = "bs-en"
    if argv and argv[0] == "--direction":
        direction, argv = argv[1], argv[2:]
    engine = get_engine(direction)
    spec = DIRECTIONS[direction]
    print(f"[model: {engine.name}, device: {engine.device}, "
          f"{spec['reads']} -> {spec['writes']}]", file=sys.stderr)
    if argv:
        print(engine.translate(" ".join(argv)))
        return 0
    print(f"Type {spec['reads']}, press Enter (Ctrl-D to quit):", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(engine.translate(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
