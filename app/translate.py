#!/usr/bin/env python3
"""Lilly's translation engine — Bosnian -> English on this machine.

Serves the quantised model in models/lilly/translator/, built by
scripts/build_translator.py. That build is where our fine-tuning gets folded in,
so what runs here is Lilly rather than the untuned base once the adapter exists.

Quantised on purpose: the same weights served as float32 PyTorch cost about
1.8 GB resident and run roughly half as fast, for a chrF2 that matches to two
decimal places.

Usage:
    python3 app/translate.py "Dobar dan, kako ste?"
    python3 app/translate.py            # interactive: type a line, get the translation
"""
import json
import re
import sys
import threading

from app.lilly import BadInput, TRANSLATOR_DIR

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

_engine = None
_engine_lock = threading.Lock()
# Translation is CPU-bound and the server hands requests to a wide threadpool.
# Without this, ten callers each get their own copy of the peak allocation.
_translate_lock = threading.Lock()


class TextTooLong(BadInput):
    """Raised when a request asks for more work than one translation may do."""


class Engine:
    def __init__(self):
        import ctranslate2
        from transformers import AutoTokenizer

        if not (TRANSLATOR_DIR / "model.bin").exists():
            raise FileNotFoundError(
                f"no translator at {TRANSLATOR_DIR} — run scripts/build_translator.py")
        self.tokenizer = AutoTokenizer.from_pretrained(str(TRANSLATOR_DIR))
        self.device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        self.translator = ctranslate2.Translator(
            str(TRANSLATOR_DIR), device=self.device, compute_type="int8")
        built = TRANSLATOR_DIR / "built.json"
        tuned = json.loads(built.read_text())["fine_tuned"] if built.exists() else False
        self.name = "Lilly (fine-tuned)" if tuned else "base model (not fine-tuned yet)"

    def translate(self, text: str, truncate: bool = False) -> str:
        """Bosnian in, English out.

        The model silently drops sentences when fed several at once, so the text
        is split and translated sentence by sentence. Those sentences go out in
        small groups rather than one wide batch: peak memory is set by the widest
        batch, and an unbounded one is how a paste becomes an out-of-memory kill.

        truncate=True quietly drops the overflow instead of refusing. That is for
        text the caller never typed — a photo of a dense page, a long recording —
        where a refusal would be baffling.
        """
        sentences = [s for s in SENTENCE_BREAK.split(text.strip()) if s.strip()] or [text]
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


def get_engine() -> Engine:
    """Shared lazy singleton so the web server loads the model once.

    Locked, because the server answers requests from a threadpool: without it a
    first burst of traffic starts several loads at once.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = Engine()
    return _engine


def main() -> int:
    engine = get_engine()
    print(f"[model: {engine.name}, device: {engine.device}]", file=sys.stderr)
    if len(sys.argv) > 1:
        print(engine.translate(" ".join(sys.argv[1:])))
        return 0
    print("Type Bosnian, press Enter (Ctrl-D to quit):", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(engine.translate(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
