#!/usr/bin/env python3
"""Lilly's translation engine — Bosnian -> English on this machine.

Weights are read from models/lilly/translate/. If our fine-tuned adapter
exists at models/lilly/adapter/ it is applied on top automatically, otherwise
the untuned base is used — so everything works even before training finishes.

Usage:
    python3 app/translate.py "Dobar dan, kako ste?"
    python3 app/translate.py            # interactive: type a line, get the translation
"""
import re
import sys
import threading

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.lilly import ADAPTER_DIR, TRANSLATE_DIR, BadInput

# Cost is driven by sentences x longest sentence, not by how many characters
# arrived, so the limits are in tokens. Beam search widens each batch four
# times over, and every sentence in a batch is padded to the longest one, so a
# single wide batch is what turns a modest paste into gigabytes.
# Measured on this repo, CPU-only, translating 200 sentences: a budget of 1024
# adds 1,364 MB over the resting model, 512 adds 718 MB, and 256 adds 178 MB for
# about 20% more time. Predictable memory is worth more than the seconds.
MAX_INPUT_TOKENS = 2048      # one request may not ask for more work than this
BATCH_TOKEN_BUDGET = 256     # sentences x padded length per generate() call
MAX_SENTENCE_TOKENS = 256

_engine = None
_engine_lock = threading.Lock()
# Translation is CPU-bound and the server hands requests to a wide threadpool.
# Without this, ten callers each get their own copy of the peak allocation.
_generate_lock = threading.Lock()


class TextTooLong(BadInput):
    """Raised when a request asks for more work than one translation may do."""


class Engine:
    def __init__(self):
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(TRANSLATE_DIR))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(TRANSLATE_DIR))
        if (ADAPTER_DIR / "adapter_config.json").exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR)).merge_and_unload()
            self.name = "Lilly (fine-tuned)"
        else:
            self.name = "base model (adapter not trained yet)"
        self.model = model.to(self.device).eval()

    def translate(self, text: str, truncate: bool = False) -> str:
        """Bosnian in, English out.

        The model silently drops sentences when fed several at once, so the
        text is split and translated sentence by sentence. Those sentences go
        out in small groups rather than one wide batch: peak memory is set by
        the widest batch, and an unbounded one is how a paste becomes an
        out-of-memory kill.

        truncate=True quietly drops the overflow instead of refusing. That is
        for text the caller never typed — a photo of a dense page, a long
        recording — where a refusal would be baffling.
        """
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s] or [text]
        lengths = [len(self.tokenizer(s, truncation=True,
                                      max_length=MAX_SENTENCE_TOKENS).input_ids)
                   for s in sentences]

        total = 0
        for i, length in enumerate(lengths):
            if total + length > MAX_INPUT_TOKENS:
                if not truncate:
                    raise TextTooLong(
                        f"that is about {sum(lengths)} tokens of text; "
                        f"this translates up to {MAX_INPUT_TOKENS} at a time")
                sentences, lengths = sentences[:i], lengths[:i]
                break
            total += length
        if not sentences:
            return ""

        out = []
        for group in self._grouped(sentences, lengths):
            enc = self.tokenizer(group, return_tensors="pt", padding=True,
                                 truncation=True,
                                 max_length=MAX_SENTENCE_TOKENS).to(self.device)
            with _generate_lock, torch.no_grad():
                gen = self.model.generate(**enc, max_length=MAX_SENTENCE_TOKENS,
                                          num_beams=4)
            out.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return " ".join(out)

    @staticmethod
    def _grouped(sentences: list, lengths: list):
        """Batches whose cost — count x padded width — stays under the budget."""
        group, widest = [], 0
        for sentence, length in zip(sentences, lengths):
            candidate = max(widest, length)
            if group and (len(group) + 1) * candidate > BATCH_TOKEN_BUDGET:
                yield group
                group, widest = [sentence], length
            else:
                group.append(sentence)
                widest = candidate
        if group:
            yield group


def get_engine() -> Engine:
    """Shared lazy singleton so the web server loads the model once.

    Locked, because the server answers requests from a threadpool: without it
    a first burst of traffic starts several loads at once and each one wants
    its own gigabyte.
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
