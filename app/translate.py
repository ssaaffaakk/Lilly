#!/usr/bin/env python3
"""Lilly's translation engine — Bosnian -> English on this machine.

If our fine-tuned LoRA adapter exists at models/lilly-bs-en-lora/ it is applied
automatically, otherwise the untuned base is used (so everything works even
before training finishes).

Usage:
    python3 app/translate.py "Dobar dan, kako ste?"
    python3 app/translate.py            # interactive: type a line, get the translation
"""
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "Helsinki-NLP/opus-mt-tc-big-zls-en"
ADAPTER_DIR = Path(__file__).resolve().parents[1] / "models" / "lilly-bs-en-lora"

_engine = None


class Engine:
    def __init__(self):
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        if (ADAPTER_DIR / "adapter_config.json").exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR)).merge_and_unload()
            self.name = "Lilly (fine-tuned)"
        else:
            self.name = "base model (adapter not trained yet)"
        self.model = model.to(self.device).eval()

    def translate(self, text: str) -> str:
        # The model silently drops sentences when fed several at once,
        # so split into sentences and translate them as one batch.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip()) or [text]
        enc = self.tokenizer(sentences, return_tensors="pt", padding=True,
                             truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_length=256, num_beams=4)
        return " ".join(self.tokenizer.batch_decode(gen, skip_special_tokens=True))


def get_engine() -> Engine:
    """Shared lazy singleton so the web server loads the model once."""
    global _engine
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
