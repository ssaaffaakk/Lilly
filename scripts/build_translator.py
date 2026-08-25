#!/usr/bin/env python3
"""Build the translator the app actually serves from.

Training produces PyTorch weights in float32 — 915 MB of parameters for a model
whose published weights are float16, because from_pretrained widens them. Serving
from that costs about 1.8 GB of resident memory and is roughly twice as slow as it
needs to be. The listener in this same app has always run as a quantised
CTranslate2 model; this does the same for translation.

Measured on this model: 1,805 MB resident becomes 670 MB, inference runs about
twice as fast, and chrF2 is unchanged to two decimal places.

    python3 scripts/build_translator.py

If models/lilly/adapter/ exists, the fine-tuning is merged into the weights first,
so the served model is Lilly rather than the untuned base. Run this again after
every training run — the adapter cannot be applied at serving time once the
weights are quantised.
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "models" / "lilly"
SOURCE = MODELS / "translate"          # transformers format, what training reads
ADAPTER = MODELS / "adapter"           # our fine-tuning, if it has been trained
DEST = MODELS / "translator"           # quantised, what the app serves
# the app needs these beside the weights to turn text into tokens and back
TOKENIZER_FILES = ("source.spm", "target.spm", "vocab.json",
                   "tokenizer_config.json", "special_tokens_map.json")


def merge_adapter(into: Path) -> bool:
    """Fold the fine-tuning into the weights. Quantised models cannot take it later."""
    if not (ADAPTER / "adapter_config.json").exists():
        return False
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print(f"merging {ADAPTER}")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(SOURCE))
    model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload()
    model.save_pretrained(str(into))
    AutoTokenizer.from_pretrained(str(SOURCE)).save_pretrained(str(into))
    return True


def convert(source: Path, dest: Path, quantization: str) -> None:
    from ctranslate2.converters import TransformersConverter

    # The converter names the weight type differently from older transformers,
    # which hands anything it does not recognise to the model constructor and
    # raises. Try it their way first and translate only if that fails.
    original = TransformersConverter.load_model

    def load_model(self, model_class, path, **kwargs):
        try:
            return original(self, model_class, path, **kwargs)
        except TypeError as exc:
            if "dtype" not in kwargs or "dtype" not in str(exc):
                raise
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            return original(self, model_class, path, **kwargs)

    TransformersConverter.load_model = load_model
    try:
        if dest.exists():
            shutil.rmtree(dest)
        TransformersConverter(str(source)).convert(str(dest), quantization=quantization)
    finally:
        TransformersConverter.load_model = original


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantization", default="int8",
                    help="int8 (default), int8_float32, float16 or float32")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()

    if not (args.source / "config.json").exists():
        print(f"no model at {args.source} — run scripts/fetch_models.py first",
              file=sys.stderr)
        return 1

    source = args.source
    merged_dir = MODELS / "translate-merged"
    if merge_adapter(merged_dir):
        source = merged_dir
        print("serving the fine-tuned weights")
    else:
        print("no adapter yet — serving the untuned base")

    convert(source, args.dest, args.quantization)
    for name in TOKENIZER_FILES:
        found = source / name
        if not found.exists():
            found = SOURCE / name
        if found.exists():
            shutil.copy(found, args.dest / name)

    # Record what went in, so the app can say honestly which model it is serving
    # rather than guessing from whether an adapter folder happens to exist.
    import json
    (args.dest / "built.json").write_text(json.dumps(
        {"fine_tuned": source == merged_dir, "quantization": args.quantization}))

    if source == merged_dir:
        shutil.rmtree(merged_dir, ignore_errors=True)

    size = sum(f.stat().st_size for f in args.dest.rglob("*") if f.is_file())
    print(f"built {args.dest}: {size / 1048576:.0f} MB ({args.quantization})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
