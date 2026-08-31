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
# Each direction is a separate base, a separate adapter and a separate served
# build. They are named together here so a reverse-direction adapter can never
# be merged into the forward base -- peft applies it without complaint, and the
# result translates confidently in the wrong direction.
DIRECTIONS = {
    "bs-en": {"source": SOURCE, "adapter": ADAPTER, "dest": DEST},
    "en-bs": {"source": MODELS / "translate-en-bs",
              "adapter": MODELS / "adapter-en-bs",
              "dest": MODELS / "translator-en-bs"},
}
# the app needs these beside the weights to turn text into tokens and back
TOKENIZER_FILES = ("source.spm", "target.spm", "vocab.json",
                   "tokenizer_config.json", "special_tokens_map.json")


def merge_adapter(into: Path, adapter: Path, source: Path) -> bool:
    """Fold the fine-tuning into the weights. Quantised models cannot take it later."""
    if not (adapter / "adapter_config.json").exists():
        return False
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print(f"merging {adapter} into {source.name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(source))
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(into))
    AutoTokenizer.from_pretrained(str(source)).save_pretrained(str(into))
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
    ap.add_argument("--direction", default="bs-en", choices=sorted(DIRECTIONS),
                    help="which direction to build; sets source, adapter and dest")
    ap.add_argument("--quantization", default="int8",
                    help="int8 (default), int8_float32, float16 or float32")
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--dest", type=Path, default=None)
    # Comparing the fine-tuned model against the base is only honest if both are
    # built the same way. Serving one as int8 CTranslate2 and scoring the other
    # as float32 PyTorch measures the quantisation as much as the training.
    ap.add_argument("--no-adapter", action="store_true",
                    help="build the untuned base, for comparing against")
    # Naming the adapter lets a candidate be built and scored without displacing
    # the model being served. A retrain that turns out worse should cost nothing.
    ap.add_argument("--adapter", type=Path, default=None,
                    help="use this adapter instead of models/lilly/adapter")
    args = ap.parse_args()
    spec = DIRECTIONS[args.direction]
    base = args.source or spec["source"]
    dest = args.dest or spec["dest"]
    adapter = args.adapter or spec["adapter"]

    if not (base / "config.json").exists():
        hint = ("scripts/fetch_models.py" if args.direction == "bs-en" else
                f"scripts/fetch_translate_base.py --direction {args.direction}")
        print(f"no model at {base} — run {hint} first", file=sys.stderr)
        return 1

    source = base
    merged_dir = MODELS / f"translate-merged-{args.direction}"
    if args.no_adapter:
        print("building the untuned base on purpose")
    elif merge_adapter(merged_dir, adapter, base):
        source = merged_dir
        print("serving the fine-tuned weights")
    else:
        print("no adapter yet — serving the untuned base")

    convert(source, dest, args.quantization)
    for name in TOKENIZER_FILES:
        found = source / name
        if not found.exists():
            found = base / name
        if found.exists():
            shutil.copy(found, dest / name)

    # Record what went in, so the app can say honestly which model it is serving
    # rather than guessing from whether an adapter folder happens to exist.
    import json
    (dest / "built.json").write_text(json.dumps(
        {"fine_tuned": source == merged_dir, "quantization": args.quantization,
         "direction": args.direction}))

    if source == merged_dir:
        shutil.rmtree(merged_dir, ignore_errors=True)

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"built {dest}: {size / 1048576:.0f} MB ({args.quantization})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
