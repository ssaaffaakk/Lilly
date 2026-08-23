#!/usr/bin/env python3
"""Fine-tune Lilly's listening on real Bosnian audio.

The app runs its listener in CTranslate2 format, which is fast but cannot be
trained. So this does the round trip: fine-tune the trainable checkpoint, then
convert the result back into the format the app loads.

    audio + transcripts  ->  fine-tune  ->  convert  ->  models/lilly/listen/

Data: a TSV with two columns, an audio file path and what is actually said in
it. Paths may be absolute or relative to the TSV.

    recordings/001.wav<TAB>Dobar dan, kako ste?
    recordings/002.wav<TAB>Gdje je autobuska stanica?

Usage:
    python3 training/train_speech.py --data data/speech/train.tsv
    python3 training/train_speech.py --quick-test     # 2 steps on made-up audio,
                                                      # just to prove the pipeline runs

There is no large public corpus of transcribed Bosnian speech. Realistic
sources are recordings you collect yourself, or Croatian/Serbian speech as a
proxy — the three are close enough acoustically that it helps.
"""
import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LISTEN_DIR = REPO_ROOT / "models" / "lilly" / "listen"
SAMPLE_RATE = 16_000
# The trainable checkpoint. models/lilly/listen is a converted copy and cannot
# be trained, so training starts from the original; override with --base.
BASE_MODEL = os.environ.get("LILLY_SPEECH_BASE", "openai/whisper-small")


def load_audio(path: Path) -> np.ndarray:
    """Read any soundfile-readable audio as 16 kHz mono, which is all Whisper takes."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, rate).astype("float32")
    return audio


def read_tsv(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].strip():
            clip = Path(parts[0])
            rows.append(((clip if clip.is_absolute() else path.parent / clip),
                         parts[1].strip()))
    return rows


class ClipDataset(Dataset):
    def __init__(self, rows, processor, language):
        self.rows = rows
        self.processor = processor
        self.language = language

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        clip, text = self.rows[idx]
        features = self.processor.feature_extractor(
            load_audio(clip), sampling_rate=SAMPLE_RATE).input_features[0]
        labels = self.processor.tokenizer(text).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class Collator:
    """Audio is already fixed-length; only the transcripts need padding."""
    processor: object

    def __call__(self, batch):
        features = torch.tensor(np.array([b["input_features"] for b in batch]))
        labels = self.processor.tokenizer.pad(
            [{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        # the decoder prepends the start token itself, so drop a leading copy
        if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all():
            ids = ids[:, 1:]
        return {"input_features": features, "labels": ids}


def make_test_clips(out_dir: Path) -> Path:
    """Build a handful of clips with Lilly's own voice, so --quick-test needs no data."""
    sys.path.insert(0, str(REPO_ROOT))
    from app.tts import speak_to_file
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["Good morning", "Where is the station", "Thank you very much", "See you tomorrow"]
    tsv = out_dir / "train.tsv"
    with open(tsv, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            clip = out_dir / f"{i:03d}.wav"
            if not clip.exists():
                speak_to_file(line, str(clip))
            f.write(f"{clip.name}\t{line}\n")
    return tsv


def convert_for_app(trained_dir: Path, out_dir: Path) -> None:
    """Turn the fine-tuned checkpoint into the format the app loads."""
    from ctranslate2.converters import TransformersConverter

    # The converter asks for the weight type by the newer name; the pinned
    # transformers still calls it torch_dtype and hands anything else straight
    # to the model constructor, which then raises. Translate it on the way past.
    original_load = TransformersConverter.load_model

    def load_model(self, model_class, model_name_or_path, **kwargs):
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return original_load(self, model_class, model_name_or_path, **kwargs)

    TransformersConverter.load_model = load_model
    try:
        if out_dir.exists():
            backup = out_dir.with_name(out_dir.name + "-previous")
            if backup.exists():
                shutil.rmtree(backup)
            out_dir.rename(backup)
            print(f"kept the old listener at {backup}")
        TransformersConverter(
            str(trained_dir),
            copy_files=["tokenizer.json", "preprocessor_config.json"],
        ).convert(str(out_dir), quantization="int8")
    finally:
        TransformersConverter.load_model = original_load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="TSV of audio path + transcript")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--language", default="bs")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "models" / "lilly" / "listen-trained")
    ap.add_argument("--quick-test", action="store_true")
    ap.add_argument("--no-convert", action="store_true",
                    help="stop after training, leave models/lilly/listen alone")
    args = ap.parse_args()

    if args.quick_test:
        args.data = make_test_clips(REPO_ROOT / "data" / "speech-quicktest")
        args.base, args.epochs, args.batch_size = "openai/whisper-tiny", 1.0, 2
    if not args.data or not args.data.exists():
        print("need --data pointing at a TSV of audio paths and transcripts",
              file=sys.stderr)
        return 1

    rows = read_tsv(args.data)
    if not rows:
        print(f"no usable rows in {args.data}", file=sys.stderr)
        return 1
    print(f"clips: {len(rows):,}  base: {args.base}")

    processor = WhisperProcessor.from_pretrained(args.base, language=args.language,
                                                 task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(REPO_ROOT / "models" / "checkpoints-speech"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=41,
        max_steps=2 if args.quick_test else -1,
    )
    Seq2SeqTrainer(model=model, args=train_args,
                   train_dataset=ClipDataset(rows, processor, args.language),
                   data_collator=Collator(processor)).train()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    processor.save_pretrained(str(args.output))
    # the app's listener wants the fast tokenizer's single tokenizer.json
    from transformers import WhisperTokenizerFast
    WhisperTokenizerFast.from_pretrained(args.base).save_pretrained(str(args.output))
    print(f"trained checkpoint: {args.output}")

    if args.no_convert:
        print("skipped the conversion; the app still uses the old listener")
        return 0
    convert_for_app(args.output, LISTEN_DIR)
    print(f"the app now listens with this model: {LISTEN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
