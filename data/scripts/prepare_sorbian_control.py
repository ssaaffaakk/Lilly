#!/usr/bin/env python3
"""The sr_RS voice's own training set, rebuilt for the control run.

Piper's `sr_RS-serbski_institut-medium` was trained on 747 utterances -- 408
Lower Sorbian (speaker dsb, 0) and 339 Upper Sorbian (speaker hsb, 1) -- and
its checkpoint repository carries the list: `dataset.jsonl.gz`, one line per
utterance with the text, the phonemes it was trained with, the speaker and the
source file's name. The recordings are public (Sorbian Institute, CC BY-NC-SA
4.0): one FLAC per language at 44.1 kHz and a YAML that names each prompt code
with its start and end inside the FLAC. Every one of the 747 is found by its
code and its text matches the YAML's exactly (checked 10 September 2026).

This cuts those 747 utterances out of the two FLACs, writes them as WAV at
the source rate, and writes the metadata.csv piper.train reads -- speaker
names "dsb" then "hsb", in that order, so Piper numbers them 0 and 1 as the
checkpoint does and the warm start copies the speaker table too. Nothing is
chosen: the control trains on exactly what the voice was trained on.

It also phonemizes every text with espeak-ng's `sr` and with `bs` and counts
how many utterances come out identical to the phonemes stored in the list --
a free check of whether this box's phonemizer is the one the voice knows.

    python3 data/scripts/prepare_sorbian_control.py --list dataset.jsonl.gz \\
        --dsb-flac dsb.flac --dsb-yaml dsb.yaml --hsb-flac hsb.flac --hsb-yaml hsb.yaml --out /kaggle/temp/sorbian
"""
import argparse
import collections
import gzip
import json
import os
import sys
from pathlib import Path

SPEAKERS = ("dsb", "hsb")   # the checkpoint's order: dsb is 0, hsb is 1
MIN_SEC = 0.3


def read_list(path: Path) -> list:
    rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    for r in rows:
        r["code"] = os.path.splitext(os.path.basename(r["audio_path"]))[0]
    return rows


def read_yaml(path: Path) -> dict:
    import yaml
    return {x["prompt"]: x for x in yaml.safe_load(open(path, encoding="utf-8"))}


def cut(rows: list, prompts: dict, flac: Path, speaker: str, clips: Path) -> list:
    """(wav path, speaker, text) for this speaker's utterances, cut from the FLAC."""
    import soundfile as sf
    info = sf.info(str(flac))
    out = []
    with sf.SoundFile(str(flac)) as fh:
        for r in rows:
            if r["speaker"] != speaker:
                continue
            p = prompts.get(r["code"])
            if p is None:
                raise SystemExit(f"{speaker} {r['code']}: not in the YAML -- the release is not the one the voice was cut from")
            if p["text"].strip() != r["text"].strip():
                raise SystemExit(f"{speaker} {r['code']}: the YAML's text differs from the training list's")
            start, end = int(round(p["start"] * info.samplerate)), int(round(p["end"] * info.samplerate))
            if end - start < MIN_SEC * info.samplerate:
                raise SystemExit(f"{speaker} {r['code']}: {p['end'] - p['start']:.2f} s -- not an utterance")
            fh.seek(start)
            audio = fh.read(end - start, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            path = clips / f"{speaker}-{r['code']}.wav"
            sf.write(str(path), audio, info.samplerate, subtype="PCM_16")
            out.append((str(path.resolve()), speaker, r["text"].replace("|", " ").strip()))
    return out


def phoneme_agreement(rows: list) -> dict:
    """How many stored phoneme lists this box's espeak reproduces, per voice."""
    from piper.phonemize_espeak import EspeakPhonemizer
    ph = EspeakPhonemizer()
    out = {}
    for voice in ("sr", "bs"):
        same = 0
        for r in rows:
            mine = [p for sentence in ph.phonemize(voice, r["text"]) for p in sentence]
            same += mine == r["phonemes"]
        out[voice] = {"identical": same, "of": len(rows), "share": round(same / len(rows), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True, help="dataset.jsonl.gz from the checkpoint repository")
    ap.add_argument("--dsb-flac", type=Path, required=True)
    ap.add_argument("--dsb-yaml", type=Path, required=True)
    ap.add_argument("--hsb-flac", type=Path, required=True)
    ap.add_argument("--hsb-yaml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = read_list(args.list)
    counts = collections.Counter(r["speaker"] for r in rows)
    if tuple(sorted(counts)) != SPEAKERS or len(rows) != 747:
        raise SystemExit(f"the training list holds {dict(counts)} -- not the 747 (408 dsb + 339 hsb) of the voice")
    clips = args.out / "train"
    clips.mkdir(parents=True, exist_ok=True)
    kept = []
    for speaker, flac, yml in (("dsb", args.dsb_flac, args.dsb_yaml), ("hsb", args.hsb_flac, args.hsb_yaml)):
        got = cut(rows, read_yaml(yml), flac, speaker, clips)
        print(f"  {speaker}: {len(got)} utterances cut from {flac.name}", flush=True)
        kept.extend(got)
    if len(kept) != 747:
        raise SystemExit(f"{len(kept)} utterances cut, not 747")
    csv = args.out / "metadata.csv"
    with csv.open("w", encoding="utf-8") as fh:
        for path, speaker, text in kept:
            fh.write(f"{path}|{speaker}|{text}\n")
    agreement = phoneme_agreement(rows)
    import soundfile as sf
    seconds = sum(sf.info(p).duration for p, _, _ in kept)
    manifest = {"rows": len(kept), "csv": str(csv), "minutes": round(seconds / 60, 1),
                "speakers": {s: counts[s] for s in SPEAKERS}, "phoneme_agreement": agreement}
    Path(str(csv) + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"{manifest['rows']} rows, {manifest['minutes']} minutes -> {csv}")
    print("phonemes identical to the voice's own, by espeak voice:", agreement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
