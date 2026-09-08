#!/usr/bin/env python3
"""How well a voice is understood: sentences spoken by a voice, heard by the listener.

Nobody on this project can score Bosnian speech by ear, so the instrument is
the shipped listener -- app.speech.transcribe, the product's own path. Each
sentence is synthesized by each voice, the listener writes down what it heard,
and the word error rate against the sentence is the voice's intelligibility.
The human recordings of the same sentences go through the same listener and
are the ceiling: a voice cannot be expected to be heard better than people
are. It is intelligibility and nothing else: a voice can be perfectly
understood and still sound wrong, and no number here says otherwise.

    python3 training/evaluate_speak.py --tsv data/speech/test.tsv --clips first200 \\
        --voice before=models/lilly/speak-bs/voice.onnx:0 \\
        --voice candidate=/kaggle/temp/cand/voice.onnx:3 \\
        --human --json out.json

`--voice name=path.onnx[:speaker]`; the Piper config is path.onnx.json beside
it. `--clips first200` is the 200-clip prefix of the test split behind every
published speech number (167 distinct sentences); `distinct` is one clip per
sentence over the whole file; `--limit` caps either. The paired bootstrap
resamples sentences, as training/speech_bench.py does, between every voice and
the first one named.

What stops the run: a voice that renders a sentence as silence (under 0.2 s),
a listener that hears nothing in every clip of a voice (100% on every sentence
is a broken path, not a measurement), fewer sentences than were asked for.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.cluster_speakers import read_tsv  # noqa: E402
from training.evaluate_speech import edits, normalise  # noqa: E402

MIN_SECONDS = 0.2
SEED = 11   # onnxruntime's global seed for the voice's own noise; the bootstrap's seed is speech_bench's


def choose(rows: list, mode: str, limit: int) -> list:
    """The clips to score. first200 is speech_bench's prefix; distinct is one per sentence."""
    if mode == "first200":
        picked = rows[:200]
    else:
        seen, picked = set(), []
        for clip, text in rows:
            if text not in seen:
                seen.add(text)
                picked.append((clip, text))
    return picked[:limit] if limit else picked


def parse_voice(spec: str) -> tuple:
    """name=path.onnx[:speaker] -> (name, Path, speaker id or None)."""
    name, _, rest = spec.partition("=")
    if not name or not rest:
        raise SystemExit(f"--voice wants name=path.onnx[:speaker], not {spec!r}")
    path, _, speaker = rest.rpartition(":")
    if not path or not speaker.isdigit():
        path, speaker = rest, None
    return name, Path(path), (int(speaker) if speaker is not None else None)


def synthesize(onnx: Path, speaker, sentences: list, out_dir: Path) -> list:
    """One WAV per sentence with Piper, the engine the app speaks Bosnian with."""
    import wave
    import onnxruntime
    from piper import PiperVoice, SynthesisConfig
    config = onnx.with_name(onnx.name + ".json")
    if not (onnx.is_file() and config.is_file()):
        raise SystemExit(f"no voice at {onnx} (+ .json)")
    # A VITS voice samples its noise inside the graph, so the same sentence is
    # never quite the same audio twice: two runs of the before voice on the same
    # two sentences were heard with 13 and 20 words wrong. The judgment has to
    # be repeatable, so onnxruntime's random seed is fixed before the session
    # is built; every voice is then drawn from the same seed.
    onnxruntime.set_seed(SEED)
    voice = PiperVoice.load(onnx, config)
    if speaker is not None and voice.config.num_speakers <= speaker:
        raise SystemExit(f"{onnx} has {voice.config.num_speakers} speakers; no speaker {speaker}")
    syn = SynthesisConfig(speaker_id=speaker)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, text in enumerate(sentences):
        path = out_dir / f"{i:04d}.wav"
        with wave.open(str(path), "wb") as wav:
            voice.synthesize_wav(text, wav, syn_config=syn)
        with wave.open(str(path)) as wav:
            seconds = wav.getnframes() / wav.getframerate()
        if seconds < MIN_SECONDS:
            raise SystemExit(f"{onnx} rendered sentence {i} as {seconds:.2f} s of audio: "
                             f"{text[:60]!r} -- a voice that falls silent is not scored")
        clips.append((str(path), seconds))
    return clips


def listen(paths: list, listener: Path, language: str = "bs") -> list:
    from app.speech import transcribe
    out = []
    for i, path in enumerate(paths):
        out.append(transcribe(path, language=language, build=listener))
        if (i + 1) % 50 == 0:
            print(f"    heard {i + 1}/{len(paths)}", flush=True)
    return out


def marks_of(pairs: list) -> list:
    """[(sentence, hypothesis)] -> per-clip edit counts keyed by sentence."""
    out = []
    for sentence, hyp in pairs:
        ref = normalise(sentence)
        out.append({"sentence": sentence, "edits": edits(normalise(hyp), ref), "words": len(ref)})
    return out


def wer(marks: list) -> float:
    return 100.0 * sum(m["edits"] for m in marks) / max(sum(m["words"] for m in marks), 1)


def summarize(marks: list, extra: dict = None) -> dict:
    rec = {"wer": round(wer(marks), 2), "edits": sum(m["edits"] for m in marks),
           "words": sum(m["words"] for m in marks), "n": len(marks)}
    rec.update(extra or {})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=REPO_ROOT / "data" / "speech" / "test.tsv")
    ap.add_argument("--clips", choices=("first200", "distinct"), default="first200")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--voice", action="append", default=[], help="name=path.onnx[:speaker]")
    ap.add_argument("--human", action="store_true",
                    help="also hear the recordings themselves, the ceiling")
    ap.add_argument("--listener", type=Path, default=REPO_ROOT / "models" / "lilly" / "listen")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--keep", type=Path, default=None, help="keep the synthesized clips here")
    args = ap.parse_args()
    if not args.voice and not args.human:
        raise SystemExit("nothing to score: name a --voice or ask for --human")

    rows = read_tsv(args.tsv)
    chosen = choose(rows, args.clips, args.limit)
    if not chosen or (args.limit and len(chosen) < args.limit):
        raise SystemExit(f"{args.tsv}: {len(chosen)} clips chosen, asked for {args.limit or 'all'}")
    sentences = list(dict.fromkeys(text for _, text in chosen))
    print(f"{len(chosen)} clips, {len(sentences)} distinct sentences, listener {args.listener}")

    from scripts.fetch_models import listen_fingerprint
    from scripts.guard import claim
    claim(2.0, "voice scoring")
    record = {"tsv": str(args.tsv), "clips": args.clips, "limit": args.limit,
              "n_clips": len(chosen), "n_sentences": len(sentences),
              "listener": {"path": str(args.listener),
                           "fingerprint": listen_fingerprint(args.listener)},
              "voices": {}, "human": None, "paired": {}, "hypotheses": {}}
    marks = {}
    work = args.keep or Path(tempfile.mkdtemp(prefix="lilly-speak-"))
    for spec in args.voice:
        name, onnx, speaker = parse_voice(spec)
        t = time.time()
        clips = synthesize(onnx, speaker, sentences, work / name)
        print(f"  {name}: {len(clips)} sentences spoken in {time.time() - t:.0f} s")
        hyps = listen([p for p, _ in clips], args.listener)
        if not any(h.strip() for h in hyps):
            raise SystemExit(f"{name}: the listener heard nothing in any of {len(hyps)} clips -- "
                             "a broken path, not a score")
        marks[name] = marks_of(list(zip(sentences, hyps)))
        record["voices"][name] = summarize(marks[name], {
            "onnx": str(onnx), "speaker": speaker,
            "seconds_audio": round(sum(s for _, s in clips), 1)})
        record["hypotheses"][name] = list(zip(sentences, hyps))
        print(f"  {name}: {record['voices'][name]['wer']:.1f}% word error "
              f"({record['voices'][name]['edits']} of {record['voices'][name]['words']} words)")
    if args.human:
        paths = [str((args.tsv.parent / clip).resolve()) for clip, _ in chosen]
        hyps = listen(paths, args.listener)
        marks["human"] = marks_of([(text, hyp) for (_, text), hyp in zip(chosen, hyps)])
        record["human"] = summarize(marks["human"], {"n_clips": len(chosen)})
        record["hypotheses"]["human"] = [[text, hyp] for (_, text), hyp in zip(chosen, hyps)]
        print(f"  human recordings: {record['human']['wer']:.1f}% word error "
              f"({record['human']['edits']} of {record['human']['words']} words)")

    names = [parse_voice(s)[0] for s in args.voice]
    if len(names) >= 2:
        from training.speech_bench import paired_bootstrap
        first = names[0]
        for other in names[1:]:
            delta, p = paired_bootstrap(marks[first], marks[other], wer)
            record["paired"][f"{other} vs {first}"] = {"delta_points": round(delta, 2), "p": p}
            print(f"  {other} vs {first}: {delta:+.2f} points, p = {p:.4f}")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
