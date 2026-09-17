#!/usr/bin/env python3
"""Measure waveform quality without assigning human-legibility labels.

SNR, bandwidth, clipping, and duration are useful diagnostics. They are not
proof that a person can understand a recording, so this program deliberately
does not produce clean/noisy/unintelligible cohorts. Cohorts live in the
frozen, prediction-independent manifest used by the clean evaluation.

    python3 training/probe_speech_clips.py --data data/speech/test.tsv \
        --out /tmp/speech-waveform-diagnostics.json

The output is keyed by audio SHA-256 rather than filename. FLEURS filenames
have shifted across downloads in the past while the audio bytes stayed the
same. A clip that cannot be read is a hard failure, never silently dropped.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

def probe(x, sr):
    """Return descriptive waveform metrics; make no intelligibility decision."""
    n, hop = 1024, 256
    if len(x) < n:
        return dict(snr_db=None, rolloff_hz=None,
                    clipping=float(np.mean(np.abs(x) > 0.985)))
    frames = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    power = spec ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)

    db = 10 * np.log10(power.sum(axis=1) + 1e-12)
    speech = db >= np.percentile(db, 55)
    snr = np.percentile(db, 95) - np.percentile(db, 5)

    avg = power[speech].mean(axis=0) if speech.any() else power.mean(axis=0)
    cum = np.cumsum(avg) / max(avg.sum(), 1e-12)
    rolloff = float(freqs[np.searchsorted(cum, 0.995)])
    clip = float(np.mean(np.abs(x) > 0.985))
    return dict(snr_db=float(snr), rolloff_hz=rolloff, clipping=clip)


def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            clip, said = line.split("\t", 1)
            rows.append((clip.strip(), said))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="tsv of clip<TAB>text")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import soundfile as sf
    base = args.data.resolve().parent
    rows = read_rows(args.data)[: args.limit]
    if not rows:
        print(f"no rows in {args.data}", file=sys.stderr)
        return 1

    manifest = []
    for clip, said in rows:
        wav = (base / clip) if not Path(clip).is_absolute() else Path(clip)
        try:
            x, sr = sf.read(str(wav), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            dur = len(x) / sr
            m = probe(x, sr)
        except Exception as exc:
            print(f"cannot read {wav}: {exc} — diagnostics with holes refused",
                  file=sys.stderr)
            return 1
        manifest.append({
            "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            "transcript_sha256": hashlib.sha256(said.encode("utf-8")).hexdigest(),
            "duration_s": round(dur, 3), "sample_rate": sr,
            "snr_db": None if m["snr_db"] is None else round(m["snr_db"], 2),
            "rolloff_hz": None if m["rolloff_hz"] is None else round(m["rolloff_hz"]),
            "clipping": round(m["clipping"], 6),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": 1,
        "purpose": "diagnostic metadata only; never a human-legibility classifier",
        "n": len(manifest), "clips": manifest,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"wrote diagnostics for {len(manifest)} clips to {args.out}")
    print("no legibility labels were inferred from SNR, bandwidth, clipping, or duration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
