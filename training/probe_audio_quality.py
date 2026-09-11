"""Probe the YouTube lecture audio for TTS suitability.

Three things decide whether found audio can train a voice:
  bandwidth  -- a 22.05 kHz voice needs real content up to ~11 kHz
  noise      -- the gap between the noise floor and speech level
  dryness    -- reverberation fills the pauses between syllables; close-mic
                speech keeps them deep. Measured as the dB spread between the
                95th and 10th percentile of frame energy inside speech.
"""
import json
import sys
from pathlib import Path

import av
import numpy as np

SRC = Path("data/speech-extra/youtube-voice")
MANIFEST = json.loads((SRC / "manifest.json").read_text())
SECONDS = 90          # sampled from the middle of each file
SR = 22050


def decode_middle(path, seconds=SECONDS):
    """Return mono float32 at SR, taken from the middle of the file."""
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        codec, rate = stream.codec_context.name, stream.rate
        if duration > seconds * 2:
            container.seek(int((duration / 2) / stream.time_base), stream=stream)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SR)
        chunks, total = [], 0
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                a = out.to_ndarray().reshape(-1)
                chunks.append(a)
                total += a.size
            if total >= seconds * SR:
                break
    if not chunks:
        return None, codec, rate, duration
    return np.concatenate(chunks)[: seconds * SR], codec, rate, duration


def probe(x):
    n = 1024
    hop = 256
    frames = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    power = spec ** 2
    freqs = np.fft.rfftfreq(n, 1 / SR)

    energy = power.sum(axis=1)
    db = 10 * np.log10(energy + 1e-12)

    # Speech frames: the loud half. Noise floor: the quietest tenth.
    speech = db >= np.percentile(db, 55)
    floor = np.percentile(db, 5)
    level = np.percentile(db, 95)
    snr = level - floor

    # Dryness: the dB spread inside speech. Reverb fills the dips.
    sdb = db[speech]
    dryness = np.percentile(sdb, 95) - np.percentile(sdb, 10)

    # Bandwidth: highest frequency still holding 0.5% of speech energy,
    # measured on the average speech spectrum.
    avg = power[speech].mean(axis=0)
    cum = np.cumsum(avg) / avg.sum()
    rolloff = freqs[np.searchsorted(cum, 0.995)]
    # Energy above 8 kHz, as a share -- lossy YouTube audio often cuts here.
    high = avg[freqs >= 8000].sum() / avg.sum()

    clip = float(np.mean(np.abs(x) > 0.985))
    return dict(snr_db=snr, dryness_db=dryness, rolloff_hz=rolloff,
                share_above_8k=high, clipping=clip)


rows = []
for path in sorted(SRC.glob("*.m4a")):
    vid = path.stem
    try:
        x, codec, rate, duration = decode_middle(path)
        if x is None or len(x) < SR * 5:
            print(f"{vid}: too short", file=sys.stderr)
            continue
        r = probe(x)
        r.update(video=vid, codec=codec, src_rate=rate,
                 minutes=round(duration / 60, 1),
                 channel=MANIFEST.get(vid, {}).get("channel", "?"))
        rows.append(r)
        print(f"  {vid} done", file=sys.stderr)
    except Exception as e:
        print(f"{vid}: {type(e).__name__} {e}", file=sys.stderr)

print(f"\n{'video':<13} {'channel':<28} {'min':>5} {'rate':>6} "
      f"{'SNR dB':>7} {'dry dB':>7} {'rolloff':>8} {'>8kHz':>7} {'clip%':>6}")
print("-" * 100)
for r in sorted(rows, key=lambda r: -r["dryness_db"]):
    print(f"{r['video']:<13} {r['channel'][:27]:<28} {r['minutes']:>5} "
          f"{r['src_rate']:>6} {r['snr_db']:>7.1f} {r['dryness_db']:>7.1f} "
          f"{r['rolloff_hz']:>7.0f}  {r['share_above_8k']*100:>6.2f} "
          f"{r['clipping']*100:>5.2f}")

if rows:
    def col(k):
        return np.array([r[k] for r in rows])
    print("\nmedian  SNR %.1f dB | dryness %.1f dB | rolloff %.0f Hz | >8kHz %.2f%%"
          % (np.median(col("snr_db")), np.median(col("dryness_db")),
             np.median(col("rolloff_hz")), np.median(col("share_above_8k")) * 100))
    print("range   SNR %.0f-%.0f dB | dryness %.1f-%.1f dB | rolloff %.0f-%.0f Hz"
          % (col("snr_db").min(), col("snr_db").max(),
             col("dryness_db").min(), col("dryness_db").max(),
             col("rolloff_hz").min(), col("rolloff_hz").max()))
    Path("/private/tmp/claude-501/-Users-safaksurmeli-Desktop-Lilly/"
         "2287cfc9-0f8a-425b-90fe-f8dcb520b8ee/scratchpad/audio_probe.json"
         ).write_text(json.dumps(rows, indent=2, default=float))
