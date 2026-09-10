#!/usr/bin/env python3
"""Are a channel's excerpts one voice? Resemblyzer embeddings, pairwise cosine.

Reads screen.json from data/scripts/screen_youtube_voice.py, embeds each
excerpt, and prints per channel the mean similarity between its excerpts (one
speaker sits around 0.8 and above; strangers around 0.6 and below) and the
similarity between channels (two channels with the same lecturer show up).

    python3 data/scripts/speaker_consistency.py --screen screen/screen.json
"""
import argparse
import itertools
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    from resemblyzer import VoiceEncoder, preprocess_wav
    rep = json.loads(args.screen.read_text(encoding="utf-8"))
    enc = VoiceEncoder(device="cpu", verbose=False)
    emb = {}
    for vid, r in rep.items():
        if "excerpt" not in r:
            continue
        wav = preprocess_wav(Path(r["excerpt"]))
        e = enc.embed_utterance(wav)
        emb[vid] = e / np.linalg.norm(e)
    by_channel = {}
    for vid, r in rep.items():
        if vid in emb:
            by_channel.setdefault(r["channel"], []).append(vid)
    print(f"{'channel':<34} {'excerpts':>8} {'within':>7}")
    for ch, vids in by_channel.items():
        pairs = [float(emb[a] @ emb[b]) for a, b in itertools.combinations(vids, 2)]
        print(f"{ch:<34} {len(vids):>8} {np.mean(pairs) if pairs else float('nan'):>7.3f}  "
              + " ".join(f"{p:.2f}" for p in pairs))
    chans = list(by_channel)
    print("\nbetween channels (mean over excerpt pairs):")
    for a, b in itertools.combinations(chans, 2):
        sims = [float(emb[x] @ emb[y]) for x in by_channel[a] for y in by_channel[b]]
        if np.mean(sims) > 0.7:
            print(f"  {a} ~ {b}: {np.mean(sims):.3f}  <- same person?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
