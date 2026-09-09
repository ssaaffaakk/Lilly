#!/usr/bin/env python3
"""Which parliament speakers train the voice, and which of their segments.

From the metadata scan (data/scripts/parlaspeech_speakers.py), by a rule fixed
in training/PREREGISTRATION.md ("v6 -- speak") before any training:

  * the gender with more clean hours among the ten largest speakers is the
    voice's gender -- the served voice is the MEAN of the trained speakers'
    embeddings, and a mean across genders is nobody's voice at all;
  * the K largest speakers of that gender by clean hours train;
  * each contributes at most H hours of clean segments, taken in dataset
    order (shard, row), which keeps a speaker's sentences in the runs they
    were spoken in and the fetch to as few row groups as possible.

Writes the selection the box fetches (data/scripts/download_parlaspeech_voice.py):
the shard URLs and sizes the scan saw, the speakers by index, and every
segment as [shard, row, id, seconds, speaker]. Committed, so the run and its
write-up name the same audio.

    python3 training/select_parlaspeech_voice.py \\
        --scan data/speech-extra/parlaspeech-hr-speakers.json --out training/speak-parla/selection.json
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

K_SPEAKERS = 5
HOURS_EACH = 3.0
TOP_FOR_GENDER = 10


def choose(scan: dict, k: int = K_SPEAKERS, hours_each: float = HOURS_EACH,
           top_for_gender: int = TOP_FOR_GENDER) -> dict:
    speakers = scan["speakers"]                      # sorted by clean hours, descending
    top = speakers[:top_for_gender]
    by_gender = collections.Counter()
    for s in top:
        by_gender[s["gender"]] += s["hours_clean"]
    if not by_gender:
        raise SystemExit("the scan lists no speakers")
    gender = by_gender.most_common(1)[0][0]
    chosen = [s for s in speakers if s["gender"] == gender][:k]
    if len(chosen) < k:
        raise SystemExit(f"only {len(chosen)} {gender} speakers in the scan; the rule wants {k}")
    segments, names = [], {}
    for idx, s in enumerate(chosen):
        segs = scan["segments"].get(s["name"])
        if not segs:
            raise SystemExit(f"{s['name']} has no listed segments; scan with a larger TOP")
        budget = hours_each * 3600
        taken = 0.0
        for shard, row, seg_id, seconds in sorted(segs, key=lambda x: (x[0], x[1])):
            if taken + seconds > budget:
                break
            segments.append([shard, row, seg_id, seconds, idx])
            taken += seconds
        names[str(idx)] = s["name"]
        s_out = s.copy()
        s_out.update({"index": idx, "hours_taken": round(taken / 3600, 2),
                      "segments_taken": sum(1 for x in segments if x[4] == idx)})
        chosen[idx] = s_out
    segments.sort(key=lambda x: (x[0], x[1]))
    return {"repo": scan["repo"], "rule": {**scan["rule"], "k": k, "hours_each": hours_each,
                                            "gender_from_top": top_for_gender, "gender": gender},
            "shard_urls": scan["shard_urls"], "shard_sizes": scan["shard_sizes"],
            "speakers": names, "chosen": chosen,
            "segments": segments,
            "minutes_total": round(sum(x[3] for x in segments) / 60, 1),
            "shards_touched": len(set(x[0] for x in segments))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=REPO_ROOT / "data" / "speech-extra" / "parlaspeech-hr-speakers.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "training" / "speak-parla" / "selection.json")
    ap.add_argument("--speakers", type=int, default=K_SPEAKERS)
    ap.add_argument("--hours-each", type=float, default=HOURS_EACH)
    args = ap.parse_args()
    scan = json.loads(args.scan.read_text(encoding="utf-8"))
    if "shard_urls" not in scan:
        # An older scan listed no shards; the listing is what the scan read.
        sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))
        from download_extra_speech import list_shards
        shards = list_shards(scan["repo"], "default", "train")
        if len(shards) != scan["shards_scanned"]:
            raise SystemExit(f"the listing has {len(shards)} shards, the scan read {scan['shards_scanned']}")
        scan["shard_urls"] = [e["url"] for e in shards]
        scan["shard_sizes"] = [e["size"] for e in shards]
    sel = choose(scan, args.speakers, args.hours_each)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sel, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"gender {sel['rule']['gender']}; {len(sel['segments'])} segments, {sel['minutes_total']:.0f} minutes, "
          f"{sel['shards_touched']} shards")
    for s in sel["chosen"]:
        print(f"  {s['index']}: {s['name']:<30} clean {s['hours_clean']:.1f} h -> {s['hours_taken']:.2f} h, "
              f"{s['segments_taken']} segments")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
