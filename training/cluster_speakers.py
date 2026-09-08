#!/usr/bin/env python3
"""Who is speaking in FLEURS bs_ba: speaker clusters for the voice training.

FLEURS carries no speaker ids, and a voice trained on ten hours of unlabelled
strangers is one voice averaged over all of them -- breathy, and nobody's. So
every clip is embedded once (resemblyzer's GE2E encoder, 256 numbers per
clip) and the embeddings are clustered by cosine distance; each cluster is
treated as one speaker. The clusters that hold enough minutes train the
multi-speaker voice; the rest of the clips are left out.

Two things are checked, and the second can stop the run:

  * cohesion -- the mean cosine similarity inside each kept cluster, printed
    so a cluster that is really several people is visible;
  * collisions -- FLEURS records most sentences with two or three different
    readers, so two clips of the SAME sentence should land in DIFFERENT
    clusters. The share of same-sentence pairs that share a cluster is the
    clustering's own error rate, measured from the data's structure with no
    listening. Above --max-collision the clusters are not speakers and the
    run stops rather than train a voice on a label that means nothing.

The threshold, the minimum minutes and the speaker cap are fixed by
training/PREREGISTRATION.md ("v5 -- speak") before any training; the
defaults here are those values. Everything is written to --out/speakers.json,
the embeddings beside it (embeddings.npy, clip_order.json) so a re-run at a
different threshold costs seconds.

    python3 training/cluster_speakers.py --tsv data/speech/train.tsv --out data/speech/speakers
    python3 training/cluster_speakers.py --tsv data/speech/train.tsv --out ... --threshold 0.25 --dry-run
"""
import argparse
import collections
import itertools
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Pre-registered (PREREGISTRATION.md, "v5 -- speak"); change there first.
THRESHOLD = 0.30       # cosine distance; 1 - 0.70 similarity, resemblyzer's own guidance
MIN_MINUTES = 20.0     # a cluster with less than this does not train
MAX_SPEAKERS = 8       # the largest clusters, by minutes
MAX_COLLISION = 0.50   # above this share of same-sentence pairs in one cluster, stop


def read_tsv(path: Path) -> list:
    """(clip path relative to the TSV's directory, text), in file order."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        clip, text = line.split("\t", 1)
        rows.append((clip, text.strip()))
    return rows


def embed(rows: list, audio_root: Path, cache: Path, device: str) -> "np.ndarray":
    import numpy as np
    order_path = cache / "clip_order.json"
    emb_path = cache / "embeddings.npy"
    if emb_path.is_file() and order_path.is_file():
        order = json.loads(order_path.read_text(encoding="utf-8"))
        if order == [clip for clip, _ in rows]:
            print(f"embeddings cached: {emb_path}")
            return np.load(emb_path)
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder(device=device, verbose=False)
    out = np.zeros((len(rows), 256), dtype=np.float32)
    for i, (clip, _) in enumerate(rows):
        wav = preprocess_wav(audio_root / clip)
        if len(wav) < 1600:
            raise SystemExit(f"{clip}: under 0.1 s of speech after the VAD -- not a clip to cluster")
        out[i] = encoder.embed_utterance(wav)
        if (i + 1) % 250 == 0:
            print(f"  embedded {i + 1}/{len(rows)}", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, out)
    order_path.write_text(json.dumps([clip for clip, _ in rows]), encoding="utf-8")
    return out


def cluster(embeddings, threshold: float) -> list:
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                    metric="cosine", linkage="average")
    return [int(x) for x in model.fit_predict(embeddings)]


def seconds_of(path: Path) -> float:
    import soundfile as sf
    return float(sf.info(str(path)).duration)


def cohesion(embeddings, members: list, rng: random.Random, sample: int = 400) -> float:
    """Mean cosine similarity over (a sample of) the pairs inside one cluster."""
    import numpy as np
    if len(members) < 2:
        return 1.0
    pairs = list(itertools.combinations(members, 2))
    if len(pairs) > sample:
        pairs = rng.sample(pairs, sample)
    e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return float(np.mean([float(e[a] @ e[b]) for a, b in pairs]))


def collisions(rows: list, labels: list) -> tuple:
    """Same-sentence clip pairs, and how many of them fell into one cluster."""
    by_text = collections.defaultdict(list)
    for i, (_, text) in enumerate(rows):
        by_text[text].append(i)
    pairs = same = 0
    for idxs in by_text.values():
        for a, b in itertools.combinations(idxs, 2):
            pairs += 1
            same += labels[a] == labels[b]
    return pairs, same


def build(rows: list, labels: list, durations: list, embeddings, args) -> dict:
    rng = random.Random(7)
    members = collections.defaultdict(list)
    for i, lab in enumerate(labels):
        members[lab].append(i)
    speakers = []
    for lab, idxs in members.items():
        secs = sum(durations[i] for i in idxs)
        speakers.append({"id": lab, "clips": len(idxs), "seconds": round(secs, 1),
                         "minutes": round(secs / 60, 1),
                         "cohesion": round(cohesion(embeddings, idxs, rng), 3)})
    speakers.sort(key=lambda s: -s["seconds"])
    pairs, same = collisions(rows, labels)
    rate = same / pairs if pairs else 0.0
    eligible = [s for s in speakers if s["minutes"] >= args.min_minutes]
    selected = [s["id"] for s in eligible[: args.max_speakers]]
    return {
        "threshold": args.threshold, "min_minutes": args.min_minutes,
        "max_speakers": args.max_speakers, "max_collision": args.max_collision,
        "n_clips": len(rows), "n_clusters": len(speakers),
        "same_sentence_pairs": pairs, "same_sentence_same_cluster": same,
        "collision_rate": round(rate, 4),
        "selected": selected,
        "selected_minutes": round(sum(s["minutes"] for s in eligible[: args.max_speakers]), 1),
        "speakers": speakers,
        "clips": {clip: {"speaker": labels[i], "seconds": round(durations[i], 2)}
                  for i, (clip, _) in enumerate(rows)},
    }


def report(rec: dict) -> None:
    print(f"\n{rec['n_clips']} clips -> {rec['n_clusters']} clusters at cosine distance "
          f"{rec['threshold']}; same-sentence pairs in one cluster: "
          f"{rec['same_sentence_same_cluster']} of {rec['same_sentence_pairs']} "
          f"({100 * rec['collision_rate']:.1f}%)")
    print(f"{'speaker':>8} {'clips':>6} {'minutes':>8} {'cohesion':>9}  trains")
    for s in rec["speakers"][:20]:
        mark = "yes" if s["id"] in rec["selected"] else ""
        print(f"{s['id']:>8} {s['clips']:>6} {s['minutes']:>8.1f} {s['cohesion']:>9.3f}  {mark}")
    if rec["n_clusters"] > 20:
        rest = rec["speakers"][20:]
        print(f"     ... {len(rest)} smaller clusters, {sum(s['minutes'] for s in rest):.1f} minutes together")
    print(f"selected: {rec['selected']} -- {rec['selected_minutes']} minutes train")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=REPO_ROOT / "data" / "speech" / "train.tsv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "speech" / "speakers")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--min-minutes", type=float, default=MIN_MINUTES)
    ap.add_argument("--max-speakers", type=int, default=MAX_SPEAKERS)
    ap.add_argument("--max-collision", type=float, default=MAX_COLLISION)
    ap.add_argument("--device", default="cpu", help="cpu, or cuda on a box that has one")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (a smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the table for this threshold; write nothing")
    args = ap.parse_args()

    rows = read_tsv(args.tsv)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {args.tsv}")
    audio_root = args.tsv.parent
    missing = [clip for clip, _ in rows if not (audio_root / clip).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} clips named in {args.tsv} are not on disk, "
                         f"first: {missing[0]} -- not clustering a partial split")

    cache = args.out if not args.limit else args.out / f"limit-{args.limit}"
    embeddings = embed(rows, audio_root, cache, args.device)
    durations = [seconds_of(audio_root / clip) for clip, _ in rows]
    labels = cluster(embeddings, args.threshold)
    rec = build(rows, labels, durations, embeddings, args)
    report(rec)

    if rec["collision_rate"] > args.max_collision:
        raise SystemExit(
            f"{100 * rec['collision_rate']:.1f}% of same-sentence pairs share a cluster; "
            f"above {100 * args.max_collision:.0f}% these clusters are not speakers. "
            f"Not training a voice on labels that mean nothing.")
    if not rec["selected"]:
        raise SystemExit(f"no cluster holds {args.min_minutes} minutes; nothing to train")
    if args.dry_run:
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "speakers.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    print(f"wrote {args.out / 'speakers.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
