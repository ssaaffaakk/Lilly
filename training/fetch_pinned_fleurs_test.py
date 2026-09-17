#!/usr/bin/env python3
"""Fetch and validate the exact FLEURS test parquet used by clean evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != source["manifest_sha256"]:
        raise SystemExit("speech manifest hash differs from frozen source record")
    args.out.mkdir(parents=True, exist_ok=True)
    parquet = args.out / ".test.parquet"
    request = urllib.request.Request(source["parquet_url"], headers={"User-Agent": "Lilly/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, parquet.open("wb") as sink:
        while chunk := response.read(1 << 20):
            sink.write(chunk)
    if parquet.stat().st_size != source["parquet_size"] or file_hash(parquet) != source["parquet_sha256"]:
        raise SystemExit("FLEURS parquet size/hash mismatch; source revision changed or download is partial")

    import pyarrow.parquet as pq
    reader = pq.ParquetFile(parquet)
    names = reader.schema_arrow.names
    audio_col = "audio" if "audio" in names else names[0]
    text_col = next(c for c in ("raw_transcription", "transcription", "text") if c in names)
    clips = args.out / "test"
    clips.mkdir(exist_ok=True)
    tsv = args.out / "test.tsv"
    written = []
    with tsv.open("w", encoding="utf-8") as table:
        for batch in reader.iter_batches(batch_size=64, columns=[audio_col, text_col]):
            for clip, text in zip(batch.column(audio_col).to_pylist(),
                                  batch.column(text_col).to_pylist()):
                raw = clip.get("bytes") if isinstance(clip, dict) else None
                text = (text or "").strip()
                if not raw or not text:
                    continue
                name = f"{len(written):05d}.wav"
                path = clips / name
                path.write_bytes(raw)
                table.write(f"test/{name}\t{text}\n")
                written.append((hashlib.sha256(raw).hexdigest(),
                                hashlib.sha256(text.encode("utf-8")).hexdigest()))
    with args.manifest.open(encoding="utf-8", newline="") as fh:
        expected = {(r["audio_sha256"], r["transcript_sha256"])
                    for r in csv.DictReader(fh, delimiter="\t")}
    if len(written) != source["expected_clips"] or set(written) != expected:
        raise SystemExit(f"unpacked FLEURS identity mismatch: rows={len(written)}, "
                         f"missing={len(expected - set(written))}, extra={len(set(written) - expected)}")
    parquet.unlink()
    print(f"FLEURS pinned revision {source['revision']}: {len(written)} clips verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
