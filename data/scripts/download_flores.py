#!/usr/bin/env python3
"""Fetch a Bosnian test set the base model has never seen.

Our own test split came from the same OPUS releases the base model was trained
on, so scoring on it flatters the base and cannot answer "is Lilly better at
Bosnian". FLORES-200 can: it was built separately, by human translators, and
none of it appears in the base model's training manifest.

    python3 data/scripts/download_flores.py

Writes data/flores/{devtest,dev}.{bs,en} — 2,009 sentence pairs in total, one
per line, line N of one matching line N of the other. training/evaluate.py picks
them up automatically and scores both halves together.
"""
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
OUT = Path(__file__).resolve().parents[1] / "flores"
# Both halves, not just devtest. FLORES ships a dev split of 997 sentences
# alongside the 1,012 of devtest, built the same way and equally unseen. Scoring
# on both nearly doubles the sample, which narrows the confidence interval by
# about a third — the difference between a result that is unproven and one that
# is settled either way.
WANTED = {"devtest/bos_Latn.devtest": "devtest.bs",
          "devtest/eng_Latn.devtest": "devtest.en",
          "dev/bos_Latn.dev": "dev.bs",
          "dev/eng_Latn.dev": "dev.en"}


def main() -> int:
    if all((OUT / name).exists() for name in WANTED.values()):
        print(f"already here: {OUT}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "lilly-translator"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    print(f"got {len(blob) / 1048576:.0f} MB, unpacking…", flush=True)

    found = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            for suffix, name in WANTED.items():
                if member.name.endswith(suffix):
                    data = tar.extractfile(member).read()
                    (OUT / name).write_bytes(data)
                    lines = data.decode("utf-8").count("\n")
                    print(f"  {name}: {lines:,} sentences")
                    found += 1
    if found != len(WANTED):
        print(f"expected {len(WANTED)} files, found {found}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
