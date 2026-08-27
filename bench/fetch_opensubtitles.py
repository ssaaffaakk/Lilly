#!/usr/bin/env python3
"""Fetch OpenSubtitles bs-en from OPUS, so bench/build.py can count it.

The news and Wikipedia text this project already has contains no Turkisms at
all — *kahva*, *čaršija*, *avlija*, *sokak*, *mahala* are zero across every
professional corpus in the repo, which is why bench/terms-rejected.tsv rejects
them on a count of nothing. Subtitles are the register those words live in, so
this pulls the biggest bs-en subtitle corpus that exists and puts it where
build.py looks.

    python3 bench/fetch_opensubtitles.py

Writes data/raw/OpenSubtitles/OpenSubtitles.{bs,en} — 18,477,297 sentence
pairs, 1.2 GB unpacked, following the data/raw/<CORPUS>/<CORPUS>.<lang> layout
the other corpora already use. Both are gitignored.

Licence: OPUS redistributes subtitle text it does not own, under the terms on
http://opus.nlpl.eu/legacy/OpenSubtitles-v2024.php — free to redistribute, to
be withdrawn on challenge. Work using it has to cite Lison & Tiedemann (2016),
*OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV
Subtitles* (LREC 2016), and link http://www.opensubtitles.org/. The release
carries that request explicitly, so it is repeated here and in build.py.

The version is pinned. OPUS's own API lists v2016, v2018 and v2024 for this
pair; v2024 is the current one and the largest, and a rebuild that silently
changed corpus underneath the counts would make two runs incomparable. To move
to a later release, change VERSION and rebuild — do not let "latest" decide it.
"""
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "data/raw/OpenSubtitles"
VERSION = "v2024"
URL = f"https://object.pouta.csc.fi/OPUS-OpenSubtitles/{VERSION}/moses/bs-en.txt.zip"
# sha256 of the v2024 moses archive, recorded 2026-08-27. A mismatch means the
# release was rebuilt in place; stop rather than count something else.
SHA256 = "0dbc09ab66f9e1337c5bb2c9dd2b18f526e2eb4aaa0e4647849987a763cba51e"
EXPECT_LINES = 18477297


def download(zip_path: Path) -> None:
    if zip_path.exists():
        print(f"  archive already here: {zip_path} ({zip_path.stat().st_size:,} bytes)")
        return
    print(f"  downloading {URL}")
    tmp = zip_path.with_suffix(".part")
    with urllib.request.urlopen(URL) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("content-length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(zip_path)


def verify(zip_path: Path) -> None:
    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    got = h.hexdigest()
    if got != SHA256:
        raise SystemExit(
            f"checksum mismatch for {zip_path}\n"
            f"  expected {SHA256}\n  got      {got}\n"
            f"OPUS rebuilt {VERSION} in place. Check the release, then update SHA256 "
            f"deliberately — counts taken from a different corpus are not comparable "
            f"with the ones already in bench/terms.tsv.")
    print(f"  sha256 ok: {got}")


def extract(zip_path: Path) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member, out_name in (("OpenSubtitles.bs-en.bs", "OpenSubtitles.bs"),
                                 ("OpenSubtitles.bs-en.en", "OpenSubtitles.en")):
            out = DEST / out_name
            print(f"  unpacking {member} -> {out}")
            with z.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
        for member in ("README", "LICENSE"):
            with z.open(member) as src, open(DEST / member, "wb") as dst:
                shutil.copyfileobj(src, dst)


def check(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            n += chunk.count(b"\n")
    return n


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    zip_path = DEST / f"bs-en-{VERSION}.txt.zip"
    download(zip_path)
    verify(zip_path)
    extract(zip_path)
    counts = {p.name: check(p) for p in (DEST / "OpenSubtitles.bs", DEST / "OpenSubtitles.en")}
    for name, n in counts.items():
        print(f"  {name}: {n:,} lines")
    if len(set(counts.values())) != 1:
        raise SystemExit(f"the two sides disagree on length: {counts}")
    if next(iter(counts.values())) != EXPECT_LINES:
        raise SystemExit(f"expected {EXPECT_LINES:,} lines, got {counts}")
    print(f"\nOK. Now: python3 bench/build.py")
    print("If you publish anything counted from this, cite Lison & Tiedemann (2016)")
    print("and link http://www.opensubtitles.org/ — OPUS asks for both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
