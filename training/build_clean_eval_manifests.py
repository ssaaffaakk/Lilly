#!/usr/bin/env python3
"""Build frozen, prediction-independent Listen and Read eval manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training" / "clean-eval"
SPEECH_TSV = ROOT / "data" / "speech" / "test.tsv"
OCR_SOURCES = ROOT / "data" / "ocr" / "real-photos" / "scored-sources.tsv"
OCR_TRUTH = ROOT / "data" / "ocr" / "real-photos" / "truth.json"

FLEURS = {
    "dataset": "google/fleurs", "config": "bs_ba", "split": "test",
    "revision": "168de341b3db6859a9bac1c50a2ef5e3b47647e0",
    "parquet_url": ("https://huggingface.co/datasets/google/fleurs/resolve/"
                    "168de341b3db6859a9bac1c50a2ef5e3b47647e0/bs_ba/test/0000.parquet"),
    "parquet_size": 714890422,
    "parquet_sha256": "7c587762fb5bbd0baa3ca527d1cac9b5fdddec67e34d572f2c77081f97e7b5dc",
    "expected_clips": 925,
}

# Frozen from visual inspection of the originals, before fresh inference.
OCR_NOISY = {
    "Jewish_Street_Tuzla_Bosnia.jpg": "shadowed and small street signs, still readable",
    "Mostar_signs.JPG": "motion blur and haze, text still readable",
    "Putokaz2.jpg": "distant direction signs, still readable",
    "Road_to_Baljevac_-_panoramio.jpg": "dim and distant sign, still readable",
    "Sarajevo_Trebević_Sign.jpg": "graffiti partly obstructs the Latin line",
    "WV_banner_NE_Bosnia_Tuzla_old_town.jpg": "very small panorama labels, still readable",
}
OCR_UNINTELLIGIBLE = {
    "Mis_Irbina_Street_in_Sarajevo_03.jpg":
        "distant billboard fragments; full reference is not reliably human-readable",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_speech() -> None:
    import soundfile as sf
    rows = []
    for line in SPEECH_TSV.read_text(encoding="utf-8").splitlines():
        clip, text = line.split("\t", 1)
        wav = SPEECH_TSV.parent / clip
        info = sf.info(str(wav))
        rows.append({
            "audio_sha256": sha256(wav.read_bytes()),
            "transcript_sha256": sha256(text.encode("utf-8")),
            "cohort": "clean",
            "review_basis": ("official FLEURS held-out Bosnian read-speech with human "
                             "transcript; source and waveform integrity accepted before decode"),
            "duration_ms": round(1000 * info.frames / info.samplerate),
            "sample_rate": info.samplerate,
        })
    identities = {(r["audio_sha256"], r["transcript_sha256"]) for r in rows}
    if len(rows) != 925 or len(identities) != 925:
        raise SystemExit(f"speech manifest refused: rows={len(rows)}, unique={len(identities)}")
    path = OUT / "speech-fleurs-bs-test.tsv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: r["audio_sha256"]))
    source = dict(FLEURS)
    source.update({
        "manifest": path.name, "manifest_sha256": sha256(path.read_bytes()),
        "cohort_counts": {"clean": 925, "noisy_but_understandable": 0,
                          "human_unintelligible": 0},
        "classification_note": ("No cohort was inferred from SNR or model output. This is "
                                "the curated FLEURS held-out read-speech split; it has no "
                                "deliberately noisy or unintelligible cohort. Zero counts are "
                                "reported rather than fabricating corruptions."),
    })
    (OUT / "speech-source.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def commons_title(page_url: str) -> str:
    return urllib.parse.unquote(page_url.split("/wiki/", 1)[1]).replace("_", " ")


def commons_infos(titles: list[str]) -> dict[str, dict]:
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": 2,
        "prop": "info|imageinfo", "iiprop": "url|sha1|size|timestamp",
        "titles": "|".join(titles),
    })
    request = urllib.request.Request(
        "https://commons.wikimedia.org/w/api.php?" + query,
        headers={"User-Agent": "Lilly-clean-eval/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        query_data = json.load(response)["query"]
    aliases = {row["from"]: row["to"] for row in query_data.get("normalized", [])}
    pages = {page["title"]: page for page in query_data["pages"]}
    result = {}
    for title in titles:
        canonical = aliases.get(title, title)
        page = pages.get(canonical)
        if not page or "missing" in page or not page.get("imageinfo"):
            raise SystemExit(f"Commons source missing: {title} (canonical {canonical})")
        info = page["imageinfo"][0]
        result[title] = {
            "commons_title": page["title"], "commons_lastrevid": page["lastrevid"],
            "original_url": info["url"], "commons_sha1": info["sha1"],
            "bytes": info["size"], "width": info["width"], "height": info["height"],
            "commons_timestamp": info["timestamp"],
        }
    return result


def build_ocr() -> None:
    truth = json.loads(OCR_TRUTH.read_text(encoding="utf-8"))["photos"]
    with OCR_SOURCES.open(encoding="utf-8", newline="") as fh:
        sources = list(csv.DictReader(fh, delimiter="\t"))
    if len(sources) != 40 or set(truth) != {r["file"] for r in sources}:
        raise SystemExit("OCR source/truth set is not exactly the same 40 photographs")
    titles = [commons_title(source["page_url"]) for source in sources]
    source_info = commons_infos(titles)
    rows = []
    for source in sources:
        name = source["file"]
        has_text = bool(truth[name]["lines"])
        if not has_text:
            cohort, reason = "no_text", "answer key contains no human-legible text"
        elif name in OCR_UNINTELLIGIBLE:
            cohort, reason = "human_unintelligible", OCR_UNINTELLIGIBLE[name]
        elif name in OCR_NOISY:
            cohort, reason = "noisy_but_understandable", OCR_NOISY[name]
        else:
            cohort, reason = "clean", "sharp and human-readable at the original resolution"
        row = {"file": name, "cohort": cohort, "has_text": str(has_text).lower(),
               "review_basis": reason, "license": source["license"],
               "attribution": source["attribution"], "page_url": source["page_url"]}
        row.update(source_info[commons_title(source["page_url"])])
        rows.append(row)
    counts = {c: sum(r["cohort"] == c for r in rows)
              for c in ("clean", "noisy_but_understandable", "human_unintelligible", "no_text")}
    expected = {"clean": 21, "noisy_but_understandable": 6,
                "human_unintelligible": 1, "no_text": 12}
    if counts != expected:
        raise SystemExit(f"unexpected OCR cohort counts: {counts}")
    path = OUT / "ocr-commons-40.tsv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: r["file"]))
    (OUT / "ocr-source.json").write_text(json.dumps({
        "schema": 1, "provider": "Wikimedia Commons API", "expected_photos": 40,
        "expected_text_photos": 28, "cohort_counts": counts,
        "manifest": path.name, "manifest_sha256": sha256(path.read_bytes()),
        "classification_note": ("Cohorts were frozen by viewing original photographs "
                                "before fresh PaddleOCR inference. Image heuristics and "
                                "model predictions were not used."),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    build_speech(); build_ocr()
    for path in sorted(OUT.iterdir()):
        print(path.relative_to(ROOT), sha256(path.read_bytes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
