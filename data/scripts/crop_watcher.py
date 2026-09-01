#!/usr/bin/env python3
"""Continuously crop text regions from new Mapillary photos and write labels.

Watches data/ocr/real-photos/mapillary/ for new .jpg files.
For each new photo: runs EasyOCR detection, crops each text region,
appends to data/ocr/crops-mapillary/labels.tsv.

    python3 data/scripts/crop_watcher.py

Runs until killed. Safe to restart — already-processed photos are skipped.
After enough crops accumulate, run:
    python3 training/prepare_ocr_data.py --labels data/ocr/crops-mapillary/labels.tsv
to build train/valid splits.
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PHOTO_DIR = REPO_ROOT / "data" / "ocr" / "real-photos" / "mapillary"
CROPS_DIR = REPO_ROOT / "data" / "ocr" / "crops-mapillary"
LABELS_FILE = CROPS_DIR / "labels.tsv"
STATE_FILE = CROPS_DIR / "processed.txt"
MIN_CONFIDENCE = 0.4
MIN_CROP_PX = 8
POLL_SECONDS = 30


def load_processed() -> set:
    if not STATE_FILE.exists():
        return set()
    return set(STATE_FILE.read_text(encoding="utf-8").splitlines())


def save_processed(done: set) -> None:
    STATE_FILE.write_text("\n".join(sorted(done)), encoding="utf-8")


def crop_photo(photo: Path, out_dir: Path) -> list:
    """Run EasyOCR on one photo, return list of (crop_name, text, confidence)."""
    from PIL import Image
    from app.ocr import read_regions

    try:
        image = Image.open(photo).convert("RGB")
        regions = read_regions(str(photo))
    except Exception as exc:
        print(f"  ERROR {photo.name}: {exc}", flush=True)
        return []

    rows = []
    for i, (box, text, confidence) in enumerate(regions):
        text = text.strip()
        if not text or confidence < MIN_CONFIDENCE:
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x0, y0 = max(min(xs), 0), max(min(ys), 0)
        x1, y1 = max(xs), max(ys)
        crop = image.crop((x0, y0, x1, y1))
        if crop.width < MIN_CROP_PX or crop.height < MIN_CROP_PX:
            continue
        name = f"{photo.stem}_{i:03d}.png"
        crop.save(out_dir / name)
        rows.append((name, text, confidence))
    return rows


def append_labels(rows: list) -> None:
    fresh = not LABELS_FILE.exists()
    with open(LABELS_FILE, "a", encoding="utf-8") as f:
        if fresh:
            f.write("file\ttext\tconfidence\n")
        for name, text, conf in rows:
            f.write(f"{name}\t{text}\t{conf:.2f}\n")


def main():
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading EasyOCR reader...", flush=True)
    # Warm up — triggers model load once
    from app.ocr import read_regions  # noqa: F401
    print("Reader ready. Watching for new photos...", flush=True)

    processed = load_processed()
    total_crops = 0

    while True:
        photos = sorted(PHOTO_DIR.glob("mly_*.jpg"))
        new = [p for p in photos if p.name not in processed]

        if new:
            print(f"\n{len(new)} new photos to crop ({len(photos)} total seen)",
                  flush=True)
            for photo in new:
                rows = crop_photo(photo, CROPS_DIR)
                if rows:
                    append_labels(rows)
                    total_crops += len(rows)
                processed.add(photo.name)
                print(f"  {photo.name}: {len(rows)} crops "
                      f"(total {total_crops:,})", flush=True)

            save_processed(processed)
            print(f"Done batch. {total_crops:,} crops in {LABELS_FILE}",
                  flush=True)
        else:
            harvest_running = any(True for _ in
                                  Path("/proc").glob("*/cmdline")
                                  if False) or True  # just poll
            print(f"  [{time.strftime('%H:%M')}] {len(processed)} processed, "
                  f"{total_crops:,} crops. Waiting...", flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
