#!/usr/bin/env python3
"""One shard of the parallel crop pipeline.

    .venv/bin/python3 data/scripts/crop_shard.py --shard 0 --total 3
    .venv/bin/python3 data/scripts/crop_shard.py --shard 1 --total 3
    .venv/bin/python3 data/scripts/crop_shard.py --shard 2 --total 3

Each shard writes to data/ocr/crops-mapillary/shardN/.
Run merge_shards.py afterwards to combine into one labels.tsv.
"""
import argparse, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PHOTO_DIR = REPO_ROOT / "data" / "ocr" / "real-photos" / "mapillary"
CROPS_BASE = REPO_ROOT / "data" / "ocr" / "crops-mapillary"
MIN_CONF = 0.4
MIN_PX = 8


def crop_one(photo: Path, out_dir: Path):
    from PIL import Image
    from app.ocr import read_regions
    try:
        image = Image.open(photo).convert("RGB")
        regions = read_regions(str(photo))
    except Exception:
        return []
    rows = []
    for i, (box, text, conf) in enumerate(regions):
        text = text.strip()
        if not text or conf < MIN_CONF:
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x0, y0 = max(min(xs), 0), max(min(ys), 0)
        x1, y1 = max(xs), max(ys)
        crop = image.crop((x0, y0, x1, y1))
        if crop.width < MIN_PX or crop.height < MIN_PX:
            continue
        name = f"{photo.stem}_{i:03d}.png"
        crop.save(out_dir / name)
        rows.append((name, text, conf))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--total", type=int, default=3)
    args = ap.parse_args()

    tag = f"[shard{args.shard}]"
    out_dir = CROPS_BASE / f"shard{args.shard}"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "processed.txt"
    labels_file = out_dir / "labels.tsv"

    processed = set()
    if state_file.exists():
        processed = set(state_file.read_text(encoding="utf-8").splitlines())

    all_photos = sorted(PHOTO_DIR.glob("mly_*.jpg"))
    my_photos = [p for i, p in enumerate(all_photos) if i % args.total == args.shard]
    todo = [p for p in my_photos if p.name not in processed]

    print(f"{tag} {len(my_photos)} total, {len(todo)} to process", flush=True)
    print(f"{tag} Loading EasyOCR...", flush=True)
    from app.ocr import read_regions  # noqa — triggers model load
    print(f"{tag} Ready", flush=True)

    total_crops = 0
    fresh = not labels_file.exists()
    lf = open(labels_file, "a", encoding="utf-8")
    if fresh:
        lf.write("file\ttext\tconfidence\n")

    t0 = time.time()
    for i, photo in enumerate(todo):
        rows = crop_one(photo, out_dir)
        for name, text, conf in rows:
            lf.write(f"{name}\t{text}\t{conf:.2f}\n")
        total_crops += len(rows)
        processed.add(photo.name)
        lf.flush()
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate / 60
            print(f"{tag} {i+1}/{len(todo)} | crops:{total_crops} | "
                  f"{rate:.1f} img/s | ETA {eta:.0f}m", flush=True)
            state_file.write_text("\n".join(sorted(processed)), encoding="utf-8")

    lf.close()
    state_file.write_text("\n".join(sorted(processed)), encoding="utf-8")
    print(f"{tag} DONE — {total_crops} crops from {len(todo)} photos", flush=True)


if __name__ == "__main__":
    main()
