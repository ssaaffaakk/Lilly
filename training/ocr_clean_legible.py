#!/usr/bin/env python3
"""Reproduce RESULTS-ocr-clean-legible.md — the reader on clean legible signage.

A re-slice of the committed 40-Commons answer key, not a new run. It needs no
model and no GPU: the reader's text is already in reader-output-paddle-v6.json
and the floor-0.9 counts are in training/paddle-floor/the40-floor0.9.json.

The point it makes: on clear, legible, MEANINGFUL Bosnian/English signage the
reader reads 89% of the Latin words and 0% of the Cyrillic ones. The all-40
figure of 67% mixes those clean signs with brand logos and distant street
fragments; this file scores the clean signs on their own, and splits by script
so the Cyrillic blindness is not hidden inside the average.

    python3 training/ocr_clean_legible.py

Exit 1 if the published 89.0 / 0.0 split stops reproducing.
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRUTH = REPO / "data/ocr/real-photos/truth.json"
READ = REPO / "data/ocr/real-photos/reader-output-paddle-v6.json"
FLOOR = REPO / "training/paddle-floor/the40-floor0.9.json"

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)  # letters only, matching evaluate_ocr.words

# Clear, close, legible signs/plaques with meaningful Bosnian/English words.
# Classified by legibility before any score was looked at; the reasons are the
# text each photograph actually carries. Toponyms and memorial text count as
# meaningful; brand logos and distant incidental shop signage do not.
KEEP = [
    "Assasination_Plaque.JPG",
    "Direction_sign_to_Old_city_of_Kljuc.jpg",
    "Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg",
    "Sarajevo_road_M-a8_IMG_1166.JPG",
    "Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg",
    "War_Memorial_in_Kučine_BiH_2024.jpg",
    "Road_to_Baljevac_-_panoramio.jpg",
    "Putokaz_za_manastir_Krupu.jpg",
    "Banjaluka_streetmap.jpg",
    "Putokaz2.jpg",
    "Sarajevo_Trebević_Sign.jpg",
    "Editing_Wikipedia_Workshop_in_Visegrad_-_76.JPG",
    "Mostar_signs.JPG",
    "Trg-žrtava-ŠB03078.JPG",
]

PUBLISHED = {"latin": (113, 127, 89.0), "cyrillic": (0, 25, 0.0)}


def words(text):
    return [w.lower() for w in WORD.findall(unicodedata.normalize("NFC", text))]


def is_cyrillic(word):
    return any("Ѐ" <= ch <= "ӿ" for ch in word)


def recall(truth_words, found_words):
    want, got = Counter(truth_words), Counter(found_words)
    return sum((want & got).values()), sum(want.values())


def main():
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))["photos"]
    readings = json.loads(READ.read_text(encoding="utf-8"))["readings"]
    floor = json.loads(FLOOR.read_text(encoding="utf-8"))["per_photograph"]

    lat = [0, 0]
    cyr = [0, 0]
    combined_floor = [0, 0]
    print("clean legible meaningful signage — per photograph (floor 0.9)")
    for name in sorted(KEEP, key=lambda n: -floor[n][0] / max(floor[n][1], 1)):
        tw = words(" ".join(truth[name]["lines"]))
        fw = words(readings.get(name, ""))
        tl = [w for w in tw if not is_cyrillic(w)]
        tc = [w for w in tw if is_cyrillic(w)]
        h, t = recall(tl, fw)
        lat[0] += h
        lat[1] += t
        h2, t2 = recall(tc, fw)
        cyr[0] += h2
        cyr[1] += t2
        combined_floor[0] += floor[name][0]
        combined_floor[1] += floor[name][1]
        rate = 100 * floor[name][0] / floor[name][1]
        print(f"  {floor[name][0]:>3}/{floor[name][1]:<3} {rate:5.1f}%  {name}")

    failures = []
    print("\nby script, clean subset:")
    for key, (h, t) in (("latin", lat), ("cyrillic", cyr)):
        pct = 100 * h / t if t else 0.0
        ph, pt, ppct = PUBLISHED[key]
        ok = (h, t) == (ph, pt) and abs(pct - ppct) < 0.05
        print(f"  {key:9} {h:>3}/{t:<3} = {pct:5.1f}%   published {ph}/{pt} = {ppct:.1f}"
              f"   {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{key}: published {ph}/{pt}, recomputed {h}/{t}")

    # The floor must drop no correct word on this clean subset, or the pre-floor
    # split above would not describe the shipped product.
    pre = lat[0] + cyr[0]
    if (combined_floor[0], combined_floor[1]) != (pre, lat[1] + cyr[1]):
        failures.append(f"floor changed the clean-subset count: floor "
                        f"{combined_floor[0]}/{combined_floor[1]} vs pre-floor "
                        f"{pre}/{lat[1] + cyr[1]}")
    else:
        print(f"\n  floor 0.9 drops zero correct words here: {combined_floor[0]}/"
              f"{combined_floor[1]} both ways")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 1
    print("\nclean-legible split reproduces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
