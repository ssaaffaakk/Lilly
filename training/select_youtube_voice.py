#!/usr/bin/env python3
"""Which Creative-Commons YouTube speakers train the voice, and which videos.

From the Creative-Commons search the Mac ran (a JSON of videos with channel,
title and duration), by a rule fixed in training/PREREGISTRATION.md
("v8 -- speak") before any training:

  * a speaker is a person NAMED in the video titles -- the lecturer -- merged
    across the spellings the channels use; a 90-second screening showed
    resemblyzer separates two lecturers in the same hall only narrowly, so
    the name, not a cluster, is the identity, and the box removes clips that
    stray from the speaker's own centroid;
  * one channel's unnamed single reader (Islam4Peace.com's translation
    readings, two videos at 0.97 similarity) counts as a named speaker;
  * the K largest speakers by CC minutes, each with at least MIN_HOURS;
  * per speaker, videos longest first until RAW_HOURS of video; the listener
    then keeps what passes the clip rule on the box.

Writes training/speak-youtube/selection.json: {"speakers": {name: [video, ...]}}
with every video's id, channel, title and minutes. The downloader reads the
license from each video again before fetching a byte.

    python3 training/select_youtube_voice.py --search scratch/yt/search.json --out training/speak-youtube/selection.json
"""
import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
K_SPEAKERS = 7
MIN_HOURS = 3.0
RAW_HOURS = 3.0
CHANNELS = ("Švedska Dawetska Organizacija", "Nauči islam", "Islamska Predavanja", "tawus7691", "dzindoadis",
            "Tevhid Ocagi Sarajevo", "Islam4Peace.com")
# name -> the spellings that name them in titles (case-insensitive, diacritics as written)
SPEAKERS = {
    "Safet Kuduzović": ("kuduzović", "kuduzovic"),
    "Hajrudin Ahmetović": ("ahmetović", "ahmetovic"),
    "Elvedin Pezić": ("pezić", "pezic"),
    "Dževad Gološ": ("gološ", "golos"),
    "Zuhdija Adilović": ("adilović", "adilovic"),
    "Zijad Ljakić": ("ljakić", "ljakic"),
    "Selmir Hadžić": ("hadžić", "hadzic"),
    "Adnan Mrkonjić": ("mrkonjić", "mrkonjic"),
    "Islam4Peace.com reader": ("quran bosanski bosnian translation",),
}


def choose(search: dict, k: int = K_SPEAKERS, min_hours: float = MIN_HOURS, raw_hours: float = RAW_HOURS) -> dict:
    per = {name: [] for name in SPEAKERS}
    for v in search.values():
        if v["channel"] not in CHANNELS or not v.get("duration"):
            continue
        title = (v["title"] or "").lower()
        hits = [name for name, keys in SPEAKERS.items() if any(key in title for key in keys)]
        if len(hits) != 1:      # two lecturers in one title, or none: not one person's video
            continue
        per[hits[0]].append({"id": v["id"], "channel": v["channel"], "title": v["title"],
                             "minutes": round(v["duration"] / 60, 1)})
    ranked = sorted(per.items(), key=lambda kv: -sum(x["minutes"] for x in kv[1]))
    chosen = {}
    for name, vids in ranked:
        total = sum(x["minutes"] for x in vids) / 60
        if total < min_hours or len(chosen) >= k:
            continue
        taken, picked = 0.0, []
        for x in sorted(vids, key=lambda x: -x["minutes"]):
            if taken + x["minutes"] / 60 > raw_hours + 0.5:
                continue
            picked.append(x)
            taken += x["minutes"] / 60
            if taken >= raw_hours:
                break
        chosen[name] = {"cc_hours_found": round(total, 1), "raw_hours_taken": round(taken, 2), "videos": picked}
    if len(chosen) < 5:
        raise SystemExit(f"only {len(chosen)} speakers with {min_hours} h; the rule wants at least 5")
    return {"rule": {"k": k, "min_hours": min_hours, "raw_hours": raw_hours, "channels": list(CHANNELS)},
            "speakers": chosen,
            "raw_hours_total": round(sum(s["raw_hours_taken"] for s in chosen.values()), 1),
            "videos_total": sum(len(s["videos"]) for s in chosen.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "training" / "speak-youtube" / "selection.json")
    args = ap.parse_args()
    sel = choose(json.loads(args.search.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sel, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(sel['speakers'])} speakers, {sel['videos_total']} videos, {sel['raw_hours_total']} h of video")
    for name, s in sel["speakers"].items():
        print(f"  {name:<26} found {s['cc_hours_found']:>5.1f} h CC -> {s['raw_hours_taken']:.2f} h in {len(s['videos'])} videos "
              f"({', '.join(sorted(set(v['channel'] for v in s['videos'])))[:70]})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
