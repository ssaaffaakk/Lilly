# Cyrillic rescue under PP-OCRv6 — the pre-registered runs

Rule, bars and order: `training/PREREGISTRATION.md`, "Cyrillic rescue under
PP-OCRv6" (written 5 September 2026 before any run). Code: `app/ocr.py`,
`LILLY_PADDLE_CYRILLIC_RESCUE=1`. Report: `training/rescue_report.py`. Raw
scorer output and the rescue logs: `training/paddle-rescue/`.

The 40 are a **smoke test only** (26 Cyrillic words on 7 photographs decide
nothing); test-v2 is the one look that decides. Both sections below are the
script's output, unedited.

**What the 40 showed, 5 September 2026.** The mechanical check holds: rescue
off equals shipped on every photograph. The Cyrillic recogniser was asked
about 45 dropped boxes and kept 7; two Cyrillic lines on `Putokaz2.jpg`
(Аутобуска станица, read as "Дутобуска"; Железничка станица, right) and four
Latin fragments on the Spanish-square memorial ("04-NOV-94", "SGTO", "D",
"StAll"), plus one "27". Six key words gained, two invented words added. So
the second recogniser also rescues Latin boxes the first one was unsure of,
which the rule did not anticipate and does not forbid.

## the40 — Cyrillic rescue against the shipped configuration

Shipped arm `aea5890abfdb7ba3`; rescue arm `f7d79d3d3858129d`.

**Rescue-off equals shipped, mechanically:** every shipped word is in the rescue reading on all 40 photographs, and no photograph's hit count fell.

| arm | words per photograph | pooled | invented | diacritic | folded |
|---|---|---|---|---|---|
| shipped | **67.0%** | 69.4% | 65 | 68.0% | 92.0% |
| rescue | **67.9%** | 71.0% | 67 | 68.0% | 92.0% |

Paired per photograph (n=28): mean Δ **+0.8** points, 95% +0.0 to +2.4, 2 up, 0 down, bootstrap p 0.247.

Dropped boxes the Cyrillic recogniser was asked about: 45; kept at ≥ 0.9: 7.

Photographs whose hit count changed (2):

- Putokaz2.jpg: 4 → 7 of 14
- Spanish_square_08034.JPG: 106 → 109 of 144

**Bars** (pre-registered): words per photograph must rise with the 95% interval excluding zero — **does not hold**; invented ≤ 65 — **does not hold** (67).
