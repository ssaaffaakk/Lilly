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

## test-v2 — Cyrillic rescue against the shipped configuration

Shipped arm `aea5890abfdb7ba3`; rescue arm `f7d79d3d3858129d`.

**Rescue-off equals shipped, mechanically:** every shipped word is in the rescue reading on all 280 photographs, and no photograph's hit count fell.

| arm | words per photograph | pooled | invented | diacritic | folded |
|---|---|---|---|---|---|
| shipped | **57.8%** | 64.8% | 450 | 62.1% | 81.8% |
| rescue | **59.0%** | 70.0% | 530 | 62.1% | 81.8% |

Paired per photograph (n=132): mean Δ **+1.3** points, 95% +0.4 to +2.4, 10 up, 0 down, bootstrap p 0.000.

Dropped boxes the Cyrillic recogniser was asked about: 867; kept at ≥ 0.9: 63.

Photographs whose hit count changed (10):

- Banja_Luka_April_2011_5645905076_.jpg: 11 → 14 of 20
- Bihac_postcard_1899.jpg: 0 → 1 of 4
- Catholique_Serbe_-_A1604.jpg: 0 → 1 of 5
- Info_Atik_džamija.jpg: 716 → 830 of 1091
- Kosača-fort-Ljubuški02210.JPG: 171 → 173 of 206
- Life_scupture_Banja_Luka_01.jpg: 158 → 166 of 290
- Mostar14BIH22.JPG: 2 → 3 of 4
- Зеница_-_1884_-_ред_вожње.jpg: 84 → 102 of 147
- Зеница_-_1931_-_жељезничари_ускотрачне_пруге.jpg: 0 → 1 of 2
- Зеница_20190821_174400.jpg: 28 → 30 of 39

**Bars** (pre-registered): words per photograph must rise with the 95% interval excluding zero — **holds**; invented ≤ 450 — **does not hold** (530).

**Verdict, 5 September 2026 — the rescue does not ship.** The pre-registration
named this outcome before the run: "recall rises and invented words exceed
450: does not ship." Words found per photograph rose 57.8% → 59.0% (paired
+1.3, 95% +0.4 to +2.4, 10 photographs up, none down) and pooled 64.8% →
70.0%, because the Cyrillic recogniser read **145 of the 628 Cyrillic key
words that the shipped configuration cannot read at all** — 111 of them on
`Info_Atik_džamija.jpg`, 18 on the Zenica timetable. But the 63 boxes it kept
(of 867 it was asked about) also added **80 invented words, 450 → 530**: 25 on
`Kosača-fort-Ljubuški02210.JPG` and 26 on the Atik mosque board
(`training/invented_words.py`, strict definition, decision 4). About 150 real
words bought with 80 that are on no sign is the trade the invented bar exists
to refuse, and it refuses it here exactly as it refused every EasyOCR union
rule (`RESULTS-ocr-cyrillic.md`: +2.3 points for 180 → 246). The shipped
configuration stays; `LILLY_PADDLE_CYRILLIC_RESCUE` stays in `app/ocr.py`,
off by default, as the measured way back to this number.

**What it says for step 5 and step 7.** A second untrained recogniser reading
the dropped boxes at the same floor is not a substitute for a recogniser that
knows Cyrillic: the rescued words are concentrated where the Cyrillic model is
sure (one printed board), and its confident misreadings of Latin boxes it was
never meant for are the 80. The Cyrillic question moves to training — a
dictionary that carries Serbian Cyrillic and labelled Cyrillic lines (the 272
blind-labelled Cyrillic crops are the seed, and they are few) — and is its
own pre-registration, not a variant of this rule. Nothing here was re-tuned
after the number: one floor, one look, one verdict.
