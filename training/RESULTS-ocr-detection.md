# Detection recall R_d on the 40 — two detectors, two blind counters each

Pre-registered: `training/PREREGISTRATION.md`, "v2 — picture — picture-olcum"
(the definition, 373 words on 28 photographs, partial overlap counts) and its
addendum "for two detectors, written 5 September 2026, before the count" (both
detectors, floor off for PP-OCRv6, the validity check per arm, the pooled
figures named in advance, the count by agents instead of a person, the
reliability check at 38 disagreements).

**How the numbers were made.** `training/measure_detection.py` drew every box
each detector produced on the 40 (PP-OCRv6 medium detector with the
recognition floor off, 5 September 2026; CRAFT with `lilly.pth`, cv2
4.10.0.84). `training/count_detection.py validity` checked each boxes file
against the scorer's cache of the same arm on the 40: **40 of 40 identical for
both**, so the detectors drawn are the ones the bake-off scored. Two blind
counters per detector, each a vision agent working from
`training/transcribe/COUNT-BRIEF.md` with the photograph, the overlay (box
indices only, never the reader's text) and the key's words; every one of the
373 words judged twice per detector, saved as
`data/ocr/real-photos/detection-count/<arm>/counter-{a,b}-NN.json`.
`count_detection.py merge` takes the agreed count as R_d.

**The deviation, recorded.** The pre-registration says "a human count". The
counters were agents, on the owner's delegation of 5 September 2026 ("sen
nasıl uygun gördüysen öyle"); the key itself was built the same way. Each
counter's own figure and the disagreements are beside the agreed figure so a
reader can see how much the two eyes differed. The counters were cut off once
by the account's usage limit and resumed from their saved files; entries
already judged were not revisited.

**Pooled figures divided by R_d** (named before the count): PP-OCRv6 floor off
71.0%, PP-OCRv6 at the shipped floor 0.9 69.4%, EasyOCR 44.5%.

The two sections below are `count_detection.py merge` output, unedited.

## Detection recall — paddle-PP-OCRv6_medium_det_PP-OCRv6_medium_rec-3.7.0

Key: 373 agreed words on 28 photographs (`data/ocr/real-photos/truth.json`). Two blind counters; a word is covered when both say a box overlaps it.

| | covered | of | R_d |
|---|---|---|---|
| counter a | 316 | 373 | 84.7% |
| counter b | 318 | 373 | 85.3% |
| **agreed (R_d)** | **316** | 373 | **84.7%** |

Disagreements: 2 of 373 words (0.5%); not located by a counter: 0 judgements.

| pooled recall | R_d | recognition given detection |
|---|---|---|
| PP-OCRv6 floor off: 71.0% | 84.7% | 83.8% |
| PP-OCRv6 at the shipped floor 0.9: 69.4% | 84.7% | 81.9% |

| photograph | words | a | b | agreed | disagree |
|---|---|---|---|---|---|
| War_Memorial_in_Kučine_BiH_2024.jpg | 7 | 7 | 7 | 7 | 0 |
| Road_to_Baljevac_-_panoramio.jpg | 4 | 4 | 4 | 4 | 0 |
| Trg-žrtava-ŠB03078.JPG | 3 | 3 | 3 | 3 | 0 |
| Spanish_square_08034.JPG | 144 | 116 | 116 | 116 | 0 |
| Editing_Wikipedia_Workshop_in_Visegrad_-_76.JPG | 8 | 5 | 7 | 5 | 2 |
| Sarajevo_road_M-a8_IMG_1166.JPG | 12 | 12 | 12 | 12 | 0 |
| Putokaz_za_manastir_Krupu.jpg | 10 | 9 | 9 | 9 | 0 |
| Narrow-Gauge-Railway_Spalatobahn_Station-Travnik.jpg | 9 | 9 | 9 | 9 | 0 |
| Trg_Krajine_čajavčev_i_ulaz_u_gospodsku.jpg | 5 | 5 | 5 | 5 | 0 |
| Banjaluka_streetmap.jpg | 34 | 34 | 34 | 34 | 0 |
| Sarajevo_EP-Gas-Station_Bulevar-Mese-Selimovica_2011-11-04_2_.jpg | 4 | 2 | 2 | 2 | 0 |
| Direction_sign_to_Old_city_of_Kljuc.jpg | 3 | 3 | 3 | 3 | 0 |
| Sarajevo_Trebević_Sign.jpg | 6 | 6 | 6 | 6 | 0 |
| Mis_Irbina_Street_in_Sarajevo_03.jpg | 6 | 5 | 5 | 5 | 0 |
| Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg | 20 | 14 | 14 | 14 | 0 |
| Putokaz2.jpg | 14 | 13 | 13 | 13 | 0 |
| GiPS_Bus_Lion_s_City_03.jpg | 3 | 3 | 3 | 3 | 0 |
| Mostar_signs.JPG | 4 | 4 | 4 | 4 | 0 |
| Street_in_Međugorje.jpg | 20 | 12 | 12 | 12 | 0 |
| Tuzla_-_Ulica_Turalibegova_-_n18_2019_.jpg | 1 | 1 | 1 | 1 | 0 |
| Jewish_Street_Tuzla_Bosnia.jpg | 7 | 3 | 3 | 3 | 0 |
| Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg | 5 | 5 | 5 | 5 | 0 |
| Trg-kralja-tomislav-livno00658.JPG | 1 | 1 | 1 | 1 | 0 |
| Gospodska_ulica_27.JPG | 2 | 2 | 2 | 2 | 0 |
| Assasination_Plaque.JPG | 22 | 22 | 22 | 22 | 0 |
| WV_banner_NE_Bosnia_Tuzla_old_town.jpg | 1 | 1 | 1 | 1 | 0 |
| Sarajevo_Trolleybus-4416_Line-102_2011-11-05.jpg | 1 | 1 | 1 | 1 | 0 |
| Tuzla_-_INA_petrol_station_2019_.jpg | 17 | 14 | 14 | 14 | 0 |

**Reading the PP-OCRv6 number** (the addendum's bands, applied as written):
R_d = **84.7%** (316 of 373) sits at the top of the "middling" band, 0.3
points under the "high" line. The detector loses 57 of the key's words; 28 of
them are on `Spanish_square_08034.JPG` — the leading "D." initials and the
rank abbreviations of a memorial's name columns, text a few pixels high — and
8 on `Street_in_Međugorje.jpg` (unboxed shop signs). Recognition given
detection is **83.8%** with the floor off and 81.9% at the shipped floor, so
the two stages lose words in near-equal shares: of 100 key words, about 15 are
never boxed and about 14 more are boxed and misread. Reported beside, deciding
nothing: the per-photograph mean of R_d is 89.8%, above the pooled figure
because the misses are concentrated on the one 144-word photograph. What this
says for step 7: a recogniser fine-tune has 84.7% pooled to work under on this
set, and the 75% per-photograph bar of `picture-egitim` is arithmetically
reachable (per-photograph R_d 89.8%); but the detector is not innocent, and
the dense small-type case is where it fails. The two counters disagreed on 2
words of 373 (both on the Višegrad bridge plaque's descenders), far inside the
38-word reliability line.

## Detection recall — easyocr-lilly

Key: 373 agreed words on 28 photographs (`data/ocr/real-photos/truth.json`). Two blind counters; a word is covered when both say a box overlaps it.

| | covered | of | R_d |
|---|---|---|---|
| counter a | 340 | 373 | 91.2% |
| counter b | 339 | 373 | 90.9% |
| **agreed (R_d)** | **339** | 373 | **90.9%** |

Disagreements: 1 of 373 words (0.3%); not located by a counter: 0 judgements.

| pooled recall | R_d | recognition given detection |
|---|---|---|
| EasyOCR lilly.pth, cv2 4.10.0: 44.5% | 90.9% | 49.0% |

| photograph | words | a | b | agreed | disagree |
|---|---|---|---|---|---|
| War_Memorial_in_Kučine_BiH_2024.jpg | 7 | 7 | 7 | 7 | 0 |
| Road_to_Baljevac_-_panoramio.jpg | 4 | 4 | 4 | 4 | 0 |
| Trg-žrtava-ŠB03078.JPG | 3 | 3 | 3 | 3 | 0 |
| Spanish_square_08034.JPG | 144 | 139 | 138 | 138 | 1 |
| Editing_Wikipedia_Workshop_in_Visegrad_-_76.JPG | 8 | 8 | 8 | 8 | 0 |
| Sarajevo_road_M-a8_IMG_1166.JPG | 12 | 12 | 12 | 12 | 0 |
| Putokaz_za_manastir_Krupu.jpg | 10 | 10 | 10 | 10 | 0 |
| Narrow-Gauge-Railway_Spalatobahn_Station-Travnik.jpg | 9 | 9 | 9 | 9 | 0 |
| Trg_Krajine_čajavčev_i_ulaz_u_gospodsku.jpg | 5 | 5 | 5 | 5 | 0 |
| Banjaluka_streetmap.jpg | 34 | 24 | 24 | 24 | 0 |
| Sarajevo_EP-Gas-Station_Bulevar-Mese-Selimovica_2011-11-04_2_.jpg | 4 | 3 | 3 | 3 | 0 |
| Direction_sign_to_Old_city_of_Kljuc.jpg | 3 | 3 | 3 | 3 | 0 |
| Sarajevo_Trebević_Sign.jpg | 6 | 6 | 6 | 6 | 0 |
| Mis_Irbina_Street_in_Sarajevo_03.jpg | 6 | 5 | 5 | 5 | 0 |
| Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg | 20 | 19 | 19 | 19 | 0 |
| Putokaz2.jpg | 14 | 14 | 14 | 14 | 0 |
| GiPS_Bus_Lion_s_City_03.jpg | 3 | 3 | 3 | 3 | 0 |
| Mostar_signs.JPG | 4 | 4 | 4 | 4 | 0 |
| Street_in_Međugorje.jpg | 20 | 10 | 10 | 10 | 0 |
| Tuzla_-_Ulica_Turalibegova_-_n18_2019_.jpg | 1 | 1 | 1 | 1 | 0 |
| Jewish_Street_Tuzla_Bosnia.jpg | 7 | 3 | 3 | 3 | 0 |
| Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg | 5 | 5 | 5 | 5 | 0 |
| Trg-kralja-tomislav-livno00658.JPG | 1 | 1 | 1 | 1 | 0 |
| Gospodska_ulica_27.JPG | 2 | 2 | 2 | 2 | 0 |
| Assasination_Plaque.JPG | 22 | 22 | 22 | 22 | 0 |
| WV_banner_NE_Bosnia_Tuzla_old_town.jpg | 1 | 1 | 1 | 1 | 0 |
| Sarajevo_Trolleybus-4416_Line-102_2011-11-05.jpg | 1 | 1 | 1 | 1 | 0 |
| Tuzla_-_INA_petrol_station_2019_.jpg | 17 | 16 | 16 | 16 | 0 |

**Reading the CRAFT number, beside PP-OCRv6's.** CRAFT (the detector behind
the 54.7%) boxes **90.9%** of the key's words (339 of 373; the counters agree
on 372 of 373), 6.2 points more than PP-OCRv6's detector, and its per-photograph
mean is 93.1%. Its 34 misses are the Banja Luka street map's route shields
(M-4, M-16, R405 — 8 of its 10 there) and Međugorje's shop signs (10); on the
Spanish-square memorial it boxes 138 of 144 where PP-OCRv6 boxes 116, so the
few-pixel type that PP-OCRv6's detector skips is not beyond a detector's
reach. But recognition given detection is **49.0%** for the EasyOCR reader
against 83.8% for PP-OCRv6: of 100 key words, CRAFT misses 9 and then the
fine-tuned recogniser misreads 46 of the 91 it was handed. The switch to
PP-OCRv6 (step 6) therefore bought 34 points of recognition for 6 points of
detection, and the decomposition says where each engine's ceiling is: for the
shipped configuration the recogniser and the detector now lose words in equal
measure (about 15 and 14 per 100), and the detector's share sits on dense
small type. A detector-side move (a larger `text_det_limit_side_len`, or
PP-OCRv6's server detector) is a separate experiment and gets its own
pre-registration before anyone runs it; nothing here chooses it.
