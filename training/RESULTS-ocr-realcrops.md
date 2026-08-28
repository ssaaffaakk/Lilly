# What the reader reads off a real photograph — after training on real crops

Measured 28 August 2026 through `app.ocr.scan`, the path the photo endpoint
serves, on the same 40 photographs and the same answer key as the 36% figure it
is being compared with. The reading cache was re-read from scratch: it now
carries a fingerprint of the weight files, saw they had changed, and refused
itself.

## The comparison

| | before | after |
|---|---|---|
| **words found per photograph** | **36.0%** | **54.7%** |
| all words, pooled | 16.9% (63/373) | 45.0% (168/373) |
| words with č ć đ š ž | 36.0% (9/25) | 44.0% (11/25) |
| the same, diacritics folded | 48.0% (12/25) | 60.0% (15/25) |
| words invented that are on no sign | 224 | **180** |

The last row matters as much as the first. Recall alone can be bought by
guessing more, and this reader guesses *less* than the one it replaces while
finding more — 44 fewer invented words. That is the direction a user feels.

## What it trained on

22,946 crops: **1,294 hand-transcribed real ones**, 18,060 synthetic, 3,592
photographed-synthetic. Valid was 2,480 with zero label text shared with train.

The real crops are the new thing. 1,914 were cut from 285 photographs that
exclude all 40 scored ones, twelve annotators transcribed them blind — never
shown the reader's own guess, since a proofreader agrees with a confident wrong
answer far more readily than a transcriber independently reproduces it — and
1,702 were usable, 39 being false detections with no text and 173 unreadable.

## Two ceilings, both measured

**Cyrillic is unreachable.** 276 of the 1,702 labels (16.2%) are Cyrillic and
had to be dropped: `latin_g2` has 351 output classes and not one is Cyrillic, so
the trainer dies on the first batch containing one. On the scored photographs
7.0% of the answer key's words are Cyrillic, across 7 of the 40. `app/ocr.py`
builds `easyocr.Reader(["bs","en"])`, which is Latin-only, so **the honest
ceiling on this ruler is 93%, not 100%**, and closing that gap needs a second
recogniser rather than a better one.

**Diacritics are barely taught by real signage.** Only 180 of 1,702 real labels
contain any of č ć đ š ž. đ appears exactly once in the whole set, č fourteen
times, ž nine. That is why the synthetic crops were kept rather than replaced:
they are the only thing in the mix that teaches the letters the reader is worst
at. The diacritic column above moved least of any figure, which is consistent.

## The held-out crop numbers, for completeness

62.2% → 85.4% words exact, 70.5% → 88.8% of Bosnian letters. Those are the
trainer's own gate and they are measured on a valid set that is 94% synthetic —
the generator already known to read 75% where real photographs read 36%. They
are reported because the gate used them, not because they describe the product.
The 54.7% above is the number that does.


28 photographs from Wikimedia Commons, none of them ours, transcribed by eye before the reader was run on them. 12 more carried no legible text and are excluded from the rates below — a photograph with nothing to read cannot be read wrongly.

| | words found | of | rate |
|---|---|---|---|
| All words | 168 | 373 | **45.0%** |
| Words with č ć đ š ž | 11 | 25 | **44.0%** |
| The same words, diacritics folded away | 15 | 25 | **60.0%** |

Weighting every photograph equally instead of every word, the reader finds **54.7%** of the words on a photograph. The two differ because the text is not spread evenly: `Spanish_square_08034.JPG` alone holds 144 of the 373 words in the answer key. The per-photograph figure is the one that describes pointing a camera at a sign; the pooled figure describes reading a wall of text.

Two independent readers transcribed these photographs without seeing each other's work or the machine's, and agreed on 91.2% of the words either of them saw. Only the words both saw are in the answer key, so that agreement is also the ceiling on how precise anything here can be.

The reader also returned 180 words that are not on any sign in these photographs. That is the cost a user pays for text the detector invented out of brickwork and foliage, and it is not visible in a recall figure.

## Per photograph

| Photograph | words | diacritic words |
|---|---|---|
| Gospodska_ulica_27.JPG | 0/2 | 0/1 |
| Jewish_Street_Tuzla_Bosnia.jpg | 0/7 | — |
| Sarajevo_Trebević_Sign.jpg | 0/6 | 0/1 |
| WV_banner_NE_Bosnia_Tuzla_old_town.jpg | 0/1 | — |
| Putokaz2.jpg | 1/14 | 0/2 |
| Street_in_Međugorje.jpg | 2/20 | — |
| Mis_Irbina_Street_in_Sarajevo_03.jpg | 1/6 | — |
| Trg_Krajine_čajavčev_i_ulaz_u_gospodsku.jpg | 1/5 | 0/1 |
| Spanish_square_08034.JPG | 45/144 | — |
| Trg-žrtava-ŠB03078.JPG | 1/3 | — |
| Mostar_signs.JPG | 2/4 | — |
| Banjaluka_streetmap.jpg | 18/34 | 5/11 |
| Tuzla_-_INA_petrol_station_2019_.jpg | 10/17 | — |
| Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg | 12/20 | 1/1 |
| Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg | 3/5 | 0/1 |
| Editing_Wikipedia_Workshop_in_Visegrad_-_76.JPG | 5/8 | — |
| GiPS_Bus_Lion_s_City_03.jpg | 2/3 | — |
| Putokaz_za_manastir_Krupu.jpg | 7/10 | — |
| War_Memorial_in_Kučine_BiH_2024.jpg | 5/7 | 3/5 |
| Sarajevo_EP-Gas-Station_Bulevar-Mese-Selimovica_2011-11-04_2_.jpg | 3/4 | — |
| Assasination_Plaque.JPG | 19/22 | — |
| Direction_sign_to_Old_city_of_Kljuc.jpg | 3/3 | 1/1 |
| Narrow-Gauge-Railway_Spalatobahn_Station-Travnik.jpg | 9/9 | — |
| Road_to_Baljevac_-_panoramio.jpg | 4/4 | — |
| Sarajevo_Trolleybus-4416_Line-102_2011-11-05.jpg | 1/1 | — |
| Sarajevo_road_M-a8_IMG_1166.JPG | 12/12 | 1/1 |
| Trg-kralja-tomislav-livno00658.JPG | 1/1 | — |
| Tuzla_-_Ulica_Turalibegova_-_n18_2019_.jpg | 1/1 | — |

---

Generated by `training/evaluate_ocr.py`.
