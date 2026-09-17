# What the reader reads off a CLEAN, legible, meaningful sign

Run 2026-09-17. Reader: PP-OCRv6 at confidence floor 0.9 (the shipped product,
`app/ocr.py`, `DEFAULT_READER = "paddle"`). No training, no new weights — this is
a **re-slice of the 40-Commons answer key** (`data/ocr/real-photos/truth.json`),
not a new run. The reader output is `reader-output-paddle-v6.json`; the floor-0.9
per-photo counts are `training/paddle-floor/the40-floor0.9.json`. Both give the
same total on the subset below (113/152), so the confidence floor drops **zero**
correct words on clean signage — the split here is at the shipped configuration.

## Why this slice exists

The published 40-Commons figure is **67.0% per-photograph / 65 invented**
(`training/paddle-floor/the40-floor0.9.md`). That set mixes three different
things and reports them as one number:

1. clear, close, legible signs and plaques with **meaningful** Bosnian/English
   words — "STARI GRAD KLJUČ", "OVDJE POČIVA 3301 BORAC SA SUTJESKE";
2. **brand logos** — "EP", "Bavaria", "Eurohaus", "INA", "Stella Artois";
3. **distant, incidental** shop signage in street scenes, legible to nobody.

The owner's instruction (2026-09-17): judge the reader on (1) only — clear HD
writing that means something, not brand names, not far-off specks. So each
text-bearing photograph was classified by legibility **before** looking at its
score, and the subset scored on its own.

## The number, on clean legible meaningful signage (14 photographs)

| slice | words | recall |
|---|---|---|
| **Latin-script words** | 113 / 127 | **89.0%** |
| Cyrillic-script words | 0 / 25 | **0.0%** |
| combined | 113 / 152 | 74.3% |

**On clean, legible, meaningful Latin Bosnian/English text the reader reads 89% of
the words.** The remaining misses are almost all Cyrillic: PP-OCRv6 ships a
Latin recogniser, so it is structurally blind to the Cyrillic copy that many BiH
signs carry beside the Latin one. That is a **script-coverage** limit fixable by
enabling the Cyrillic/multilingual recogniser — not a photo-quality problem and
not a thing retraining the Latin model would fix.

### Per photograph (KEEP), floor 0.9

| found/total | rate | photograph | what it says |
|---|---|---|---|
| 3/3 | 100% | Direction_sign_to_Old_city_of_Kljuc.jpg | STARI GRAD KLJUČ |
| 12/12 | 100% | Sarajevo_road_M-a8_IMG_1166.JPG | Trnovo Foča Mostar Zenica Tuzla |
| 4/4 | 100% | Road_to_Baljevac_-_panoramio.jpg | BALJEVAC |
| 21/22 | 95% | Assasination_Plaque.JPG | Franz Ferdinand assassination plaque, BS+EN |
| 30/34 | 88% | Banjaluka_streetmap.jpg | Banja Luka district names |
| 6/7 | 86% | War_Memorial_in_Kučine_BiH_2024.jpg | yellow direction sign, village names |
| 4/5 | 80% | Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg | OVDJE POČIVA 3301 BORAC SA SUTJESKE |
| 13/20 | 65% | Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg | DOBRO DOŠLI / WELCOME (Latin got, Cyrillic missed) |
| 5/8 | 63% | Editing_Wikipedia_Workshop_in_Visegrad_-_76.JPG | NA DRINI ĆUPRIJA / bridge over Drina |
| 6/10 | 60% | Putokaz_za_manastir_Krupu.jpg | manastir Krupa XIII vijek |
| 2/4 | 50% | Mostar_signs.JPG | Sarajevo Dubrovnik (Latin got, Cyrillic missed) |
| 2/6 | 33% | Sarajevo_Trebević_Sign.jpg | VRH TREBEVIĆA (Latin got, Cyrillic missed) |
| 1/3 | 33% | Trg-žrtava-ŠB03078.JPG | na ovom mjestu... |
| 4/14 | 29% | Putokaz2.jpg | željeznička/autobuska stanica (Latin got, Cyrillic missed) |

Every photograph scoring below 70% here loses its missing words to Cyrillic, not
to illegibility: Brod, Mostar_signs, Trebević and Putokaz2 all carry the same
words twice, once in each script, and the reader gets the Latin copy every time.

## Dropped from the slice, and why (13 photographs)

Brand logos (not meaningful words): Sarajevo_EP-Gas-Station (EP×4),
Trg-kralja-tomislav (Bavaria), Sarajevo_Trolleybus (Eurohaus), GiPS_Bus (GiPS),
Tuzla_INA_petrol_station (INA/LPG), Gospodska_ulica_27 (Malbašić company),
Trg_Krajine (Siemens/Lanaco). Distant incidental street signage:
Jewish_Street_Tuzla (Coca-Cola/Stella Artois), Street_in_Međugorje (shop/gold/
souvenirs), Mis_Irbina_Street (fragments), WV_banner_NE_Bosnia (1 far word).
Other: Narrow-Gauge-Railway (vintage postcard, half German), Spanish_square_08034
(dense wall of 144 Spanish surnames — legible, but not Bosnian/English words).

## What this does and does not license

- **Publish 89% (clean legible Latin) beside 67% (all 40 Commons).** The 67% is
  the honest number for "point a camera at any BiH sign"; the 89% is the honest
  number for "point it at a clear sign in Latin". Both are real; neither replaces
  the other.
- It does **not** reopen OCR training. The Latin reader is already at 89% on clean
  Latin text; the open lever is Cyrillic recogniser coverage (config), tested
  under the same 40-Commons gate before anything ships.
- Counts are small (127 Latin words, 25 Cyrillic). Report them beside the
  percentages; a 25-word slice is ±~19 points.

---

Re-slice of committed data; reproduce with the classification in this file
against `data/ocr/real-photos/truth.json` and `reader-output-paddle-v6.json`.
