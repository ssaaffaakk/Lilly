# What the readers read off `test-v2`

The first numbers on the set docs/OCR-ROADMAP.md step 2 built. Set: `data/ocr/real-photos/test-v2/`, the 280 photographs frozen in `sample.txt`, key `truth-v2.json`. **Split: held out entire.** No reader in this project has trained on any of it — the pool excludes the 40 scored photographs and every photograph a labelled crop was cut from, and nothing here has ever been a training set. That is the whole point of the set and it is why these numbers compare readers where the crop scores cannot.

n: **132 photographs carry agreed text** of 280 drawn; **2,908 agreed word tokens**, 214 of them carrying č ć đ š ž, 628 of them Cyrillic. Two blind passes agreed on 2,907 of the 3,297 words either saw (88.2%), and that agreement is the ceiling on anything below. The other 148 photographs carry no text the two readers agreed on; they are excluded from every rate and are the reason the invented-word column is large.

**Measured 4 September 2026 on the owner's Mac**, all three arms in one process each, one environment (`opencv-python-headless` files at 4.10.0 — see do-not-repeat 17), through `app.ocr.scan`. The `lilly` arm is the **real shipped reader**, `models/lilly/read/lilly.pth` = md5 `2010a2d417e6c253195fa3d95ff11d33`, the weights verified at 54.7% / 180 on the 40 (`training/RESULTS-ocr-weights.md`). This replaces the first version of this file, whose `lilly` row was the 27 August Hugging Face bundle and measured a reader nobody should run. `paddle-v6` reproduced the earlier cloud measurement to the digit (60.0% / 66.5% / 2,373 / 64.0%), so the two machines agree.

## The 132 photographs with text

| arm | reader | words per photograph | pooled | invented words | diacritic words | folded |
|---|---|---|---|---|---|---|
| lilly | `easyocr:lilly` — the shipped reader, `2010a2d4` | **34.6%** | 41.3% | 2,071 | 44.9% (96/214) | 57.5% (123/214) |
| stock | `easyocr:stock` — EasyOCR's own latin_g2 | **30.0%** | 34.6% | 2,205 | 19.2% (41/214) | 41.6% |
| paddle-v6 | `paddle:PP-OCRv6_medium_det+PP-OCRv6_medium_rec`, untrained | **60.0%** | 66.5% | 2,373 | 64.0% (137/214) | 86.0% |

Diacritic words are **214**, not 25. On the 40 that column moved 4 points per letter and the roadmap wrote it off as unable to move by design; here its 95% interval is about ±6.7 points and the gap between the arms is far outside it.

**The invented column is not comparable to the 40's 180.** 148 of the 280 photographs carry no agreed text, and every word a reader returns on those counts as invented, as does every word on a sign only one transcriber could read. Compare this column between arms on this set; never against another set.

## The two numbers that matter most here

**The shipped reader reads 34.6% of a photograph's words on 132 held-out photographs, against the 54.7% published on 28.** Same weights, same code, same door; a bigger and entirely held-out set. 54.7% rests on 28 photographs with a ±5-point interval on its pooled figure; 34.6% rests on 132. The published figure is not wrong — it is the number for that set — but it is the optimistic end of what this reader does on unseen Bosnian signage, and the product claim should be built on this one.

**PP-OCRv6 wins on recall and loses on invented words.** On the 40 it was ahead on both rows (67.7% against 54.5%, 106 invented against 182). Here it takes +25.4 points of recall and returns **302 more** invented words than the shipped reader (2,373 against 2,071). The pre-registered bake-off rule — better on at least one row, worse on neither — is written against the 40 and is not applied here; but had it been applied to this set, PP-OCRv6 would **fail** it on the invented row. That reversal is the single most decision-relevant fact in this file, and it is why the engine question should not be closed on the 40 alone.

## Signs against boards

The product reads small signs and street names (roadmap, decision 1).

| arm | signs, 1–5 words (n=76) | short boards, 6–20 (n=38) | long boards, 21+ (n=18) |
|---|---|---|---|
| lilly | 33.1% (81/212, 12 read in full) | 34.4% (153/389, 1 read in full) | 41.5% (968/2307, 0 read in full) |
| stock | 27.4% (65/212, 6 read in full) | 33.6% (151/389, 1 read in full) | 33.7% (791/2307, 0 read in full) |
| paddle-v6 | 56.6% (121/212, 27 read in full) | 62.3% (246/389, 7 read in full) | 69.2% (1566/2307, 0 read in full) |

**What the bigger set changed.** On the 40, PP-OCRv6's sign row read 72.6% over 13 photographs. Over 76 it reads 56.6%. Sixteen points, and the direction of the error is the one the roadmap warned about: 13 photographs cannot carry a claim. The shipped reader's published 61.9% sign row rests on the same 13; over 76 photographs it reads **33.1%**, a 29-point correction. The sign class is where the product lives, and it is where the 40 flattered both readers most.

## Paired, photograph by photograph

Against the `lilly` arm — the shipped reader, measured here.

| arm | photographs | mean Δ points | 95% interval | better on | worse on | bootstrap p |
|---|---|---|---|---|---|---|
| stock | 132 | −4.6 | −8.5 to −0.7 | 16 | 41 | 0.023 |
| paddle-v6 | 132 | +25.4 | +19.7 to +31.2 | 78 | 11 | 0.000 |

The fine-tune is worth 4.6 points over stock EasyOCR on held-out photographs — real, but the interval reaches −0.7, and it is a fraction of the 18.7-point gain the 40 suggested (54.7% against 36.0%).

## What this does and does not settle

- **It does not settle the engine question.** The pre-registered rule (`training/PREREGISTRATION.md`, "v2 — read — bake-off") is written against the 40 and against the shipped reader reproducing 54.7% / 180. `training/RESULTS-ocr-bakeoff.md` prints no verdict because the cv2 the paddle arms need moves the shipped arm to 54.5% / 182. This set is not the pre-registered one and decides nothing on its own — but it shows the rule applied to a bigger set would answer differently on the invented row, which is a reason to re-pre-register rather than to close.
- **It does settle that the set works.** A 16-point correction to one sign row, a 29-point correction to another, and a diacritic column that is a measurement instead of noise, is what 2,907 words buy over 373.
- **PP-OCRv5 was not run here.** 76.1 s a photograph on this Mac is about five hours for 280 photographs; it is in the 40-photograph bake-off and can be added when there is a machine to spare.
- **628 Cyrillic words sit in the key.** A Latin-only recogniser cannot produce them, so they are guaranteed misses in every row above. Any future Cyrillic work (roadmap step 5) has a real test set now; any Latin claim should say this denominator includes them.

---

Generated by `training/evaluate_ocr.py` per arm into `training/bakeoff/test-v2-<arm>.json`; this file is written by hand from those.
