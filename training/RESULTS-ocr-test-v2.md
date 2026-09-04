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

**The invented column counts extra words on the 132 photographs with agreed text, and nothing on the 148 without** — `evaluate_ocr.py` skips a photograph whose key is empty before it counts. (An earlier version of this file said the blanks were counted; they are not.) It is also not comparable to the 40's 180: a bigger set, and a key built to a different coverage — see "What the invented column counts here" below before reading it at all.

## The two numbers that matter most here

**The shipped reader reads 34.6% of a photograph's words on 132 held-out photographs, against the 54.7% published on 28.** Same weights, same code, same door; a bigger and entirely held-out set. 54.7% rests on 28 photographs with a ±5-point interval on its pooled figure; 34.6% rests on 132. The published figure is not wrong — it is the number for that set — but it is the optimistic end of what this reader does on unseen Bosnian signage, and the product claim should be built on this one.

**PP-OCRv6 takes +25.4 points of recall and returns 302 more "invented" words than the shipped reader (2,373 against 2,071) — and the second number is mostly one photograph.** On the 40 it was ahead on both rows (67.7% against 54.5%, 106 invented against 182); here the invented row reverses, and the next section says what that row is made of before anyone reads it as a verdict.

## What the invented column counts here

`training/invented_words.py`, over the same readings the table above was scored from, the same count three ways:

All three rows are this Mac's readings, 4 Sep; the stock row previously carried the cloud session's (2,201 / 2,178 / 2,126), which is a two-machine comparison of the kind do-not-repeat 7 forbids. paddle-v6 is identical on both machines, so only stock moved, by four words.

| arm | vs the agreed key (the table's number) | vs anything either pass marked clear | vs anything either pass wrote at all |
|---|---|---|---|
| lilly (real reader, `2010a2d4`) | 2,071 | 2,039 | **1,972** |
| stock | 2,205 | 2,185 | 2,132 |
| paddle-v6 | 2,373 | 2,281 | **2,084** |

**1,120 of paddle-v6's 2,373 — 47% — are on `Зеница_20190821_174244.jpg`**, a museum panel of typed NDH-era decrees. Its key holds 28 agreed words because both transcribers wrote that the body paragraphs were below the resolution of the 1280 px rendering and declined to guess them; PP-OCRv6 read them. Its top ten photographs — that panel, a second museum board, the 1436 charter panel, the trilingual Ljubuški board, the 1927 poster — carry 81% of its invented words; 66 of the 132 photographs have two or fewer. Counted against everything either transcriber wrote, clear or unclear, paddle-v6's 2,373 becomes 2,084 and stock's 2,201 becomes 2,126: 289 of paddle-v6's "invented" words are text a person saw and wrote down.

So on this set the column measures where the transcription stopped, not where the reader invented. The 40 were transcribed to exhaustion; test-v2's dense boards were not, on purpose and on the record in `pass-a.json` and `pass-b.json`. The reversal on the invented row is real as a number and unsafe as a decision in either direction: **the definition of an invented word on test-v2 — against the union of both passes, or with dense boards handled explicitly — goes into `PREREGISTRATION.md` before this set decides the engine question.** **Measured on the Mac, 4 Sep: the real reader's 2,071 are not concentrated on that panel.** Only **53** of them sit on `Зеница_20190821_174244.jpg` — the museum panel that carries 1,120 of paddle-v6's, 47% of its total. The shipped reader's own worst photograph is `Info_Atik_džamija.jpg` (536, a board whose key holds 8 agreed words against 1,091 words of unclear lines), then the Ljubuški fort board (207) and the 1436 charter panel (204). Its top ten carry **1,482 of 2,071 (72%)**, against paddle-v6's 81%, and **56 of the 132 photographs have two or fewer**. So both readers pile their extra words onto the same handful of dense, under-transcribed boards; they simply pile them onto different ones, and PP-OCRv6 concentrates harder.

**The comparison the bake-off's invented row would actually make on this set: against anything either transcriber wrote, paddle-v6 is _above_ the real shipped reader — 2,084 to 1,972, 112 words worse.** That is the strictest possible reading in PP-OCRv6's favour, counting every unclear line a person managed to put down as legitimate text, and it still does not reverse. The gap narrows as the definition loosens — 302 words against the agreed key, 242 against anything marked clear, 112 against anything written at all — but on all three definitions the shipped reader returns fewer invented words than PP-OCRv6. The invented row does not clear on test-v2 under any of the three, and 289 of paddle-v6's words being real text a person saw explains part of the gap without closing it.

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

- **It does not settle the engine question.** The pre-registered rule (`training/PREREGISTRATION.md`, "v2 — read — bake-off") is written against the 40 and against the shipped reader reproducing 54.7% / 180. `training/RESULTS-ocr-bakeoff.md` prints no verdict because the cv2 the paddle arms need moves the shipped arm to 54.5% / 182. This set is not the pre-registered one and decides nothing on its own. The rule applied to it would answer differently on the invented row — but that row here is dominated by transcription coverage on a few dense boards ("What the invented column counts here"), so the different answer is not yet the reader's. Pre-register the definition first; then this set can decide, either way.
- **It does settle that the set works.** A 16-point correction to one sign row, a 29-point correction to another, and a diacritic column that is a measurement instead of noise, is what 2,907 words buy over 373.
- **PP-OCRv5 was not run here.** 76.1 s a photograph on this Mac is about five hours for 280 photographs; it is in the 40-photograph bake-off and can be added when there is a machine to spare.
- **628 Cyrillic words sit in the key.** A Latin-only recogniser cannot produce them, so they are guaranteed misses in every row above. Any future Cyrillic work (roadmap step 5) has a real test set now; any Latin claim should say this denominator includes them.

---

Generated by `training/evaluate_ocr.py` per arm into `training/bakeoff/test-v2-<arm>.json`; this file is written by hand from those.
