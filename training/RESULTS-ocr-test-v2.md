# What the readers read off `test-v2`

The first numbers on the set docs/OCR-ROADMAP.md step 2 built. Set: `data/ocr/real-photos/test-v2/`, the 280 photographs frozen in `sample.txt`, key `truth-v2.json`. **Split: held out entire.** No reader in this project has trained on any of it — the pool excludes the 40 scored photographs and every photograph a labelled crop was cut from, and nothing here has ever been a training set. That is the whole point of the set and it is why these numbers compare readers where the crop scores cannot.

n: **132 photographs carry agreed text** of 280 drawn; **2,908 agreed word tokens**, 214 of them carrying č ć đ š ž, 628 of them Cyrillic. Two blind passes agreed on 2,907 of the 3,297 words either saw (88.2%), and that agreement is the ceiling on anything below. The other 148 photographs carry no text the two readers agreed on; they are excluded from every rate and are the reason the invented-word column is large.

## Which reader is which — read this before the table

**The `lilly` arm below is not the shipped 54.7% reader.** It is what `scripts/fetch_models.py` installs from `Safak11/lilly`, whose last push was 27 Aug 2026, before the real-crop training. On the 40 scored photographs it reproduces the *old* 36.0% report's rows on 25 of 28 photographs, not 54.7% / 180 (`training/RESULTS-ocr-bakeoff.md`). The 54.7% weights exist on the owner's Mac and in the Kaggle `lilly-ocr` output only. **So the `lilly` row measures a reader nobody should be running, and no engine decision may be taken from it.** Re-run this on the Mac after `scripts/publish_to_hf.py Safak11/lilly --upload`.

## The 132 photographs with text

| arm | reader | words per photograph | pooled | invented words | diacritic words | folded |
|---|---|---|---|---|---|---|
| lilly | `easyocr:lilly` — **the 27 Aug bundle, not the 54.7% reader** | **17.5%** | 12.0% | 1,843 | 12.6% (27/214) | 15.9% |
| stock | `easyocr:stock` — EasyOCR's own latin_g2 | **30.5%** | 34.8% | 2,201 | 19.2% (41/214) | 41.6% |
| paddle-v6 | `paddle:PP-OCRv6_medium_det+PP-OCRv6_medium_rec`, untrained | **60.0%** | 66.5% | 2,373 | 64.0% (137/214) | 86.0% |

Diacritic words are **214**, not 25. On the 40 that column moved 4 points per letter and the roadmap wrote it off as unable to move by design; here its 95% interval is about ±6.7 points and the gap between the arms is far outside it.

**The invented column is not comparable to the 40's 180.** 148 of the 280 photographs carry no agreed text, and every word a reader returns on those counts as invented, as does every word on a sign only one transcriber could read. Compare this column between arms on this set; never against another set.

## Signs against boards

The product reads small signs and street names (roadmap, decision 1).

| arm | signs, 1–5 words (n=76) | short boards, 6–20 (n=38) | long boards, 21+ (n=18) |
|---|---|---|---|
| lilly | 18.1% (38/212, 6 read in full) | 16.3% (70/389, 0 read in full) | 17.4% (240/2307, 0 read in full) |
| stock | 27.6% (66/212, 6 read in full) | 34.5% (154/389, 1 read in full) | 34.2% (791/2307, 0 read in full) |
| paddle-v6 | 56.6% (121/212, 27 read in full) | 62.3% (246/389, 7 read in full) | 69.2% (1566/2307, 0 read in full) |

**What the bigger set changed.** On the 40, PP-OCRv6's sign row read 72.6% over 13 photographs. Over 76 it reads 56.6%. Sixteen points, and the direction of the error is the one the roadmap warned about: 13 photographs cannot carry a claim. The same holds for the shipped reader's published 61.9% sign row — it too rests on those 13.

## Paired, photograph by photograph

Against the `lilly` arm as measured here — which, per the warning above, is the 27 Aug reader. Reported; it decides nothing.

| arm | photographs | mean Δ points | 95% interval | better on | worse on | bootstrap p |
|---|---|---|---|---|---|---|
| stock | 132 | +13.0 | +8.7 to +17.6 | 55 | 9 | 0.000 |
| paddle-v6 | 132 | +42.5 | +36.2 to +48.8 | 95 | 4 | 0.000 |

## What this does and does not settle

- **It does not settle the engine question.** The pre-registered rule (`training/PREREGISTRATION.md`, "v2 — read — bake-off") is written against the 40 and against the shipped reader reproducing 54.7% / 180. Neither condition is met here: this is a different set and the wrong weights. `training/RESULTS-ocr-bakeoff.md` prints no verdict for the same reason.
- **It does settle that the set works.** A 16-point correction to a sign row on its first use, and a diacritic column that is a measurement instead of noise, is what 2,907 words buy over 373.
- **PP-OCRv5 was not run here.** Its server detector costs ~36 s a photograph on this CPU; it is in the 40-photograph bake-off and belongs in the Mac re-run.
- **628 Cyrillic words sit in the key.** A Latin-only recogniser cannot produce them, so they are guaranteed misses in every row above. Any future Cyrillic work (roadmap step 5) has a real test set now; any Latin claim should say this denominator includes them.

## How to re-run

```
python3 data/scripts/fetch_test_v2.py          # the photographs are not in git
python3 training/evaluate_ocr.py \
  --truth data/ocr/real-photos/test-v2/truth-v2.json \
  --photos data/ocr/real-photos/test-v2/photos \
  --sample data/ocr/real-photos/test-v2/sample.txt \
  --cache data/ocr/real-photos/test-v2/reader-output-<arm>.json \
  --out training/RESULTS-ocr-test-v2.md
```

Every arm needs its own `--cache`: the cache is stamped with `app.ocr.reader_identity()`, so a shared file would be thrown away and re-read.

---

Run 4 Sep 2026 from a Claude Code cloud session, 4-core CPU. Raw counts in `training/bakeoff/test-v2-<arm>.json`.
