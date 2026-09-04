# Engine bake-off — PaddleOCR untrained against the shipped reader

Run 2026-09-04 09:09 by `scripts/bakeoff_ocr.py`, rule from `training/PREREGISTRATION.md`, "v2 — read — bake-off". Every arm through `app.ocr.scan` on the same 40 photographs and `app.ocr.read_regions` on the same crops; `training/bakeoff/<arm>-*.json` are the raw counts.

**The crop row was not measured for lilly, stock, paddle-v6, paddle-v5.** The crop PNGs under `data/ocr/crops/` are not on the machine this ran on (they are on the Mac and in the Kaggle dataset `lilly-ocr-crops`, which needs the token). The bar below is the 40 photographs, as pre-registered; the crop row is reported beside it and is filled in by running `training/score_crops.py --json training/bakeoff/<arm>-crops.json` per arm where the crops are, then `scripts/bakeoff_ocr.py --assemble-only`.

**The shipped arm does not reproduce its published numbers** (35.9% / 225 against 54.7% / 180). By the pre-registration no comparison is made: the run is invalid as a bake-off until the reason is found, and the tables below stand only as the record of what was measured.

## Notes from this run

- The shipped arm in this run is the Hugging Face bundle `Safak11/lilly` as `scripts/fetch_models.py` installs it (last commit 27 Aug 2026). Its per-photograph rows match the 36.0% report (`training/RESULTS-ocr.md`) on 25 of 28 photographs; it is the reader from before the real-crop training. The 54.7% reader exists on the Mac and in the Kaggle `lilly-ocr` output only. Publish it (`scripts/publish_to_hf.py Safak11/lilly --upload`) and re-run this command; until then the bar this rule needs does not exist off the Mac.
- Both Paddle arms are the untrained published weights with oneDNN off (paddlepaddle 3.3.1's executor dies on the detector; `app/ocr.py`, `get_paddle_reader`). Seconds per photograph are this 4-core cloud CPU on plain kernels, not the Mac; PP-OCRv5's server detector is the slow one.
- Read against the *published* 54.7% / 180 — which the pre-registration does not accept as the bar — paddle-v6 (67.7% / 106) and paddle-v5 (66.6% / 125) would both clear it, by 12–13 points of recall with 55–74 fewer invented words, and the stock EasyOCR arm (46.1% / 188) would not. That is what the re-run on the real shipped reader should expect; it decides nothing here.
- The crop row and its 71 held-out crops were not measured: the PNGs are on the Mac and behind the Kaggle token. Run `training/score_crops.py --json training/bakeoff/<arm>-crops.json` per arm there and `--assemble-only`.

## The 40 photographs

| arm | reader | words per photograph | pooled | invented words | diacritic words | folded |
|---|---|---|---|---|---|---|
| lilly | `easyocr:lilly` | **35.9%** | 16.1% | 225 | 36.0% | 52.0% |
| stock | `easyocr:stock` | **46.1%** | 35.4% | 188 | 12.0% | 52.0% |
| paddle-v6 | `paddle:PP-OCRv6_medium_det+PP-OCRv6_medium_rec:3.7.0` | **67.7%** | 71.0% | 106 | 68.0% | 92.0% |
| paddle-v5 | `paddle:PP-OCRv5_server_det+latin_PP-OCRv5_mobile_rec:3.7.0` | **66.6%** | 68.9% | 125 | 44.0% | 76.0% |

## Signs against boards

The product reads small signs and street names (docs/OCR-ROADMAP.md, decision 1). Reported beside the bar, which stays the mean over every photograph.

| arm | signs, 1-5 words | short boards, 6-20 | long boards, 21+ |
|---|---|---|---|
| lilly | 53.8% (n=13) | 23.0% (n=12) | 9.5% (n=3) |
| stock | 52.9% (n=13) | 38.6% (n=12) | 46.9% (n=3) |
| paddle-v6 | 72.6% (n=13) | 57.7% (n=12) | 86.5% (n=3) |
| paddle-v5 | 72.1% (n=13) | 56.0% (n=12) | 84.8% (n=3) |

## Paired against the shipped reader, photograph by photograph

Reported; the bar is the table above.

| arm | photographs | mean Δ points | better on | worse on | bootstrap p |
|---|---|---|---|---|---|
| stock | 28 | +10.3 | 16 | 5 | 0.183 |
| paddle-v6 | 28 | +31.8 | 19 | 1 | 0.000 |
| paddle-v5 | 28 | +30.7 | 21 | 1 | 0.000 |

## The human crops, split named

Only the held-out row compares readers; the shipped reader trained on the other.

| arm | held out: exact | held out: folded | 95% (folded) | training-side: folded |
|---|---|---|---|---|
| lilly | not measured here | — | — | — |
| stock | not measured here | — | — | — |
| paddle-v6 | not measured here | — | — | — |
| paddle-v5 | not measured here | — | — | — |

## Speed and letters

| arm | seconds per photograph (CPU) | recogniser dictionary |
|---|---|---|
| lilly | 4.2 (War_Memorial_in_Kučine_BiH_2024.jpg, 1.2 MP) | all ten letters present (allowlist, checked at load) |
| stock | 3.9 (War_Memorial_in_Kučine_BiH_2024.jpg, 1.2 MP) | all ten letters present (allowlist, checked at load) |
| paddle-v6 | 9.3 (War_Memorial_in_Kučine_BiH_2024.jpg, 1.2 MP) | all ten letters present |
| paddle-v5 | 36.4 (War_Memorial_in_Kučine_BiH_2024.jpg, 1.2 MP) | all ten letters present |

## Verdict, by the pre-registered rule

**No verdict.** The shipped arm did not reproduce 54.7% / 180, so the bar this run was to be measured against is not in it. Nothing is adopted and nothing is closed; fix the shipped arm and re-run.

---

Generated by `scripts/bakeoff_ocr.py`.
