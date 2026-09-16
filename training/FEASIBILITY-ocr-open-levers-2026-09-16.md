# OCR open detector levers — feasibility before another look

Written 16 September 2026. This is a **measurement and planning note**, not a
run result and not authority to change the shipped reader. No photograph was
passed through an OCR reader, no model was loaded, no product code was changed,
and no Kaggle/GPU job was launched for this note.

The governing records are `docs/OCR-ROADMAP.md` and
`training/PREREGISTRATION.md`. The recogniser is closed across all three looks
(exact at 0.9, exact at 0.94, meaning at 0.94); detector side length is also
closed. Mapillary passes 8–19 and Cyrillic remain closed. The only questions
considered here are the two deferred detector-side levers the roadmap names.

## Verdict

1. **Full Commons originals through the unchanged shipped 2 MP path are worth
   one owner-approved measurement.** Existing files show that 75 of the 132
   held-out test-v2 photographs (56.8%) are 1280 px downscales below the app's
   2 MP cap. Their median file is 1.092 MP, so an original large enough to reach
   the cap supplies a median 1.353x increase in linear resolution. This is a
   plausible, cheap input-only lever and changes neither weights nor product
   code.
2. **CRAFT-detect + PP-OCRv6-recognise is technically feasible but is not
   answerable from the current caches.** The cached blind detection judgements
   show useful signal, but the CRAFT cache contains EasyOCR readings and the
   PP-OCRv6 cache contains readings of PP-OCRv6's own boxes. They cannot be
   composed into the hybrid's output. A standalone measurement harness must
   crop the cached CRAFT boxes and pass those crops to the published PP-OCRv6
   recogniser before the hybrid deserves its one test-v2 look.

Recommended order: seek approval for the full-original, unchanged-cap
measurement below. Keep the hybrid deferred until its standalone measurement
path is reviewed and separately pre-registered. Neither result would itself
authorise a reader swap or an app working-size change.

## Lever 1 — higher-resolution source versus a higher working-size cap

These are two different experiments.

- `scripts/fetch_highres_and_score.py` fetches the Commons original and calls
  `app.ocr.scan`. The shipped 2 MP cap still applies. This measures a realistic
  high-resolution upload under the product as shipped.
- `training/evaluate_ocr.py --full-res` bypasses the 2 MP cap. On the committed
  test-v2 files it can only expose pixels present in those files; it does not
  recover the Commons originals. A true raised-cap/full-original experiment is
  therefore a separate candidate and needs its own pre-registration.

### What the existing held-out artifacts say

Split: **test-v2, held-out, never trained on**, 132 photographs with agreed
text. The committed baseline is the shipped PP-OCRv6 reader at confidence floor
0.9: mean recall **57.8% over 132 photographs**, pooled **1,884/2,908 scorer
tokens (64.8%)**, and **450 invented words** under owner decision 4's strict
count.

The committed image headers divide the 132 as follows:

| present input | photographs | shipped exact recall on that subset | feasibility reading |
|---|---:|---:|---|
| one side is 1280 px and the file is below 2 MP | **75/132 (56.8%)** | mean **58.2% over 75 photographs**; pooled **568/800 (71.0%)** | can gain source pixels up to the existing 2 MP cap if the Commons original is larger |
| file is already at or above 2 MP | **40/132 (30.3%)** | mean **58.6% over 40 photographs**; pooled **1,085/1,785 (60.8%)** | already enters the same 2 MP cap; a larger original should not add working pixels |
| neither side reaches 1280 px | **17/132 (12.9%)** | mean **53.8% over 17 photographs**; pooled **231/323 (71.5%)** | likely already at the Commons source size; original dimensions must be recorded rather than assumed |

Thus the realistic-input experiment has room to affect 75/132 photographs,
not all 132. That is enough to justify a measurement, not enough to predict a
pass. The informal 8/4/2 MP single-photo check in `app/ocr.py` remains a
negative prior; it is not a substitute for this paired held-out comparison.

### Readiness issue in the existing Mac script

There is no `training/highres/` result or reading cache today: this look has not
run. The script is close to ready but cannot support the requested 132-photo
gate unchanged:

- it permits up to 15% failed original fetches;
- it then invokes `evaluate_ocr.py` with the same cache and the committed
  downscaled photo directory;
- any original absent from the cache is consequently read from the downscaled
  file, producing a mixed-resolution arm without identifying which entries
  were substituted.

For the gate below, the preflight must require **132/132 originals read at the
registered treatment** and refuse otherwise. It must also record original and
working dimensions per photograph, keep the high-resolution cache separate,
and compute the paired interval before printing a verdict. This is measurement
plumbing only; it is not a change to `app/ocr.py`.

## Lever 2 — CRAFT detection with PP-OCRv6 recognition

### What is already measured

Split: **the-40 held-out Commons diagnostic**, excluded from every OCR training
set; 373 agreed words on 28 photographs with text. This is feasibility evidence,
not the test-v2 gate.

| detector | agreed words boxed | mean detection recall per photograph | cached regions |
|---|---:|---:|---:|
| PP-OCRv6 medium | **316/373 (84.7%)** | **89.8% over 28 photographs** | 266 boxes over all 40 photographs |
| CRAFT | **339/373 (90.9%)** | **93.1% over 28 photographs** | 299 boxes over all 40 photographs |

Paired per photograph, CRAFT minus PP-OCRv6 detection recall is **+3.3
points, 95% -0.9 to +7.8**, with 7 photographs up, 2 down and 19 unchanged.
CRAFT is not a superset: it has 35 gross additional covered words and loses 12
words PP-OCRv6 boxed, for a net 23. The losses are concentrated on the Banja
Luka street map (34/34 to 24/34 boxed) and `Street_in_Međugorje.jpg` (12/20 to
10/20); the main gain is the Spanish-square small type (116/144 to 138/144).

The shipped recogniser reads 259 of the 316 truth words its own detector boxed
at floor 0.9, **259/316 (82.0%) recognition given detection**. Applying that
rate unchanged to CRAFT's 339 covered words would suggest about 278 correct
words rather than 259. That is only planning arithmetic: CRAFT's extra boxes
are disproportionately tiny and hard, its crops differ, and its 299 regions
create an unmeasured strict-invented risk. It is not a hybrid score and has no
decision value.

### Why the caches cannot answer the hybrid

- `detection-boxes-easyocr-lilly.json` stores CRAFT geometry with the retired
  EasyOCR recogniser's text.
- `detection-boxes-paddle-*.json` and the Paddle reading caches store
  PP-OCRv6 recognition on PP-OCRv6 detector crops.
- Even overlapping boxes are different crops; copying the Paddle text from a
  nearby PP-OCRv6 box would measure a cache-matching heuristic, not
  CRAFT-detect + PP-OCRv6-recognise.
- No CRAFT boxes or human detection count exist for test-v2.

The composition itself is feasible outside the product path: the CRAFT boxes
are committed in the same working-image coordinates, PaddleX's
`TextRecognition` accepts arbitrary image crops, and the repository already
uses `CropByPolys` for recogniser-only rescue reads. The honest next artifact,
if the owner chooses this lever, is a standalone measurement script that:

1. loads the committed CRAFT boxes and crops them with a fixed polygon rule;
2. runs the unmodified published `PP-OCRv6_medium_rec` at a floor selected on
   the-40 only;
3. reproduces a complete the-40 report, including strict invented words and
   timing; and
4. only after that path is frozen, reads test-v2 once under its own
   pre-registration.

No product integration should be written before that measurement passes the
held-out test-v2 gate.

## Copy-ready pre-registration draft for the leader

This is a proposed **amendment to append before any run** to the existing
`training/PREREGISTRATION.md` section "the reader on full-resolution inputs".
It deliberately names the narrower experiment the current Mac script can
answer: higher-resolution originals through the unchanged shipped 2 MP cap.

> ### Amendment — full Commons originals through the shipped 2 MP path
>
> Written before an original-resolution test-v2 reading exists. This is a
> measurement of the shipped reader on realistic source photographs, not a
> working-size change and not a product candidate.
>
> **Question.** On the same held-out test-v2 photographs, does supplying the
> Commons original to the unchanged shipped path improve exact words found per
> photograph over the committed 1280 px benchmark without increasing strict
> invented words?
>
> **Fixed arm.** PP-OCRv6 medium detector + medium recogniser, confidence floor
> 0.9, document preprocessing off, app paragraph grouping unchanged, and the
> app's 2,000,000-working-pixel cap unchanged. The only treatment is the input
> file: Commons original instead of the committed downscale. No threshold,
> detector side length, model, grouping rule or working-size cap is swept.
>
> **Set and fail-stop.** test-v2 is held out and is never trained on. The gate
> is the same 132 photographs with agreed text and the same `truth-v2.json`.
> All 132 originals must be fetched and read; a failed or ambiguous fetch stops
> the measurement. A downscaled file is never substituted into the candidate
> cache. Record source and working dimensions for every photograph. The reader
> identity must reproduce the shipped PP-OCRv6 floor-0.9 identity before the
> first read.
>
> **Comparator.** The committed downscaled shipped result:
> `training/paddle-floor/test-v2-floor0.9.json`, mean exact recall 57.8% over
> 132 photographs, pooled 1,884/2,908 scorer tokens (64.8%), strict invented
> words 450.
>
> **Two bars; both must hold.** (1) Exact words found per photograph must rise
> above 57.8%, and the paired 95% bootstrap interval of candidate minus shipped
> over the 132 photographs (10,000 resamples, seed 0) must exclude zero on the
> positive side. (2) Invented words under owner decision 4's strict count must
> stay at or below 450. A point increase whose interval includes zero is "no
> change", not a pass.
>
> **One look.** Each original is read once into a fingerprinted cache. The
> confidence floor and cap do not move after the result. No second test-v2
> treatment, no per-photo selection and no exclusion after seeing readings.
>
> **Report.** State held-out test-v2 and n=132; put counts beside every
> percentage and a 95% interval beside every delta; report pooled recall,
> strict invented words, sign/short-board/long-board rows, up/down photograph
> counts, fetch integrity and source/working dimension strata. The committed
> 1280 px arm and original-source arm go through the same scorer.
>
> **Outcomes.** If both bars hold, record that 57.8% understated the shipped
> reader on realistic high-resolution uploads. This authorises no product
> change and says nothing about raising the 2 MP cap. If recall does not rise,
> the realistic-input question closes and 57.8% remains the benchmark. If
> recall rises but invented exceeds 450, the measurement fails the product
> gate and no higher figure is promoted. Raising or removing the app cap is a
> separate lever with its own pre-registration and owner approval.

## Stop

No owner approval was inferred. The shipped reader remains PP-OCRv6 at floor
0.9 and the product's 2 MP working cap is unchanged. No closed pass is reopened.
