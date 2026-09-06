# test-mly — the street-level half, both readers

The product reads street signs from a phone. `test-mly` is that domain: 240
Mapillary street-level photographs (Zagreb 107, Sarajevo 57, Split 41, Mostar
28, Tuzla 5, Zenica 2; all CC BY-SA 4.0), drawn by `training/build_test_mly.py`
and transcribed by two blind vision-agent passes (`training/transcribe/`),
intersected by `build_truth.py` into `test-mly/truth-mly.json`.

**The key is small, and that is the finding's first fact.** 427 words both
transcribers saw, of 546 either saw (**78% agreement**), on **94 of the 240**
photographs. The 40 gave 373 agreed words on 28 photographs; scaling projected
~1,500 here, and it came in at 427. Street-level Mapillary is mostly distant,
motion-blurred, 360-panorama, or has privacy-blurred plates, so a much smaller
share carries text two people can agree on. **427 words is a wide interval**
(a three-point per-photograph move is inside the noise), and every number
below is reported with its n.

**Stated selection bias, unchanged from the draw:** the harvest kept a
Mapillary photograph only when Lilly's *own EasyOCR reader* found two or more
words in it (`harvest_mapillary.py`). So this set is "street photographs the
old EasyOCR reader already finds text in" — if anything favourable to EasyOCR,
and an upper bound on its recall. The result below is despite that tilt.

## Both readers, scored through the app's door

| reader | words per photograph | pooled | invented | diacritic words | folded |
|---|---|---|---|---|---|
| **paddle-v6, floor 0.9 (shipped)** | **57.3%** | 64.2% | 118 | 22.2% | 33.3% |
| easyocr `lilly.pth` `2010a2d4` (the previous shipped reader) | **18.6%** | 26.0% | 312 | 16.7% | 22.2% |

n = 94 photographs with agreed text, 427 agreed words. Per-photograph weights
every photograph equally (what a user experiences); pooled weights every word,
and one photograph (`mly_453786132390094`, a ski shop) holds 33 of the 427.

**Paired, per photograph: PP-OCRv6 is +38.7 points** (95% bootstrap +31.0 to
+46.7, 59 photographs better, 2 worse, p < 0.001) — and it does it while
inventing **fewer** words, 118 against 312. On the product's real domain the
shipped reader more than triples the old one's recall and more than halves its
hallucination, on a set drawn to favour the old one. This is the strongest
evidence yet for step 6 (the switch to PP-OCRv6), because it is measured where
the product is actually pointed rather than on Commons photographs of plaques.

## Why the easyocr figure is 2010a2d4 and not this box's first run

The first cloud scoring used the `lilly.pth` this container fetched on 3
September (md5 `5eb18322`, `lilly-previous.pth`), before the corrected reader
was published to `Safak11/lilly`. That crippled reader scored 9.3% here — a
number of the wrong weights, not the shipped reader. The correct weights (md5
`2010a2d417e6c253195fa3d95ff11d33`, the 54.7%/34.6% reader of the 40 and
test-v2) were fetched from Hugging Face and the arm re-scored; **18.6% is the
shipped-EasyOCR figure** and the one above. (do-not-repeat 18: never trust a
reading cache, or a weight file, across an environment change without checking
the md5.)

Raw per-photograph tables: `training/bakeoff/test-mly-paddle-v6.json`,
`training/bakeoff/test-mly-lilly.json`. Caches (environment-bound) are not
committed.
