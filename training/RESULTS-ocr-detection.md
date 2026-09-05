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
