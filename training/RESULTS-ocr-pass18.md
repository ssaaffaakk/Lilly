# Pass-18 — trained without the gate, measured after, reverted

Kaggle `afaksrmeli/lilly-ocr-pass18`, 2026-09-02. Asked for directly: train
the reader on the Mapillary crops and hand over the weights, rather than let
the crop gate decide whether they were worth installing.

That is what happened. `--no-install --checkpoint` trained on the 898
lexicon-checked crops for 339 steps and wrote
`lilly-read-pass18-ungated.zip`. The kernel reports ERROR, but only because
`restore_scored_photos.py` failed afterwards while rebuilding the 40 scored
photographs — downstream of the training, and after the weights were already
saved.

## Then it was measured against the reader it would replace

60 crops sampled from `data/ocr/crops/labels-human.tsv` — human labels, not
the reader's own output — read by both readers in the same process:

| | exact | ignoring diacritics |
|---|---|---|
| shipped reader | 19/60 | 23/60 |
| pass-18 | 16/60 | **20/60** |

Worse on both counts, and worse by the diacritic-blind measure that matches
the product bar. Three crops that the shipped reader gets right, this one
does not.

Reverted. `models/lilly/read/lilly.pth` is back to the reader that was there
before, verified by md5 (`ada99ec7…`). The trained weights are kept beside it
as `read-pass18.pth` so the run is not lost.

## What this settles

The four refusals before this one were not the gate being conservative. Passes
14-17 were refused on 132 held-out crops; this pass skipped the gate entirely,
and an independent check on 60 different human-labelled crops reached the same
verdict. Two disjoint measurements, one answer: training the reader on
Mapillary shop signage makes it read Bosnian signage worse.

The 898 crops are not the problem — they were lexicon-checked and had their
diacritics restored. The corpus is shop fascias (`kuca`, `CITY`, `BAZAR`,
1.2 words a crop) and both evaluation sets are plaques, information boards and
street signs. Getting better at one is not getting better at the other, and
here it costs.

## What would answer the remaining question

Whether these crops help on *shop signage* is still unmeasured, because
neither evaluation set contains any. A held-out set of hand-labelled shop
signs would settle it. Until that exists, there is no evidence for installing
this reader and two independent measurements against it.
