# The real-photograph corpus, audited against its acceptance test

`picture-veri` asks for **at least 200 licensed real photographs, containing
signage, attributed**. 286 photographs were already on disk when this audit
started. The job was not to harvest more; it was to find out whether the 286
actually meet every clause, because nothing had ever checked them.

Audited 28 August 2026. Every number below is reproducible from the commands
named beside it.

## Verdict — the test passes on all four clauses

| clause | required | measured | |
|---|---|---|---|
| real photographs | ≥ 200 | **286** on disk, all 286 open, 0 corrupt | ✅ |
| licensed | every one | **286/286**, re-verified against the live Commons API | ✅ |
| attributed | every one | **286/286**, re-verified against the live Commons API | ✅ |
| containing signage | — | **47/50** sampled by eye → **≥ 239** at 95% confidence | ✅ |

The narrowest clause is signage, and it clears with 39 photographs to spare.

## Clause 1 — 286 real photographs, no padding

286 image files, 286 credit rows, **zero orphans in either direction**. The
287th directory entry is `.state`, the harvester's own ledger, not a photograph.

Every one of the 286 opens and passes `PIL.Image.verify()`: 280 JPEG, 4 PNG,
2 MPO. 549.9 MB. Short side ranges 506–3840 px, median 1944; 26 fall under the
800 px bar the harvester screened on, which is a property of the `keep_url`
original rather than of the 1280 px rendering that was screened.

**All 286 md5 hashes are distinct.** This was worth checking rather than
assuming: a duplicate inflates the count against a bar stated as a count, and
286 is only 43% above that bar.

## Clause 2 and 3 — licensed and attributed, checked against Commons itself

The licence and attribution columns were written by our own harvester from the
Commons API and **nothing had ever checked them against the source**. All 286
were re-queried (`prop=imageinfo&iiprop=extmetadata`, 6 batches of ≤50):

- **286/286 pages exist.** None missing, none redirected.
- **286/286 licences match** `LicenseShortName` exactly. Not one row claims a
  more permissive licence than Commons states.
- **286/286 attributions match** the tag-stripped `Artist` field exactly,
  including Cyrillic and diacritic names.
- **0 files carry a Commons `Restrictions` value**, consistent with our empty
  column.
- **0 files are NC, ND, fair-use or otherwise non-free.**

Spread: CC BY-SA 4.0 (157), CC BY-SA 3.0 (50), Public domain (20), CC BY 2.0
(19), CC BY-SA 2.0 (14), CC BY 3.0 (12), CC0 (10), CC BY 4.0 (3), CC BY 3.0 rs
(1). Every `source_url` is a `commons.wikimedia.org/wiki/File:` page. 132
distinct attributions, no placeholders. The 20 empty `license_url` cells are
exactly the 20 public-domain rows, which have no CC deed to link.

**What this check is and is not.** It proves the recorded values are faithful to
Commons and that nothing here is non-free or unattributable. It is not a fully
independent reading of the licences, because our harvester and this audit both
read the same `extmetadata` field. It rules out corruption, drift and
fabrication; it does not re-litigate what Commons itself asserts.

## Clause 4 — signage, checked by eye rather than by the detector

`CREDITS.tsv` has a `text_regions` column: minimum 2, median 3, maximum 87, and
no row below 2. That is **the detector's own opinion**, and using it to certify
that the detector has something to find is circular. So the clause was checked
by looking.

50 photographs drawn at random (seeds 20260828 and 20260828999, manifests in
the audit scratchpad) and inspected one at a time:

| | count |
|---|---|
| mounted signage — road sign, street plate, fascia, plaque, information board, banner | **47** |
| text-bearing but not signage | 2 |
| flat artwork with a caption, no signage | 1 |

47/50 gives a Wilson 95% lower bound of 83.8%, so **at least 239 of the 286
contain signage**. Against a bar of 200 that is comfortable, and it stays
comfortable — ≥227 — if the 15 photographs that also appear in the scored test
set are struck out of the corpus first.

The three that are not signage are worth naming, because two of them are still
useful and one is the failure mode this project has already rejected once:

- `Only_foreign_butter_in_Bosnian_supermarkets.jpg` — shelf price labels and
  product packaging. Dense real Bosnian text, photographed in the wild, just
  not on a sign.
- `GiPS_Bus_Lion_s_City_05.jpg` — vehicle livery (`VOZIMO NA PRIRODNI PLIN`).
  Same: real text, not a sign.
- `Narrow-Gauge-Railway_Spalatobahn_Station-Travnik.jpg` — a 1900s postcard
  whose only legible text is the publisher's caption printed flat across the
  sky. **This is exactly the distribution HANDOFF rejected the Europeana
  collection for**, and it is the same distribution as our synthetic crops,
  which is the thing that cannot close the gap to real photographs.

A targeted census for that last class — filename and attribution matched against
postcard, lithograph, poster, map and pre-1940 date patterns — finds **8
candidates in 286 (2.8%)**, consistent with the 1-in-50 the random sample hit.
It is a small contaminant, it is now identified, and it should be excluded when
crops are cut rather than discovered afterwards.

## Three problems found that the acceptance test does not ask about

None of these change the verdict. All three would have cost something later.

### 1. The 285 training photographs are gone — 285 broken symlinks

`data/ocr/real-photos/train-photos/` looks like 285 photographs. Every one of
them is a **dangling symlink** into
`data/ocr/real-photos/harvested/.state/staging/`, which the harvester empties
when a run ends (`harvest_sign_photos.py:1036`). Zero of the 285 resolve.

This is the trap that already ate 25 of the 40 scored photographs, firing a
second time on a different directory. `screened.tsv` shows why these in
particular were deleted: 218 of the 219 that are not also in `harvested/` carry
the verdict `drop` — *"only 1 confident region(s)"* and similar. **The crops were
cut from photographs the harvester then dropped**, so the drop path removed them.

Recoverability, checked rather than assumed:
- 66 of 285 still exist by name in `harvested/`.
- 218 of the remaining 219 are identified in `.state/screened.tsv` by Commons
  title and can be re-fetched.
- 1, `Mostar_street-2.jpg`, matches nothing anywhere and is unrecoverable.

**Nothing is blocked by this.** All 1,914 crops are real files and present, with
`labels.tsv` and `labels-human.tsv` beside them, so `picture-egitim` has its
training data. `picture-olcum` measures detection on the 40 scored photographs,
which exist. The source photographs are needed only to re-cut crops differently.

### 2. The provenance was untracked — the `truth.json` trap, again

`data/ocr/` is gitignored wholesale, and **none** of these were excepted:

- `harvested/CREDITS.tsv` — the only mapping from a local filename to the
  Commons page that licenses it. Lose it and all 286 photographs become
  unattributable, and clauses 2 and 3 of this acceptance test evaporate no
  matter what survives on disk.
- `.state/screened.tsv` — the only record of the 1,626 candidates judged, and
  the only route back to the 219 photographs above.
- `.state/staged.json` — the `screen_url` of every kept photograph, which is the
  1280 px rendering that was actually read. `keep_url` is a different picture.

All three are now tracked, with the same one-directory-at-a-time un-ignore the
transcriptions already use. Together they are 7.4k lines; the photographs they
describe are the gigabytes and stay ignored.

### 3. `harvested/` overlaps the scored test set — a live leak for training

15 of the 40 scored photographs are present in `harvested/` by filename, 4 of
them byte-identical. The other 11 are the same photograph at a different
rendering, which is a leak just the same.

The 36% → 54.7% run was clean and this audit confirms it at crop level: the
1,914 crops were cut from **200 distinct photographs, none of them scored**
(0 matches, case-insensitive). The exposure is forward-looking. `train-photos/`
was the directory that enforced the exclusion, and it is now 285 dead links, so
**the next crop run must not simply fall back to `harvested/`.**

Note in passing: `RESULTS-ocr-realcrops.md` says the crops were cut from 285
photographs. 285 was the candidate pool; 200 of them actually yielded crops.

## The ceiling this corpus cannot lift on its own

Unchanged by anything above, and it constrains `picture-egitim`:

- **Cyrillic.** `app/ocr.py` builds `easyocr.Reader(["bs","en"])` and `latin_g2`
  has no Cyrillic among its 351 classes. 7.0% of the scored answer key is
  Cyrillic, so the photograph ruler is capped at 93%. More photographs do not
  move this; a trained Cyrillic recogniser might.
- **Diacritics.** 180 of 1,702 real labels carry any of č ć đ š ž, and `đ`
  appears once in the entire set. Real signage barely teaches the letters the
  reader is worst at, which is why the synthetic crops stay in the mix. This
  corpus does not fix it either.

## Reproducing this

```
# counts, orphans, duplicates, corruption, dimensions
python3 - <<'PY'
import os, hashlib
from PIL import Image
D = "data/ocr/real-photos/harvested"
f = [x for x in os.listdir(D) if x != "CREDITS.tsv" and not x.startswith(".")]
h = {hashlib.md5(open(os.path.join(D, x), "rb").read()).hexdigest() for x in f}
print(len(f), "files,", len(h), "distinct")
PY

# the broken training set
find data/ocr/real-photos/train-photos -type l ! -exec test -e {} \; -print | wc -l   # 285

# the harvest ledger's verdicts
awk -F'\t' 'NR>1{print $2}' data/ocr/real-photos/harvested/.state/screened.tsv | sort | uniq -c
# 1114 drop / 286 keep / 219 skip / 7 refused
```

The Commons re-verification and the 50-photograph visual sample are recorded
above rather than scripted; the sample is reproducible from the two seeds.
