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
postcard, lithograph, poster, map and pre-1940 date patterns — first returned 8
candidates. **Every candidate was then looked at, and the pattern was wrong in
both directions.** The verified count is **6 in 286 (2.1%)**, consistent with
the 1-in-50 the random sample hit.

Three of the 8 were false positives — real photographs of real signage that the
filename pattern caught by accident:

| | what it actually is |
|---|---|
| `Franciscan_church_Mostar.jpg` | street scene; Dubrovnik/Split/Sarajevo direction signs and a HOTEL sign |
| `The_old_Vratnik_fort_map.JPG` | a photograph **of** a mounted interpretive board, trilingual, in situ |
| `Zenica_map.jpg` | a photograph **of** a mounted "PLAN GRADA ZENICE" board, buildings behind it |

A map printed on a board in the street is signage, and photographing one is
exactly what the app is for. A map file is not. No filename pattern can tell
those apart, which is why this list was checked by eye and why it is quoted as a
list of six names rather than as a rule anything could re-derive.

The pattern also missed one, for the reason patterns miss things:
`Зеница_-_1909_-_чаршија_улице_Кочева_...разгледница.jpg`, a 1909 postcard —
*разгледница* is *razglednica*, and no Latin pattern reaches it.

**The exclusion list, all six looked at:**

```
Narrow-Gauge-Railway_Spalatobahn_Station-Travnik.jpg      postcard, caption across the sky
Narrow-Gauge-Railway_Spalatobahn_Station-Travnik_3_.jpg   postcard, same series
Зеница_-_1909_-_чаршија_улице_Кочева_...разгледница.jpg   postcard, caption across the top
Sarajevo_Jugoslavija_Poster.jpg                           travel poster, flat graphic art
Old_driver_s_license_of_Yugoslavia_-_page_26-27.jpg       scan of a book spread
Banjaluka_streetmap.jpg                                   OpenStreetMap raster render
```

### Two of those six are inside the scored ruler

`Banjaluka_streetmap.jpg` is not a photograph — it is an OpenStreetMap raster
render, attributed to OpenStreetMap.org. `Narrow-Gauge-Railway_Spalatobahn_
Station-Travnik.jpg` is a 1900s postcard. Both are among the 40 scored.

| | n | per-photo | pooled |
|---|---|---|---|
| as published | 28 | 54.7% | 45.0% (168/373) |
| without the two non-photographs | 26 | 53.0% | 42.7% (141/330) |

They inflate the per-photo figure by 1.7 points and the pooled one by 2.3. The
individual scores are the part worth keeping:

```
Narrow-Gauge-Railway...Travnik.jpg    9/9  = 100.0%
Banjaluka_streetmap.jpg              18/34 =  52.9%
mean over the other 26                       53.0%
```

The postcard is **the only item in the whole set the reader gets perfect**, and
it is the one whose text is flat, high-contrast, axis-aligned type with no
perspective and no lighting — the synthetic distribution exactly. That is not a
coincidence; it is this project's own thesis (synthetic reads 75% where real
photographs read 36%) turning up inside its ruler.

**The ruler is not being changed.** Replacing it is precisely what
`training/sample_photos.py` was taught to refuse, and 54.7% / 45.0% stay the
published numbers so they stay comparable to the 36.0% they are measured
against. What this section exists to do is make the caveat permanent, and to
force the choice — bar on all 28, or on the 26 — to be made *before* a run
rather than after one.

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

- **Cyrillic, and the two figures do not share a ceiling.** `app/ocr.py` builds
  `easyocr.Reader(["bs","en"])` and `latin_g2` has no Cyrillic among its 351
  classes. 26 of the 373 answer-key words are Cyrillic, so the **pooled** figure
  is capped at **93.0%** — the number quoted everywhere else in this project.
  The **per-photo** figure is capped at **90.6%**, and it is lower because
  Cyrillic is concentrated rather than spread: it sits on 7 of the 28
  photographs that carry text, and sits heavily.

  | photograph | Cyrillic | its ceiling |
  |---|---|---|
  | `Putokaz2.jpg` | 8/14 | 42.9% |
  | `Mostar_signs.JPG` | 2/4 | 50.0% |
  | `Editing_Wikipedia_Workshop_in_Visegrad_-_76.JPG` | 3/8 | 62.5% |
  | `Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg` | 7/20 | 65.0% |
  | `Sarajevo_Trebević_Sign.jpg` | 2/6 | 66.7% |
  | `Putokaz_za_manastir_Krupu.jpg` | 3/10 | 70.0% |
  | `Trg_Krajine_čajavčev_i_ulaz_u_gospodsku.jpg` | 1/5 | 80.0% |

  Averaging a fraction capped at 42.9% on one photograph against 21 uncapped
  ones is the whole of the difference. This matters to `picture-egitim` because
  "75% is 80% of the reachable range" is true of pooled and not of per-photo,
  where 75% is 82.8% of it. Against the current figures, clearing 75% means
  taking 56.6% of the remaining headroom per-photo, or 62.5% of it pooled.
  More photographs do not move any of this; a trained Cyrillic recogniser would
  lift the per-photo ceiling from 90.6% toward 100%, which is why *which build
  is being judged* has to be fixed before the run and not after.
- **Diacritics.** 180 of 1,702 real labels carry any of č ć đ š ž — 10.6%,
  exactly as documented. One precision on the per-letter counts, because a
  recogniser treats `Đ` and `đ` as different classes and the published figures
  are lower case only:

  | | lower | upper | total |
  |---|---|---|---|
  | č | 14 | 27 | 41 |
  | ć | 32 | 31 | 63 |
  | **đ** | **1** | **7** | **8** |
  | š | 31 | 31 | 62 |
  | ž | 9 | 30 | 39 |

  "đ appears exactly once" is true of the lower case; the letter appears 8 times
  across both. Signage is mostly upper case, so the real training signal is
  three to five times what the published numbers suggest. The conclusion does
  not move — 8 examples cannot teach a letter — but 8 is the number to plan
  against, not 1.

## The Cyrillic crops, for whoever fine-tunes on them

The identified next move for this lane is fine-tuning `cyrillic_g2` on the
Cyrillic crops already transcribed. Three things about that set were measured
here rather than assumed, and each changes the plan.

**There are 272 of them, not 276.** Counted three ways — labels containing any
Cyrillic letter, labels entirely Cyrillic (271), and labels containing any
non-Latin letter at all, which is what a Latin-only model literally cannot
represent — every count gives 272 (16.0% of 1,702), with zero labels that are
non-Latin without being Cyrillic. 276 appears in `HANDOFF.md` and
`RESULTS-ocr-realcrops.md` and cannot be reconstructed from the labels on disk.
All 272 crop files are present.

**They come from 37 photographs, and one of them is over a third of the set.**

```
Information_board_in_Jajce                      97 crops   35.7%
Natpis_na_Domu_kulture_u_Derventi               32
Gavrilo_Princip_plaque_1960s                    14
A_plaque_dedicated_to_British_..._Banja_Luka    12
Tabla222                                        12
                              top five combined            61.4%
median crops per source 3, seven singletons
```

So a train/valid split **by crop leaks badly**: crops off one information board
share font, lighting, camera and repeated words, and 97 of them would sit on
both sides of the split. It has to be split by **source photograph** — the
discipline `ocr_split.py` already applies by label text on the Latin side — and
even then, one photograph carrying 36% makes the split lumpy whichever side it
lands on.

For scale: the Latin move that bought 36.0% → 54.7% had 1,294 crops from 186
photographs. This is 272 from 37. Expecting a proportional gain would be wrong.

**The distinctively Serbian-Cyrillic letters are barely in the data.**

```
Ђ 3    Ј 39    Љ 3    Њ 14    Ћ  6    Џ 0
ђ 3    ј 43    љ 8    њ 12    ћ 13    џ 2
```

`Џ` has **zero** examples in the entire transcribed set. `RESULTS-ocr-cyrillic.md`
says all twelve of Ђ Ј Љ Њ Ћ Џ were verified present — that is about
`cyrillic_g2`'s **output classes**, not about our training data. Both statements
are true, they are about different things, and the difference is exactly what
would send a training run the wrong way.

This is the same shape as the diacritic problem, on the other alphabet: real
Bosnian signage does not contain enough of the rare letters to teach them, and
the synthetic generator is the only thing that can. Synthetic Cyrillic belongs
in the plan from the start, not after a run discovers Џ was never there.

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
