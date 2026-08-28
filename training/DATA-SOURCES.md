# Internet data sources for the photo reader — hunted and verified, 28 Aug 2026

41 candidate sources found by six parallel researchers, every single one then
re-fetched by an independent skeptic instructed to refute it. 36 confirmed,
5 partially confirmed, and the confirmations include several *negative* results
recorded so nobody hunts them twice. Licenses were read off the pages, not off
the claims — in four cases the verifier downloaded the actual file and checked
byte counts or font glyph tables.

## Tier 1 — big, license-clean, immediately usable

| source | what | volume | license | use |
|---|---|---|---|---|
| **Flickr API** (`flickr.photos.search`) | geo+license-filtered photos of BiH | **8,137** CC/PD for "sarajevo", **4,719** "mostar" — live counts | CC BY / BY-SA / CC0 / PD, filterable by `license=` param | training photos + unbiased ruler pool |
| **NARA catalog** (catalog.archives.gov) | DoD/SFOR-era Bosnia street photographs | **3,582** tagged Bosnia; 107 in "Sarajevo street" alone | verified verbatim: "Access: Unrestricted, Use: Unrestricted" — US gov PD | ruler pool + training |
| **Geofabrik OSM extract, BiH** | every street/shop/place name in the country | 160,384,614-byte PBF, verified by HEAD request | ODbL 1.0 (attribution+share-alike for the *data*) | synthetic corpus |
| **GeoNames BA.zip** | BiH toponym gazetteer | **51,889 rows**, verified by actual download and count | CC BY 4.0 | synthetic corpus |

Flickr caveat: hard cap of 4,000 results per query — slice by bbox/city to
harvest fully. Openverse is NOT the way in: measured cap of 240 surfaceable
results per query, which retroactively explains why the harvester's Openverse
arm underdelivered.

## Tier 2 — fonts and corpora for the synthetic generator

Fonts (all SIL OFL; the verifier downloaded each TTF and ran fontTools over the
cmap — 10/10 Bosnian Latin diacritics AND 15/15 Serbian Cyrillic glyphs
confirmed present except where noted):

- **Fira Sans Condensed** — condensed sans, the shape of most modern signage
- **Oswald** — variable 200–700, tall condensed, road-sign-like
- **PT Sans** (+ Narrow/Caption cuts) — both scripts
- **Big Shoulders Stencil** — variable stencil, **Latin only**

Corpora:

- **PleIAs/Serbian-PD** — 157M words, public domain, pre-1884 (Cyrillic-era)
- **srWaC** — 100M–1B tokens Serbian web text, CC BY-SA 3.0
- **jerteh/SrpELTeC** — 5.26M words 1840–1920, mixed Cyrillic/Latin, CC BY 4.0
- **Sag69/bosnian-dataset** — 40,712 dictionary entries + 55,628 literary
  passages, MIT
- **Wiktionary Appendix: Serbian given names** — ~280 names, disproportionately
  đ-carrying (Đorđe, Đurađ...), CC BY-SA. Directly targets đ appearing ONCE in
  our 1,702 real labels.

## Tier 3 — Commons categories the keyword crawl missed (all API-reachable)

Counts verified per category via `categoryinfo`:

- Road signs in BiH: 72 files + 12 subcats (~130 total)
- Street signs in BiH: 37 + per-city subcats
- City limit signs in BiH: 35 (Cyrillic entries confirmed present)
- Signs for National Monuments of BiH: 28
- Cyrillic inscriptions in BiH: 9 (+ Serbia: 27; parent tree ~150–250)
- National Monuments of BiH deep tree: Sarajevo alone has 100 monument
  subcategories — several hundred plaque-bearing photos, 3-level recursion
- Wikivoyage banners of BiH: 34

Worth adding as harvester seeds; a few hundred images, not thousands.

## The Cyrillic training picture — honest and thin

- **Books-of-Jeremiah/raw-OCR-serbian-cyrillic**: 108 page scans, CC BY-SA 4.0,
  ~no labels. Small.
- **pretraziva.rs / Univ. Library Svetozar Marković**: 1.7M digitized Serbian
  newspaper pages — but web viewer only, no bulk access, and the CC badge
  covers the About page's prose, not the corpus. Restricted.
- **digitalna.nb.rs**: terms forbid reproduction. Restricted.
- **DonkeySmall OCR-Cyrillic (10M images)**: *Russian* alphabet — has Ё Ъ Ы Э,
  lacks all six of Ђ Ј Љ Њ Ћ Џ. Training on it would corrupt a Serbian
  recogniser. Rejected on glyph inventory, not on volume.

Conclusion: no ready-made Serbian-Cyrillic crop dataset exists. The viable
route is our own: 276 real transcribed Cyrillic crops + synthetic Cyrillic
generated with the verified fonts (PT Sans/Fira/Oswald all carry the full
Serbian set) over the PD Serbian corpora.

## Dead ends confirmed this hunt — do not re-visit

- **interfaze-ai/ocr-mlt-50m (HF)**: card claims 50.2M samples/2.3TB — actual
  content is **60 rows**. Fabricated by six orders of magnitude.
- **ICDAR MLT 2017/2019**: languages are Chinese, Japanese, Korean, English,
  French, Arabic, Italian, German, Bangla, Hindi — no Cyrillic, no diacritic
  Latin of ours. Registration-walled besides.
- **MASTIF/rMASTIF** (Croatian traffic signs): pictogram class IDs, zero text.
- **Mapillary Traffic Sign Dataset**: annotation schema has no transcription
  field at all.
- **Wiki Loves Monuments BiH**: the 2024 category contains exactly 1 file;
  2018/2020/2023 categories don't exist.
- **Geograph**: Great Britain/Ireland only, no Balkans equivalent exists.
- **Pexels**: ToS forbids "data mining, extraction, scraping" — excluded.
- **HierText**: real and CC BY-SA, but Cyrillic is incidental in an
  English-dominant set — marginal, not a source.
- **Zenodo/figshare sweep**: papers, not datasets.
- **DVIDS**: usable in principle (verifier corrected the finder: it is NOT
  non-commercial), but the same underlying PD photos are reachable through
  NARA with cleaner terms — use NARA.
- **Openverse as a Flickr proxy**: capped at 240 results/query. Go to Flickr
  directly.

## The plan this points at

1. **Flickr harvester arm** — biggest single pool of real BiH photographs we
   have ever had access to; feeds training and the unbiased 200-photo ruler.
2. **Synthetic v2** — 4 font families × the OSM/GeoNames/names corpora, both
   scripts, đ oversampled, harder compositing. This attacks the measured
   weakness (đ×1, č×14 in real data) directly.
3. **Cyrillic fine-tune** of cyrillic_g2 on 276 real + synthetic-v2 Cyrillic.
4. **Commons category top-up** into the existing harvester.
5. **NARA pull** for the ruler pool (PD street scenes, browser-driven).
