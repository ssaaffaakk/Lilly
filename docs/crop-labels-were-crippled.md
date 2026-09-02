# The crop labels were missing č ć đ because I configured the reader wrong

Found 2026-09-02, from practitioner research pointing at EasyOCR's per-language
character lists.

## The mechanism

`easyocr/character/bs_char.txt` contains `Š š Ž ž` and a set of Spanish and
French accents. It does **not** contain `Č č`, `Ć ć` or `Đ đ`.

`setLanguageList` (easyocr.py:295) builds the permitted set as

```python
lang_char = union(bs_char + en_char, model['character_list'] or model['symbols'])
```

and `recognition.py:119` enforces it by zeroing the probability of everything
outside that set before decoding:

```python
preds_prob[:, :, ignore_idx] = 0.
```

For the **stock** reader, `latin_g2` supplies `symbols` — punctuation only — so
the union is `bs_char + en_char + punctuation`, and č ć đ are zeroed. The
weights can produce them; EasyOCR forbids it.

For **our app**, `recog_network="lilly"` supplies `character_list` in
`lilly.yaml`, the full 351-character latin_g2 set, so the union contains them
and they decode normally.

## The consequence

`training/build_crop_notebook.py:168` cropped all 20,240 Mapillary photographs
with `easyocr.Reader(["bs", "en"], gpu=True)` — the stock configuration.

So every one of the 13,144 crop labels was produced by a reader that could not
write č, ć or đ. Measured on the same crops:

| ground truth | stock reader (labelled our crops) | our app's reader |
|---|---|---|
| Matića Brdo | Matica Brdo | **Matića Brdo** |
| MEĐU | MEDU | **MEĐU** |
| ANDRIĆ | ANDRIC | **ANDRIĆ** |
| NEŠKOVIĆ | NEŠKOVIC | **NEŠKOVIĆ** |
| OPĆI | OPér | **OPĆI** |

Note `Š` and `Ž` survive in both — they are in `bs_char.txt`. Only the three
missing letters are affected, which is exactly the pattern seen all along and
attributed to the model "dropping diacritics".

## What this invalidates

- The "1% of training rows carry a diacritic against 10.6% in human labels"
  finding was a description of this bug, not of the corpus.
- The diacritic restoration step in `lexicon_filter.py` was patching a defect
  in our own tooling rather than a limitation of the data.
- Every conclusion of the form "the reader drops diacritics" needs re-reading:
  the app's reader does not. The one that labelled the training data did.

It does not obviously explain the transfer failure — passes 14-19 also failed
on the folded score, which ignores these characters entirely. But the labels
were wrong in a systematic, correctable way, and no result from them should be
trusted until the crops are relabelled.

## The fix

Re-run the crop pass with the reader the app actually uses:

```python
easyocr.Reader(["bs", "en"], gpu=True,
               model_storage_directory=str(READ),
               user_network_directory=str(READ / "user_network"),
               recog_network="lilly")
```

The 20,240 photographs are still in `afaksrmeli/lilly-mapillary-photos`, so
this is one GPU pass, about an hour.
