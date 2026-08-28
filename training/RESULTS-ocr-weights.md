# Which weights produced which number

Written 28 August 2026, **before** the run that is supposed to answer it, so the
record exists whatever the answer turns out to be.

## The gap

`RESULTS-ocr-realcrops.md` publishes **54.7% of a photograph's words**. Nothing
on disk says which weights produced it.

Grepped across every `.md`, `.json` and `.py` in the repository for the three
hashes that could tie it down — `512343be` (the installed reader), `c59897bd`
(the stamp on the cached readings), `342ecb0c` (today's weight fingerprint) —
and there are **zero hits**. The number is published, quoted in `HANDOFF.md`,
and inherited as the baseline for `picture-egitim`'s bar, and it is not tied to
any artefact.

`PREREGISTRATION.md` does this correctly for the *other* file: it names
`latin_g2.pth` by md5 `469869130aad1a34e8f9086f4262bc59` and says "verified
pristine at load". That is the standard the headline figure does not meet.

## I tried to close it forensically and could not

The cached readings in `reader-output.json` carry the stamp `c59897bd3963c7ac`,
written 07:03:25. Today's directory fingerprints to `342ecb0c46eb677e`. If the
difference were only files that the Latin path never loads, the installed reader
would be provably unchanged since the scoring run.

The timeline makes that look likely:

```
06:57:29  latin_g2-previous.pth, latin_g2-realcrops.pth written   (trainer)
07:00:14  lilly.pth, lilly-previous.pth written                   (trainer installs)
07:03:25  reader-output.json written                              (the scoring run)
07:05:04  latin_g2.pth restored to pristine                       (ensure_pristine)
09:43:51  cyrillic_g2.pth arrives                                 (Cyrillic experiment)
```

`cyrillic_g2.pth` postdates the cache by two and a half hours and the shipped
Latin path never loads it, so it alone would explain an invalidation that means
nothing.

**It does not reconstruct.** I searched all 511 non-empty subsets of the current
nine weight files, crossed with four candidate mtimes for `latin_g2.pth` (its
own, its pre-restore value, the trainer's write, and CRAFT's), and **no
combination reproduces `c59897bd3963c7ac`**. The directory at 07:03 differed in
some way that removing files and retiming `latin_g2.pth` cannot express.

So the honest statement is: **the provenance of 54.7% is not recoverable from
what is on disk.** The timeline is suggestive and it is not evidence. The
validity check pre-registered for `picture-olcum` — re-read all 40 photographs
through `app.ocr.scan` and require 45.0% pooled and 54.7% per-photo — is the
only instrument that can answer it, and that is now its main purpose rather than
a preliminary to the detector measurement.

## What is installed, as of 28 August 2026

| file | bytes | md5 | modified |
|---|---|---|---|
| `craft_mlt_25k.pth` | 83,152,330 | `2f8227d2def4037cdb3b34389dcf9ec1` | 2026-08-22 16:44:20 |
| `cyrillic_g2.pth` | 15,256,523 | `19f85f43d9128a89ac21b8d6a06973fe` | 2026-08-28 09:43:51 |
| `latin_g2-previous.pth` | 15,406,141 | `469869130aad1a34e8f9086f4262bc59` | 2026-08-28 06:57:29 |
| `latin_g2-realcrops.pth` | 15,406,789 | `2010a2d417e6c253195fa3d95ff11d33` | 2026-08-28 06:57:29 |
| **`latin_g2.pth`** | 15,406,141 | **`469869130aad1a34e8f9086f4262bc59`** | 2026-08-28 07:05:04 |
| `lilly-previous.pth` | 15,406,289 | `5eb18322dda279af802d6f7fbbe9a3b2` | 2026-08-28 07:00:14 |
| **`lilly.pth`** | 15,406,289 | **`512343be80e0290955b0c7e0deeb5430`** | 2026-08-28 07:00:14 |
| `user_network/lilly.py` | 916 | `7f0e45e69e9b4bcd4582ed655225555a` | 2026-08-27 04:11:15 |
| `user_network/lilly.yaml` | 838 | `15f5a98443b1ec3b93661815f9b73292` | 2026-08-27 04:11:17 |

`latin_g2.pth` matches `469869130aad1a34e8f9086f4262bc59` exactly — the value
`PREREGISTRATION.md` records as pristine. The `LILLY_READER=stock` "before"
build has not drifted, and `latin_g2-previous.pth` is byte-identical to it,
which is `ensure_pristine()` having done its job at 07:05.

`lilly.pth` is what `app/ocr.py` actually loads, via `recog_network="lilly"`.
**`512343be80e0290955b0c7e0deeb5430` is the number to quote** once the validity
check ties it to a score.

## Why the cache invalidated on something that may not matter

`evaluate_ocr.py:133` fingerprints on **size and mtime**, not content:

```
content-based fingerprint : 133ebf63b6dca5bb
size+mtime fingerprint    : 342ecb0c46eb677e   <- what the cache uses
```

The docstring explains the choice and the reasoning is sound as far as it goes —
`latin_g2.pth` is 15 MB, this runs on every invocation, and hashing a hundred
megabytes to answer "did the reader change" is expensive.

The cost is that **copying a weight file, restoring one, or merely touching one
throws away an eighty-minute cache without a single weight having changed** —
which is what appears to have happened here. A content hash is stable against
all three.

This is **not being changed now.** Altering a scorer between its pre-registration
and its run is the wrong moment whatever the change, and the lead has confirmed
that as an instruction. Logged for afterwards, alongside the import-time
`claim(1.4)` at `evaluate_ocr.py:40` — which refuses a fully cached run that
would load no model at all.

## The rule this file exists to establish

**A number whose weights are not named is a number nobody can re-run.** When the
validity check passes, `lilly.pth`'s md5 goes next to the figure in
`RESULTS-ocr-realcrops.md`, the way `PREREGISTRATION.md` already does it for
`latin_g2.pth`. If it fails, that failure is the result and it outranks the
detector measurement entirely.

This is the fourth time this project has lost the link between a result and the
thing that produced it: `truth.json` untracked, the harvester deleting the
ruler, the trainer writing to a file the app does not open, and now a headline
figure with no record of its weights. The first three were each fixed where they
happened. This one is fixed by naming the hash every time a score is published.
