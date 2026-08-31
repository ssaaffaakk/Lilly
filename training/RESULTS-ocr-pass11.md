# Pass-11 refused: plates then human still spent the letters

Kernel `safaksideacc2/lilly-ocr`, Tesla T4, 31 August 2026, git `6310d90`.
API status **ERROR**. Wall ~5m39s (`19:00:40Z`–`19:06:19Z`). Fail-stop did
what it was for: the crop gate returned 1, the notebook did not zip
`lilly-read.zip`, leftover Output stubs are 829 / 853 bytes and must not be
installed.

Pass-11 was two sequential `train_ocr` calls, never mixed in one list:

1. **18,045 sign-letter plates**, 2 epochs, `--no-install --checkpoint`
2. **1,294 human crops**, 3 epochs from that checkpoint, crop gate

Same 132 real valid crops as every pass since pass-6. Same 25 Bosnian
letters on 23 of those crops.

| Stage | train | real words | Bosnian letters (25) |
| :--- | :--- | :--- | :--- |
| shipped (before stage 1) | — | **41.7%** | **64.0%** (16/25) |
| after plates | 18,045 syn* | 42.4% | 64.0% (16/25) |
| after human | 1,294 human | **41.7%** | **60.0%** (15/25) |

Stage 1 did not spend letters. The +0.7pp word tick is one crop on 132.
Stage 2 put the words back and spent one letter class: one of four
`PUTNIČKI` went Č → C (`PUTNICKI`). That is the same spend pass-10 made on
a human-only mix. `iznad česme` stayed a miss (`časou`).

The gate line:

```
real  words 42.4% -> 41.7%, letters 64.0% -> 60.0%
real-crop words did not rise — not replacing the shipped reader
```

Photographs were never scored. There is no `lilly-read.zip` and no
`read-trained.pth` worth fetching. The app reader is still pass-6
(`RESULTS-ocr-realcrops.md`: 54.7% per photograph, 45.0% pooled, 44%
diacritic, 180 invented).

**Do not relaunch pass-11.** `scripts/kaggle_train.py ocr` exits if the
notebook still says `heavy-pass11`. Weekly GPU on this account is 0.53 h
until 5 Sep 2026.

The 1,294 Latin human crops are exhausted as a train set: mixed with
plates (pass-8, pass-9), human-only (pass-10), or plates-then-human
(pass-11), the crop gate either spends čćđšž or drops real words. A later
pass needs new unique human labels, not another schedule on this 1,294.
