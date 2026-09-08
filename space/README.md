---
title: Lilly
emoji: 🇧🇦
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: other
models:
  - Safak11/lilly
---

# Lilly

Bosnian into English — typed, spoken, or photographed.

Type a sentence and it translates. Record yourself and it listens first. Point a
camera at a sign and it reads that. The English comes back written, and out loud
if you want it.

Everything runs here. No API is called and nothing you send leaves this machine.

## What it is

Four models under one roof, published at
[Safak11/lilly](https://huggingface.co/Safak11/lilly):

| | |
|---|---|
| translator | OPUS-MT, fine-tuned, quantised to int8 |
| listen | Whisper small, fine-tuned on Bosnian speech |
| read | PaddleOCR PP-OCRv6, off the shelf, chosen by a pre-registered rule over the fine-tuned EasyOCR one |
| speak | Kokoro-82M, as published |

## How well it works

Measured, not estimated. Every number has a test set and most have a p-value;
the model card carries the full tables and the method.

| | the first builds (unrecorded) | first recorded | today |
|---|---|---|---|
| Translation, BLEU on FLORES devtest, as the user sees it | **< 30** | 37.72, language tag leaked into 308 of 1,012 outputs | **42.49**, leaked into **0** |
| Speech, word error on 200 held-out clips | **> 55%** | 38.5% | **34.9%** gated · **11.9%** large-v3, closed 8 Sep by its last look, still in the bundle |
| Photographs, words found per photograph, 40 real Commons photographs | **< 30%** | 36.0% | **67.0%** |
| Photographs, words invented that are on no sign | **> 280** | 224 | **65** |

The first column is the owner's notes from before any measurement was committed.
The second is the first number that was written down. The repository README
("Where it started") and the model card carry the full tables and the method.

Three things the numbers do not say, which the model card says at length.

The fine-tuning does **not** measurably improve understanding of Bosnian-specific
terms — a benchmark built to test exactly that claim returns 91.7% against 92.2%
at p = 0.360. The base model is already trained across South Slavic languages and
arrives there on its own.

The photograph numbers are on real photographs from Wikimedia Commons — the 40,
and a second held-out set of 280 (57.8% on its 132 with text) — transcribed by
eye, never trained on. An earlier version of this card quoted 75% from synthetic
text; the first real photograph read 36%, and that number was never real. Nobody
has yet measured a photograph taken on a phone in Bosnia.

And the 11.9% listener cleared two of the three pre-registered gate rows by a
wide margin and failed the third twice — by one word on 200 clips, then by
1.1% → 6.1% Croatian substitution (p = 0.018) on all 925, the same two words
each time. By the rule written before either run it is **closed**. Yet it is
the `listen/` this bundle has carried since 5 September, swept in by the reader
publish before any gate was run on it. That is recorded, not tidied; reverting
it to the gated whisper-small is the owner's act.

## The correction box

If a translation is wrong, the box under it takes a better one. On this Space
those corrections are **not kept** — a free Space has no persistent storage and
they are lost when it restarts. They are kept when Lilly runs on a machine with
a volume.
