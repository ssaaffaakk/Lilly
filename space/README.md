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
| read | EasyOCR's recogniser, fine-tuned on Bosnian letters |
| speak | Kokoro-82M, as published |

## How well it works

Measured, not estimated. Every number has a test set and most have a p-value;
the model card carries the full tables and the method.

| | before | after |
|---|---|---|
| Translation, BLEU on 2,009 FLORES pairs | 40.81 | **42.18** |
| Speech, word error on 200 held-out clips | 38.5% | **35.5%** |
| Photographs, whole words read correctly | 47.1% | **75.0%** |

Two things the numbers do not say, which the model card says at length.

The fine-tuning does **not** measurably improve understanding of Bosnian-specific
terms — a benchmark built to test exactly that claim returns 91.7% against 92.2%
at p = 0.360. The base model is already trained across South Slavic languages and
arrives there on its own.

And every photograph the reader was trained and scored on is synthesised. Nobody
has yet measured what it does on a photograph taken in Bosnia.

## The correction box

If a translation is wrong, the box under it takes a better one. On this Space
those corrections are **not kept** — a free Space has no persistent storage and
they are lost when it restarts. They are kept when Lilly runs on a machine with
a volume.
