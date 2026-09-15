# WikiMatrix bs–en alignment — LaBSE sample filter

## Decision

Use a LaBSE cosine threshold of **0.55** as the proposed full-pass filter.
This is only a launch-ready data-quality instrument.  It has **not** been run
over the full corpus and it has not been used to retrain or otherwise change a
training mix.

The input was the WikiMatrix portion of `data/clean/train.tsv`, derived from
OPUS WikiMatrix v1 (`data/scripts/download_data.py` records the source URL).
WikiMatrix is the web-mined portion already identified as roughly one sixth
misaligned in [RESULTS-en-bs-hrv.md](RESULTS-en-bs-hrv.md).  The prior data
audit also found WikiMatrix's crossed-row flag rate to be 22/325 (6.77%),
substantially above SETIMES's 20/5,975 (0.33%); that is why a bilingual semantic
signal is worth testing here.

## Reproducible sample

Command, run locally on CPU on 15 September 2026:

```bash
python3 scripts/filter_wikimatrix_align.py \
  --sample-size 3000 --seed 20260915 --batch-size 16 \
  --threshold 0.55 --examples 20 \
  --write-scores /private/tmp/wikimatrix-labse-sample-20260915.tsv
```

The scorer uses `sentence-transformers/LaBSE`, mean-pools each side's
last-hidden-state token representations with the attention mask, L2-normalizes
them, then calculates their cosine.  Sampling is seeded reservoir sampling, so
it reads all **165,516** current WikiMatrix rows but embeds only the selected
**3,000**. It neither reads held-out data nor rewrites training data.

| quantity | result |
|---|---:|
| scored sample | 3,000 / 165,516 WikiMatrix rows |
| cosine min / p01 / p05 | 0.434 / 0.562 / 0.659 |
| p10 / median / p90 | 0.718 / 0.893 / 0.948 |
| p95 / p99 / max | 0.959 / 0.978 / 0.997 |
| proposed `< 0.55` tail | **23 / 3,000 = 0.77%** (Wilson 95% CI **0.51–1.15%**) |
| inspected low-tail pairs | **20 / 20** plainly non-translations (Wilson 95% CI **83.9–100%**) |

The number above is a sample count, not an extrapolated full-corpus drop count.
The threshold lands below the first percentile and keeps the review set sharply
about mismatched facts or entirely unrelated text.  Less conservative cutoffs
would need a separately blinded review: `<0.60` flags 59/3,000, `<0.65` flags
130/3,000, and `<0.70` flags 249/3,000.  They are not licensed by this sample.

## Twenty lowest-scoring pairs, read by hand

Each row below is a genuine mismatch (not merely a loose translation).  This
is the qualitative evidence for the conservative threshold; scores are LaBSE
cosines rounded to three places.

| cosine | Bosnian side | English side |
|---:|---|---|
| .434 | Dakle to nije ru’jet. | See also Yu. |
| .444 | Zeami je podučavan od strane njegovog oca, Kan'amija, koji je također bio glumac. | Ling Wan had a friend, Cho Cho, who was also ugly. |
| .457 | Armenija je prekinuta Azerbejdžanom da bi imala mali dio Baškend potpuno okružen njime. | Nevertheless, Thomas stayed in the Union Army with some degree of suspicion surrounding him. |
| .460 | Dodir je jednostavna, instinktivna i univerzalna ljudska radnja. | All of this serves to create a very hot, humid and smoky fire. |
| .468 | Minber (često i mimber) (arapski: منبر) jeste prostor u džamiji gdje imam (predvodnik molitve) stoji za vrijeme hutbe. | The compost is mainly used in the gardens, but on occasion has been auctioned as part of a fundraising event for the gardens. |
| .470 | U sezoni 1934-35 ispada u niži rang, kao i u sezoni 1938-39. | Late Night, as an entity, is in its 38th season. |
| .472 | Obično slijedi svoje instinkte koji mu govore kome može, a kome ne može (ili ne želi) vjerovati. | He established himself as a rationalist who did not subscribe to any religion. |
| .478 | Hikmet Hodžić (SDA). | Knowledge of right and wrong - The root of wisdom (zhi). |
| .504 | Dodjeljivane su (obično) po tri nagrade (godišnje, u vidu odgovarajućih knjiga), (na republičkom i saveznom nivou). | Three shall be the number thou shalt count, and the number of the counting shall be three. |
| .510 | Tako Edip postaje kralj Tebe i dobija kraljicu Jokastu, svoju majku, za ženu. | Arthur finally accepts his responsibilities as king and thanks the woman, who introduces herself as Guinevere. |
| .513 | Prema jednom mitu, Eros je Hefestov, a ne Afroditin sin. | Eventually it is discovered that the foe is an angel and not the real Son of God. |
| .513 | Općina Doboj Istok. | PKT \| PUPUK KALIMANTAN TIMUR. |
| .521 | Zašto ste požurili i o naređenje Gospodara svoga se oglušili? – i ploče baci, i brata svoga za kosu dohvati i poče ga vući sebi. | Dad Ku Dhaawacmay Dagaalkii Deegaanka Dharkayn Geenyo Ku Dhex Maray Labada Belood Ee Walaaalaha Ah Oo Dhaawacooda Hada La Keenay Cusbitaalka Magaalada Hargeysa. |
| .529 | U distriktu Brčko je 8 župa. | In total, the castle has eight storeys. |
| .530 | To je dakle tradicija kojoj se obraća i kojoj želi pripadati. | A man caught between what he is and what he wants to be. |
| .534 | Po nekim se statistikama ova sezona smatra drugom sezonom u ženskoj konkurenciji. | Superman fans generally regard it as the second season of The New Adventures. |
| .534 | Hokkaido je najrjeđe naseljeno od japanskih glavnih ostrva. | Eugene brought much-needed rain to the major Hawaiian Islands. |
| .537 | A mi Te slavimo, zahvaljujući Ti, i, kako Tebi dolikuje, veličamo! | By Allah, we swear by the almighty Allah we will never stop fighting you until you leave us alone. |
| .538 | 3. kv. | CV CV |
| .539 | Obavezni (farz) namaz je 4 rekata, no moguće je prije toga klanjati 4 rekata prije, kao i 2 rekata dobrovoljnog (sunnet) namaza poslije. | Crest: Issuant from a coronet heightened with four ears of corn (one and two-halves visible) alternating with four millstones (two visible) Or, a maple leaf Gules. |

## Owner-gated next step

The complete pass is intentionally explicit:

```bash
python3 scripts/filter_wikimatrix_align.py --all --threshold 0.55 \
  --write-scores /path/to/wikimatrix-labse-scores.tsv \
  --write-kept /path/to/wikimatrix-labse-kept.tsv
```

`--all` prints an owner-gated warning. The resulting kept file must still pass
the repository's held-out leakage audit before it can be proposed as a training
input. Retraining the shipped bs→en recipe is owner-gated Kaggle work and has
not been launched.

## Pre-registration text for the Leader

> **Run C — WikiMatrix semantic-alignment filter (bs→en).** Before training,
> score every WikiMatrix pair in the current bs→en mix with
> `sentence-transformers/LaBSE`; remove only pairs whose L2-normalized
> cross-lingual cosine is `<0.55`. The cutoff was fixed after a seeded,
> sample-only 3,000-pair audit: 23/3,000 (0.77%, Wilson 95% CI 0.51–1.15%) fell
> below it, and all 20 manually read tail pairs were plainly misaligned (20/20,
> Wilson 95% CI 83.9–100%). No held-out text is read by the filter. Retrain the
> shipped bs→en recipe unchanged except for this removal; compare against the
> shipped build on held-out FLORES with paired bootstrap. The arm passes only if
> chrF2 and BLEU are not below the shipped result (report point deltas, counts,
> and bootstrap intervals). No result from the training-side sample decides the
> arm.
