# The voice pipeline's control -- the reading

Kaggle, Tesla T4. Pre-registered: PREREGISTRATION.md, 'v7 -- speak -- the control'.
Listener e6bb58483586b06c; test prefix: 200 clips, 167 sentences; 747 utterances of the voice's own data; phoneme agreement {'sr': {'identical': 538, 'of': 747, 'share': 0.7202}, 'bs': {'identical': 32, 'of': 747, 'share': 0.0428}}.

| voice | word error | wrong / words | vs before | p | reading |
|---|---|---|---|---|---|
| before (the checkpoint, speaker 0, as fetched) | 22.3% | 726 / 3256 | | | |
| arm A (sr), 3000 steps | 26.6% | 865 / 3256 | +4.27 | 0.0000 | **sound** |
| arm B (bs), 3000 steps | 26.0% | 846 / 3256 | +3.69 | 0.0000 | **sound** |
| arm C (bs, gentle), 3000 steps | 25.5% | 831 / 3256 | +3.22 | 0.0000 | **sound** |
| human recordings | 11.7% | 456 / 3901 | | | |

Reading rule (pre-registered): sound if the arm is under +5 points from the before voice; BROKEN if +10 or more at p < 0.05; inconclusive between.

