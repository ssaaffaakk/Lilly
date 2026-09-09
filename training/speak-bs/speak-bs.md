# A Bosnian voice from FLEURS -- the judgment

Kaggle, Tesla T4. Pre-registered: PREREGISTRATION.md, 'v5 -- speak -- a Bosnian voice from FLEURS'.
Listener e6bb58483586b06c; test prefix: 200 clips, 167 sentences.
Trained 32962 steps / 47 epochs on 2985 clips, 7 speakers, 592.5 minutes; served speaker 2 (cluster 6).

| voice | word error | wrong / words |
|---|---|---|
| before (Piper sr_RS, as the app ships it) | 22.3% | 726 / 3256 |
| candidate (this run) | 53.9% | 1754 / 3256 |
| human recordings | 11.7% | 456 / 3901 |

candidate vs before: +31.57 points, paired bootstrap over sentences p = 0.0000.

**Bar 1 (ships): strictly below the before voice at p < 0.05 -- FAIL.**
**Bar 2 (the owner's ask, reported, not required): at or below the human recordings -- not reached.**
