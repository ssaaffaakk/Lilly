# A voice from the Croatian parliament -- the judgment

Kaggle, Tesla T4. Pre-registered: PREREGISTRATION.md, 'v6 -- speak -- a voice from ParlaSpeech-HR'.
Listener e6bb58483586b06c; test prefix: 200 clips, 167 sentences.
Trained 46818 steps / 35 epochs on 5697 clips, 5 speakers, 899.7 minutes; served voice: the mean of the 5 speakers (index 5).

| voice | word error | wrong / words |
|---|---|---|
| before (Piper sr_RS, as the app ships it) | 22.3% | 726 / 3256 |
| candidate (the mean voice, this run) | 51.9% | 1690 / 3256 |
| human recordings | 11.7% | 456 / 3901 |
| for the record: spk3 (Grmoja, Nikola) | 51.6% | 1679 / 3256 |
| for the record: spk1 (Maras, Gordan) | 57.8% | 1883 / 3256 |
| for the record: spk2 (Pernar, Ivan) | 58.0% | 1888 / 3256 |
| for the record: spk4 (Bunjac, Branimir) | 58.1% | 1891 / 3256 |
| for the record: spk0 (Bulj, Miro) | 65.7% | 2140 / 3256 |

candidate vs before: +29.61 points, paired bootstrap over sentences p = 0.0000.

**Bar 1 (ships): strictly below the before voice at p < 0.05 -- FAIL.**
**Bar 2 (the owner's ask, reported, not required): at or below the human recordings -- not reached.**
