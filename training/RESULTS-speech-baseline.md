# Does the listener's training help at all

200 held-out FLEURS Bosnian clips, 3,901 words, both models decoded the same way.

| | word error rate |
|---|---|
| whisper-small, untouched | **38.5%** |
| Lilly, fine-tuned on 3,091 Bosnian clips | **35.5%** |
| gap | **3.0 points** |

This is the comparison the project had never made. The listener's 35.5% was
known; what an untrained whisper-small does on the same clips was not, so there
was no way to say whether the fine-tuning had achieved anything or whether we
were reporting the checkpoint's own ability as our own.

Three points is a real gain and a small one. `training/RUBRIC.md` treats
anything within 2.5 points of the untrained checkpoint as indistinguishable from
doing nothing, so this clears that bar with little to spare, and it clears it on
200 clips rather than the 925 the rubric specifies — the full set is the number
to publish and this is the number to plan with.

It also sets tonight's target honestly. Beating 35.5% is not the bar; beating it
by enough to matter is, and the pre-registered thresholds require the Bosnian
term measure not to fall while it happens.
