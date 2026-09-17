# Clean speech evaluation — WER by frozen legibility cohort

Manifest SHA-256: `908cb93c92c5fe262f7e6b3cb7bc43ba45a02f8ce3f36a83cc5c189ca1d50f7c`. Product decode path; 925/925 clips.

| listener | cohort | clips | errors / words | WER |
|---|---|---:|---:|---:|
| listen-previous `a76342f6ab59b382` | clean | 925 | 6663 / 18836 | 35.37% |
| listen-previous `a76342f6ab59b382` | noisy_but_understandable | 0 | 0 / 0 | — |
| listen-previous `a76342f6ab59b382` | human_unintelligible | 0 | 0 / 0 | — |
| listen-previous | **all** | 925 | 6663 / 18836 | **35.37%** |
| listen `e6bb58483586b06c` | clean | 925 | 2170 / 18836 | 11.52% |
| listen `e6bb58483586b06c` | noisy_but_understandable | 0 | 0 / 0 | — |
| listen `e6bb58483586b06c` | human_unintelligible | 0 | 0 / 0 | — |
| listen | **all** | 925 | 2170 / 18836 | **11.52%** |

The empty noisy and unintelligible rows remain explicit; this clean FLEURS split is not padded with synthetic corruption.
