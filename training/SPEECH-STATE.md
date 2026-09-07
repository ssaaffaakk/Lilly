# speech state
Mon Sep  7 14:43:30 CEST 2026

## listen/model.bin
-rw-r--r--@ 1 safaksurmeli  staff   1.5G Sep  1 08:59 models/lilly/listen/model.bin

## built.json
{"base": "openai/whisper-large-v3", "d_model": 1280, "encoder_layers": 32, "quantization": "int8"}
## listen dirs
models/lilly/listen
models/lilly/listen-candidate
models/lilly/listen-previous
models/lilly/listen.before-training

## term recall
training/RESULTS-speech-half2.md:53:| Bosnian-specific term recall | 65.9% (legacy SpeechBench) | not below baseline | **running locally** |
training/RESULTS-speech.md:47:| Bosnian-specific term recall | baseline | not below baseline | **not yet run** |
training/RESULTS-speech.md:65:| | WER | term recall | variety substitution |
training/RESULTS-speech.md:74:What the current fine-tune bought, read honestly: +5.9 points of term recall at
training/RESULTS-speech.md:80:**So the bar for the new listener is 65.9% term recall and 5.1% substitution.**
training/RESULTS-speech.md:92:| Bosnian term recall | 65.9% | **68.2%** |
training/RESULTS-speech.md:98:| Bosnian term recall | not below 65.9% | 68.2% | pass |
