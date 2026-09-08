# Attribution notice

This repository redistributes pretrained weights from the projects below. Each is used
under its own license; the license text of each is at the linked source.

---

**translator/** — OPUS-MT `opus-mt-tc-big-zls-en`
Copyright © the Language Technology Research Group at the University of Helsinki
(Helsinki-NLP) and the OPUS-MT project.
License: Creative Commons Attribution 4.0 International (CC-BY-4.0)
<https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-zls-en>
<https://creativecommons.org/licenses/by/4.0/>
The authors request citation of Tiedemann & Thottingal (2020) and Tiedemann (2020); both
are cited in README.md.

---

**listen/** — OpenAI Whisper large-v3, fine-tuned by this project and converted to
CTranslate2 int8 by this project with the CTranslate2 converter.
Copyright © OpenAI (original Whisper model).
License: MIT
<https://huggingface.co/openai/whisper-large-v3>
<https://github.com/openai/whisper>
(Until 8 September 2026 this folder held SYSTRAN's `faster-whisper-small`
conversion, MIT, <https://huggingface.co/Systran/faster-whisper-small>, fine-tuned
here; it remains the baseline the published numbers compare against.)

---

**speak/** — Kokoro-82M
Copyright © hexgrad.
License: Apache License 2.0
<https://huggingface.co/hexgrad/Kokoro-82M>
<https://www.apache.org/licenses/LICENSE-2.0>
Modification notice, as Apache-2.0 asks: no weights were altered. Two files were renamed
— `kokoro-v1_0.pth` to `model.pth`, and `voices/af_heart.pt` to `voices/default.pt` — and
only the `af_heart` voice is included.
The upstream card credits CC-BY-licensed training audio, including Koniwa and SIWIS.

---

**read/** — EasyOCR detection and recognition weights
Copyright © JaidedAI. Detector: CRAFT, copyright © Clova AI Research, NAVER Corp.
License: Apache License 2.0 (EasyOCR); MIT (CRAFT)
<https://github.com/JaidedAI/EasyOCR>
<https://github.com/clovaai/CRAFT-pytorch>

**read/ — the engine the app reads with since 5 September 2026** — PaddleOCR PP-OCRv6
(`PP-OCRv6_medium_det`, `PP-OCRv6_medium_rec`)
Copyright © PaddlePaddle Authors, Baidu.
License: Apache License 2.0
<https://github.com/PaddlePaddle/PaddleOCR>
<https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det>
<https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec>
These weights are **not redistributed in this repository**. PaddleX fetches them at run
time from the Hugging Face mirror above. They are untrained by this project and are
listed here because the application reads with them; `read/` remains the fallback.

---

No weights in this repository were trained by the Lilly project. What Lilly adds is the
selection, the folder layout, and the application at
<https://github.com/ssaaffaakk/Lilly>.

---

# What these models were trained on

Attribution does not stop at the models: some of the data they were trained on carries
its own attribution requirement. Passed on here as those licenses ask.

**translator/** — trained on `opusTCv20210807+bt`, built by the Tatoeba Challenge project
from the OPUS corpus collections.
<https://github.com/Helsinki-NLP/Tatoeba-Challenge> · <https://opus.nlpl.eu/>
Evaluated on the Tatoeba test set (v2021-08-07) and FLORES-101 devtest.

**listen/** — Whisper was trained on 680,000 hours of audio and transcripts collected
from the internet (65% English, 18% non-English audio with English transcripts, 17%
non-English in 98 languages). OpenAI credits no named datasets for it.

**speak/** — Kokoro was trained only on permissive and non-copyrighted audio: public
domain recordings, audio under Apache/MIT-style licenses, synthetic audio from closed TTS
systems, and two datasets released under Creative Commons Attribution, which are credited
here because those licenses require it:

- **Koniwa** — the Koniwa project — CC BY 3.0 — <https://github.com/koniwa/koniwa>
  (under 1 hour used)
- **SIWIS** — the University of Edinburgh — CC BY 4.0 —
  <https://datashare.ed.ac.uk/handle/10283/2353> (under 11 hours used)

**read/** — the CRAFT detector was trained on SynthText, IC13 and IC17 by Clova AI
Research; see Baek et al., "Character Region Awareness for Text Detection", CVPR 2019
(<https://arxiv.org/abs/1904.01941>). Its training code was not released. The training
data for the recognition model shipped by EasyOCR is not documented publicly.
