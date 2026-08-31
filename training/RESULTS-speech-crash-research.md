# Why speech Kaggle runs die, and how half 1 / half 2 should actually work

*Generated: 2026-08-31 | Sources: 18 web + Lilly run logs | Confidence: High on crash taxonomy, Medium on unpublished API rate numbers*

Firecrawl and Exa MCP were not available in this session. Searches used Kaggle docs, Hugging Face docs, Kaggle CLI GitHub, Stack Overflow, and this repo’s own RESULTS.

## Executive summary

The jobs are not “randomly crashing.” Three different failures have been repeating, and splitting into two epochs only helps if we also stop doing the things that throw the weights away after train finishes.

1. **12-hour Save & Run All wall** — official Kaggle limit. Two-epoch large-v3 + download + BEFORE/AFTER WER is longer than 12h. The kernel is **cancelled**, not ERROR.
2. **Cancel deletes unsaved Output** — a zip listed on the worker 404s after cancel. `/kaggle/working` is not a backup until the version **completes**.
3. **429 on `kernels output`** — listing thousands of clone/audio files hits an unpublished rate limit at the exact moment we need the zip.

Half 1 (1 epoch) that is **already running** still clones into `/kaggle/working` and still runs BEFORE WER + convert + AFTER after train. That is the same shape as last night, only shorter. **Do not push half 2 until the notebook clones to `/kaggle/temp`, zips only a small adapter (or one zip), and the watcher backs off on 429.**

## 1. These are not mystery crashes — we already measured them

| When | What the API said | What actually happened | Source |
|---|---|---|---|
| 29 Aug (v14) | ERROR | Train finished. Notebook asserted `model.safetensors`. large-v3 writes **shards**. Convert/zip never ran. | `overnight-v2.report.md` |
| 29–30 Aug (v16) | CANCEL after ~12h | Train 8h 46m (1,528 steps). Zip written. AFTER WER started. Session died. Zip listing **404**. | `RESULTS-speech-v16.md` |
| 30–31 Aug (v2 on safaksideacc2) | CANCEL + 429 | Train ~63% at 5h 39m remaining 3h 17m. Wall at ~12h. Watcher 429 × 8. Keep folder empty. | live log + `speech-fetch.report.md` |

Kaggle’s own docs: GPU/CPU **Save & Run All must finish in 12 hours** (9h TPU). Interactive idle is 20 minutes; we are on batch Save & Run All, so the 12h wall is the one that hits. ([Kaggle notebooks docs](https://www.kaggle.com/docs/notebooks))

Weekly GPU quota is separately ~30 hours and resets; a 12h session can still die with quota left. ([Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage))

## 2. Why the zip 404s after cancel

`/kaggle/working` is the only tree that becomes Output. It is **session storage**. If the session is cancelled, times out, or never completes, those files are gone even if the UI still names them. ([Stack Overflow, models lost from `/kaggle/working`](https://stackoverflow.com/questions/76554239/cannot-download-saved-ai-models-in-kaggle-working-dir-and-models-were-lost); [fast.ai: persist by committing, then add kernel output as a dataset](https://forums.fast.ai/t/is-there-a-way-to-save-persistently-intermediate-results-in-kaggle-kernels/44545))

v16 matches that: Output still listed `lilly-listen-trained.zip`; download 404. Cancel dropped the worker disk.

Third-party tools exist because this is a known hole (kgout syncs `/kaggle/working` to Drive while the session is alive). We should not add Google Drive to Lilly without asking. The in-house equivalent is: **zip small, zip early, version must COMPLETE, download before the wall.**

Audio already goes to `/kaggle/temp` (correct). The **git clone is still `/kaggle/working/Lilly`**. Everything in working is Output. OCR already encodes this: clone to `/kaggle/temp` or the weights zip never downloads. Speech notebook does not.

That clone flood is why `kernels output` paginates forever and 429s. ([kaggle-cli#1045](https://github.com/Kaggle/kaggle-cli/issues/1045): many output files, pattern match can miss later pages; listing is expensive.)

## 3. Why 429 killed the download

Kaggle does **not publish** numeric API limits. Staff: sleep after 429; API vs web have different limits; downloading many files is worse than one archive. ([kaggle-api#132](https://github.com/Kaggle/kaggle-api/issues/132))

CLI is supposed to honor `Retry-After`. Our watcher called `kernels output` every 2 minutes for hours, then 8 retries in ~13 minutes at cancel. That is the opposite of backoff. The 429 URL was `ListKernelSessionOutput`.

## 4. Half 1 / half 2 done wrong vs done right

**Wrong (what “split the file and download twice” looks like if we only change `SPEECH_EPOCHS`):**

- Half 1: still clone into working, still 3.5h download, still BEFORE WER on large-v3, 1 epoch (~4.5h), then convert + AFTER. May fit 12h. Zip is still ~3 GB merged large-v3 (`merge_and_unload` in `train_speech.py`). Cancel still 404s a 3 GB file.
- Half 2: `--base` that merged folder starts a **new LoRA** on already-merged weights. That is **not** epoch 2 of the same trainer run. Optimizer and LR schedule restart. ([HF PEFT: resume is `trainer.train(resume_from_checkpoint=...)` on a Trainer checkpoint with `trainer_state.json` / optimizer, not adapter-only or merged weights](https://huggingface.co/docs/transformers/en/peft); [forum: adapter file ≠ full resume](https://discuss.huggingface.co/t/peft-with-sfttrainer-unexpected-resume-from-checkpoint/169973))

**Right:**

| | Half 1 | Half 2 |
|---|---|---|
| Goal | Finish **inside** 12h with a **downloadable** artifact | True second epoch **or** a documented new LoRA |
| Clone | `/kaggle/temp` (OCR already does this) | same |
| `/kaggle/working` | Only `lilly-listen-trained.zip` (and maybe a small adapter zip) | same |
| Train save | Prefer **unmerged** `checkpoints-speech/checkpoint-*` (adapter, tens of MB) **and** zip it the moment epoch 1 ends | `resume_from_checkpoint` on that directory |
| AFTER WER | **After** zip, or skip until local | After zip |
| Audio | `/kaggle/temp` | Re-download **or** attach a dataset; do not copy wavs into working |
| BEFORE WER | Optional; costs a large-v3 convert | Skip (same base, wasted hour) |
| Download | One `--file-pattern` for the zip; **stop polling** `kernels output` until status COMPLETE or the zip name appears; on 429 sleep `Retry-After` | same |
| Success condition | Kernel **COMPLETE** + local zip > 1 MB | same |

Splitting **data** in half and training 2 epochs on each half is a different experiment (half the mix, not two epochs of the full mix). Do not do that unless we rename the claim.

## 5. Mistakes to stop repeating (ours, not Kaggle’s)

1. Treating `kernels status RUNNING` as “not done” and `CANCEL` as “try 8 full outputs immediately.”
2. Cloning Lilly into `/kaggle/working` after we already learned that kills OCR downloads.
3. Asserting a single `model.safetensors` (fixed) then still packaging a 3 GB merged model that cancel cannot keep.
4. AFTER WER / convert **after** the wall with no COMPLETE.
5. Watcher + logs stream + `kernels files` + `kernels output` in the same window → 429 at the only minute the zip exists.
6. Pushing a recipe change so Kaggle can clone it, then launching, **before** fixing persistence. (Half 1 v3 is already running.)
7. `merge_and_unload` then calling the next session “epoch 2.”

## Key takeaways (do this before any more push)

1. **Do not push half 2 yet.** Fix clone path + zip-only working dir + resume vs merge, then push.
2. **Half 1 currently running** can still lose the zip if AFTER overruns or Output is flooded. Watcher must use 5–10 min polls and `--file-pattern`, never full output until one zip exists.
3. **A complete 1-epoch run with a small adapter zip is more valuable than a 2-epoch run that 404s.** Adapter checkpoints are what HF actually resumes.
4. **COMPLETE is the persistence event.** Cancel is data loss. Budget so train + zip finish with hours to spare; AFTER WER is optional on Kaggle.
5. **429 is expected** if we list a git clone. Stop listing. One zip file.

## Sources

1. [Kaggle notebooks docs](https://www.kaggle.com/docs/notebooks) — 12h Save & Run All; 20 min interactive idle.
2. [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage) — weekly GPU quota (~30h), separate from session cap.
3. [Kaggle product feedback: kernels disconnect, use Save & Run All](https://www.kaggle.com/product-feedback/83678)
4. [SO: `/kaggle/working` models vanished](https://stackoverflow.com/questions/76554239/cannot-download-saved-ai-models-in-kaggle-working-dir-and-models-were-lost)
5. [fast.ai: persist via commit, add kernel output as dataset](https://forums.fast.ai/t/is-there-a-way-to-save-persistently-intermediate-results-in-kaggle-kernels/44545)
6. [kaggle-api#132](https://github.com/Kaggle/kaggle-api/issues/132) — unpublished 429 limits; sleep and retry.
7. [kaggle-cli#1045](https://github.com/Kaggle/kaggle-cli/issues/1045) — output listing pagination / too many files.
8. [kaggle-cli kernels output docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) — `--file-pattern`, page size.
9. [HF PEFT + Trainer resume](https://huggingface.co/docs/transformers/en/peft)
10. [HF forum: resume_from_checkpoint vs adapter-only](https://discuss.huggingface.co/t/peft-with-sfttrainer-unexpected-resume-from-checkpoint/169973)
11. [kgout](https://github.com/vybhav72954/kgout) — community workaround: sync working dir off-box before timeout.
12. Lilly `overnight-v2.report.md`, `RESULTS-speech-v16.md`, `logs/kaggle/speech-fetch.report.md`, `training/train_speech.py` (`merge_and_unload`), `Lilly_Speech_Kaggle.ipynb` (clone to `/kaggle/working`), `preflight_kaggle.py` (OCR clone-to-temp rule).

## Methodology

Queries: Kaggle 12h cancel/output 404, API 429, PEFT resume, persist mid-run, GPU quota. ~20 unique URLs considered; 12 cited. Sub-questions: wall-clock vs cancel; why 404; why 429; how to split epochs correctly; how to persist before timeout.

Gaps: exact 429 quota unpublished; whether batch Save & Run All copies `/kaggle/working` incrementally while RUNNING is not clearly specified in official docs (community says interactive Output updates live; batch persistence is on COMPLETE).
