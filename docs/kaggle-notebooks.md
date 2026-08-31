# How to write Lilly Kaggle notebooks

**Audience:** any agent or human editing `training/Lilly_*_Kaggle*.ipynb`,
`scripts/kaggle_train.py`, `scripts/kaggle_poll.py`, or the train/eval scripts
those notebooks call.

**One rule:** a known failure stops the run. COMPLETE, a zip, or the 12h wall
must not override the gate. Skipping a correctness gate to “save the session”
is an **illegal variant**, not an optimization
(`test-optimizasyon-dongusu` / fail-stop).

Also read:

- `docs/kaggle-fail-stop.md` — numbered list of past mistakes
- `.cursor/rules/kaggle-fail-stop.mdc` — always-apply Cursor rule
- `docs/agent-teams-lilly.md` rule 9 — no band-aids; GitHub is the store

Encode regressions in `scripts/preflight_kaggle.py`. If you change a notebook,
**update preflight in the same commit** so the next agent cannot relaunch the
old hole.

```bash
python3 scripts/preflight_kaggle.py   # must exit 0 before any launch
```

---

## 0. What “success” means (read before editing)

| Kernel status | Zip on Output | Meaning |
| :--- | :--- | :--- |
| COMPLETE + fresh shippable zip | yes | OK to fetch/install **that** artefact |
| COMPLETE + no zip / tiny zip | — | **Failure.** Kernel finished without product |
| ERROR | maybe old zip | **Failure.** Do not install. Fix cause, relaunch |
| CANCEL / CANCEL_ACKNOWLEDGED | maybe old zip | **Failure.** Recovery is not success |
| RUNNING + old leftover zip | old | Not done. Do not treat leftover as this run |

`scripts/kaggle_poll.py` must treat ERROR and CANCEL as `crashed` even when a
zip exists. `actually_done` requires `COMPLETE` **and** a fresh large zip.

Never say “we got the zip from a cancelled run, good enough.”

---

## 1. Skeleton every notebook must have

Copy this shape. Do not invent a softer variant.

```text
Cell A  Markdown: job name, pass id, required attaches, what is NOT done here
Cell 1  GPU assert + Internet + def run(*cmd) with check=True
Cell 2  Clone to /kaggle/temp (never /kaggle/working)
Cell 3  pip from requirements.txt pins (OCR: never pip-install torch)
Cell 4  Short CUDA / LoRA / train smoke (~20–60s)
Cell …  Required data: attach or download; missing → SystemExit
Cell …  Train via run(...) / check=True
Cell …  Measure / install gate BEFORE any shippable zip
Cell …  Zip only the intended artefact into /kaggle/working
Last    Markdown: ERROR → do not install; fix + relaunch
```

### Required `run` helper (every notebook)

```python
def run(*cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)
```

### Illegal soft wrappers (do not write these)

```python
# BAD — train fails, notebook continues, later cells zip garbage
r = subprocess.run([...], check=False)
# ... zip anyway ...
assert r.returncode == 0   # too late; zip already in Output

# BAD — “inspection” zip of refused weights
if train_failed:
    zip_refused_weights_to_working()
    print("NOT SHIPPABLE but COMPLETE")

# BAD — missing data becomes a smaller run
if not harvest:
    print("training synthetic only")
```

Kaggle marks a version COMPLETE when no cell **raises**. A shell failure that
you ignore is still COMPLETE. That is why every failure must be `check=True`,
`assert`, or `raise SystemExit`.

---

## 2. Clone and Output discipline

**Fact:** everything under `/kaggle/working` becomes version Output.
`kaggle kernels output` then walks that tree. A git clone + thousands of PNGs
there made the weights zip unreachable (OCR burned 172s on PNGs/git and never
fetched the product).

```python
# GOOD
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
CLONE = SCRATCH / "Lilly"
subprocess.run(["rm", "-rf", str(CLONE)], check=True)
os.chdir(SCRATCH)
run("git", "clone", "-q", "https://github.com/ssaaffaakk/Lilly.git")
os.chdir(CLONE)

# BAD — floods Output; zip never downloads
os.chdir("/kaggle/working")
run("git", "clone", ...)
```

Audio, crops, and synthetic images go under `/kaggle/temp/...` (symlink from
repo paths if needed). **Only** the intended zip(s) belong in `/kaggle/working`.

Assert Output stay small after zipping (OCR already does `len(out) < 50`).

---

## 3. GPU and Internet (fail in ten seconds, not ten hours)

```python
import torch
assert torch.cuda.is_available(), (
    "No GPU. Right panel -> Session options -> Accelerator -> GPU, then Save & Run All again.")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU; two cards halves steps at same LR
```

Launcher must set `enable_gpu: true` and a real `machine_shape` from Kaggle’s
enum (`scripts/kaggle_train.py` + `confirm_accelerator`). Empty/wrong shape
lands on CPU or an unsupported P100; the notebook’s assert is the last line of
defence.

Reach GitHub, PyPI, Hugging Face before downloads. Speech also needs
`datasets-server.huggingface.co`.

Smoke before heavy work:

- Speech: build a tiny LoRA, `.cuda()`, backward, assert LoRA grads.
- OCR: tiny matmul backward on CUDA, then `train_ocr.py --quick-test`.

---

## 4. GitHub is the store

Kaggle clones `https://github.com/ssaaffaakk/Lilly.git`. An unpushed local fix
is invisible on the box.

| Do | Don't |
| :--- | :--- |
| Commit + push notebook/scripts after the fix | “Local only, don’t commit” |
| Launch only on a clean tree matching `origin/main` | Launch dirty / unpushed HEAD |
| Never commit `.env`, `kaggle.json`, tokens | Paste secrets into cells |

`kaggle_train.py` refuses launch if GitHub does not have the current SHA.

---

## 5. Speech — what changed and how to write it

### Why two halves exist

One session with **2 epochs + BEFORE WER + merge ~3 GB + AFTER WER** hit the
**12h Save & Run All wall** twice. CANCEL → zip 404 even when Output still
named a file.

Legal split (wall-clock), **illegal** to skip measurement on the ship half:

| Half | Notebook | Does | Does not |
| :--- | :--- | :--- | :--- |
| 1 | `Lilly_Speech_Kaggle.ipynb` | 1 epoch, zip Trainer adapter | WER, merge, convert, app zip |
| 2 | `Lilly_Speech_Kaggle_Half2.ipynb` | `--resume` epoch 2, convert, AFTER WER, app zip | Start a new LoRA on merged weights |

### Hole → now (speech)

| Hole (do not revive) | Now |
| :--- | :--- |
| 2 epochs + WER in one session | Half 1 / half 2 split |
| Clone `/kaggle/working` | Clone `/kaggle/temp` |
| Zip merged large-v3 (~3 GB) as the product | Half 1: `--keep-adapter` → `lilly-listen-half1.zip` |
| `--base models/lilly/listen-trained` called “epoch 2” | Half 2: `--resume` + `SPEECH_EPOCHS = 2` |
| Encoder LoRA 0-grad → warn, continue | `EncoderGradientCheck` → `SystemExit` |
| NaN/Inf loss filtered from logs | `FiniteLossCheck` + `logging_nan_inf_filter=False` |
| Skip AFTER WER to COMPLETE | Half 2: WER **then** `lilly-listen.zip` |
| Failed parquet/shard → skip, `return 0` | Download scripts exit 1 / refuse partial |
| CANCEL + old zip = done | `kaggle_poll.py` → crash |

### Half 1 cell contract

1. Setup (`run` + GPU + net)
2. Clone `/kaggle/temp`
3. pip pins from `requirements.txt` (`transformers`, `accelerate`, `peft`, …)
4. LoRA CUDA smoke
5. `download_speech_data.py` — assert each split &gt; 100 rows
6. `download_extra_speech.py` fleurs_hr + **voxpopuli_hr** — assert both landed
7. `build_speech_mix.py --share 0.47`, `SPEECH_EPOCHS = 1`, row-epoch budget ≤ 26 000
8. Train:

```python
run("python3", "training/train_speech.py",
    "--data", MIX, "--base", "openai/whisper-large-v3",
    "--epochs", "1", "--batch-size", "1", "--grad-accum", "16",
    "--keep-adapter", "--no-convert")
assert (Path("models/lilly/listen-adapter") / "trainer_state.json").is_file()
run("zip", "-qr", "/kaggle/working/lilly-listen-half1.zip",
    "models/lilly/listen-adapter")
```

**Half 1 must not contain:** `evaluate_speech.py`, `listen-trained`,
`lilly-listen.zip`, `merge_and_unload` as the Output product.

Markdown must say: this zip is **not** the app listener; half 2 resumes it.

### Half 2 cell contract

Same setup/clone/pip/smoke/download/mix as half 1, then:

- `SPEECH_EPOCHS = 2` (Trainer continues epoch 2; `= 1` would train nothing)
- Find `/kaggle/input/**/lilly-listen-half1.zip`, unzip, require `trainer_state.json`
- Train with `--resume <adapter_dir>` and `--no-convert` as designed in
  `train_speech.py` (do **not** pass `--base` pointing at merged weights)
- Convert to `models/lilly/listen`
- **Order is mandatory:**

```python
run("python3", "training/evaluate_speech.py",
    "--data", "data/speech/test.tsv",
    "--model", "models/lilly/listen", "--limit", "200", "--show", "3")
run("zip", "-qr", "/kaggle/working/lilly-listen.zip", "models/lilly/listen")
```

If evaluate raises, there is **no** `lilly-listen.zip`. That is correct.

### Speech anti-patterns (agents keep inventing these)

```text
✗ Put WER back into half 1 “just to have a number”
✗ Merge in half 1 “so we have something to download if half 2 fails”
✗ Half 2: --base listen-trained with a new LoRA and call it epoch 2
✗ SPEECH_EPOCHS = 1 on half 2
✗ Zip lilly-listen.zip before evaluate_speech
✗ Launch speech-half2 before half 1 COMPLETE + fresh half1 zip
✗ Launch while 2/2 GPU sessions already RUNNING
✗ Treat CANCEL_ACKNOWLEDGED + leftover zip as success
```

Launch:

```bash
python3 scripts/kaggle_train.py speech          # half 1
# wait COMPLETE + lilly-listen-half1.zip
python3 scripts/kaggle_train.py speech-half2    # only then
```

---

## 6. OCR — what changed and how to write it

### Hole → now (OCR)

| Hole (do not revive) | Now |
| :--- | :--- |
| `push_ocr_harvest` returns None → still launch | `kaggle_train.py`: `hv is None` → `SystemExit` |
| `print("no harvest — synthetic only")` | `raise SystemExit` — harvest required for pass-7c |
| `check=False`, zip refused weights, then assert 0 | `run(..., check=True)` only |
| `--keep-trained` written, then gate `return 1` | Write keep-trained **only after** gate passes |
| Too-short run exit 0 | `steps < MIN_STEPS` → return 1 |
| CANCEL/ERROR + zip = success | poll: always crash |
| Photo read fail → `continue`, still print % | `evaluate_ocr.py`: hole → stop |
| Synthetic before real crops (prepare wipes syn*) | Real merge **first**, then generate |
| Unfiltered EasyOCR auto-crops drown human labels | `AUTO_MIN_CONF`, `AUTO_CAP_MULT`; `auto*` train-only |
| Buffered stdout hid gate refusal | `python -u` + `PYTHONUNBUFFERED` |
| Markdown “attach harvest if present” while code requires it | Markdown must say **required** |

### OCR data order (wrong order already shipped a 2‑minute run)

```text
1. Copy real crops from lilly-ocr-crops (skip paths containing "harvest")
2. prepare_ocr_data.py --labels <human labels>     # MUST succeed
3. generate_ocr_data.py --count 50000 --build-splits
4. generate_ocr_photos.py --count 20000
5. Harvest scenes → auto* train rows only (not valid)
6. Oversample human rows REAL_REPEAT = 2
7. train_ocr.py --quick-test
8. train_ocr.py heavy (5 epochs, gate inside)
9. Zip only if train exited 0 and files exist
```

If step 2 fails → `SystemExit`. Never “continue with synthetic only.”

### Harvest cell (required for pass-7c)

```python
# GOOD — fail closed
if n_hv >= 20:
    # filter conf, cap at human_n * AUTO_CAP_MULT, append auto_* to train only
    ...
else:
    raise SystemExit(
        "Harvest photos are required for this pass (pass-7c). "
        "Attach lilly-ocr-harvest or relaunch: python3 scripts/kaggle_train.py ocr")

# BAD
else:
    print("no harvest photos attached — training on human crops + synthetic only")
```

Assign the filter knobs in the same cell (substring-only checks once shipped a
notebook that NameError’d on `AUTO_CAP_MULT`):

```python
AUTO_MIN_CONF = 0.7
AUTO_CAP_MULT = 2
HARVEST_EXT = {".jpg", ".jpeg", ".png", ".webp"}  # Commons is mostly JPEG
```

Auto-crops must **not** enter valid. Valid gate would score EasyOCR against
itself.

### Heavy train + zip

```python
run("python3", "-u", "training/train_ocr.py",
    "--epochs", "5", "--batch-size", "16",
    "--lr", "3e-6", "--grad-clip", "1.0", "--warmup-frac", "0.05",
    "--weights", str(INIT), "--keep-trained", str(TRAINED))
assert TRAINED.is_file() and TRAINED.stat().st_size > 100_000
run("zip", "-j", "/kaggle/working/lilly-read-trained.zip", str(TRAINED))
# app package only after gate installed lilly.pth
run("zip", "-qr", "/kaggle/working/lilly-read.zip",
    "models/lilly/read/lilly.pth",
    "models/lilly/read/user_network/lilly.yaml",
    "models/lilly/read/user_network/lilly.py")
```

**Do not** copy starting `latin_g2.pth` into `lilly.pth` and call that trained.
The app loads `recog_network="lilly"` → `lilly.pth`.

### On-box vs local measurement

| Where | What |
| :--- | :--- |
| Kaggle notebook | Install gate inside `train_ocr.py` (real words up, syn no-regression, min steps) |
| Local / later | `evaluate_ocr.py` + `truth.json` + `data/ocr/real-photos/scored/` |

Do **not** add `evaluate_ocr.py` to the Kaggle notebook unless those files are
attached. It would false-fail every GPU run or tempt “skip failed photos.”

Local eval: a failed photo read is a failed score — no `continue` that leaves
holes.

### OCR anti-patterns

```text
✗ Launch OCR when push_ocr_harvest returned None
✗ check=False on train_ocr “so we can still zip for debugging”
✗ Zip then assert returncode
✗ Write read-trained.pth before the gate, refuse, leave file looking shippable
✗ Train without lilly-ocr-crops / without harvest on a harvest pass
✗ Glob only *.png for harvest (Commons is JPEG)
✗ Put auto* rows into valid
✗ pip install torch from requirements (CPU wheels; breaks CUDA)
✗ Clone into /kaggle/working
✗ COMPLETE without lilly-read.zip → still “success” in the poller
```

Launch:

```bash
python3 scripts/kaggle_train.py ocr
```

Attaches: `lilly-read-pass1`, `lilly-ocr-crops`, `lilly-ocr-harvest` (all required
for this pass).

Artifacts:

- `lilly-read.zip` — **install this** into the app
- `lilly-read-trained.zip` — weights after a **passed** gate; not a substitute
  for the app zip layout

---

## 7. Eval contract (promotion gates)

| Notebook | Correctness gate (must stay green) | Shippable Output |
| :--- | :--- | :--- |
| Speech half 1 | `train_speech` exit 0 + `trainer_state.json` + zip &gt; 1 MB | `lilly-listen-half1.zip` |
| Speech half 2 | train 0 + convert + `evaluate_speech` 0 | `lilly-listen.zip` |
| OCR | `train_ocr` install gate pass + `lilly.pth` present | `lilly-read.zip` |

Illegal “optimizations”:

- Skip WER / gate / harvest to beat 12h
- Smaller corpus after a failed download
- Warn-and-continue on 0-grad / NaN
- Ship unmeasured weights because the wall is near

---

## 8. Markdown must match code

Agents read the header and “optimize” against it.

| If code does this | Markdown must say |
| :--- | :--- |
| Harvest required | Attach `lilly-ocr-harvest` (**required**), not “if present” |
| Half 1 has no WER | No BEFORE/AFTER WER here; half 2 scores |
| ERROR / non-zero train | Do not install Output; fix and relaunch |
| Product is adapter zip | Not the app listener |

Fix doc drift in the **same** change as the code.

---

## 9. Launcher and poller rules

### `scripts/kaggle_train.py`

- Clean tree; GitHub has HEAD SHA; preflight 0
- OCR: `needs_ocr_harvest` → if `push_ocr_harvest` is None, **exit**, do not push notebook
- `confirm_push`: stdout must contain “successfully pushed” (GPU limit returns 0 with a refusal message — that is not a launch)
- `confirm_accelerator`: pull metadata; refuse wrong/missing GPU shape
- Max **2** batch GPU sessions. Poll before launch. Do not queue a third.

### `scripts/kaggle_poll.py`

```text
actually_done = (status == COMPLETE) and fresh_zip_present
crashed = status in (ERROR, CANCEL) or (COMPLETE and no zip)
```

CANCEL + zip ⇒ crashed. COMPLETE + no zip ⇒ crashed.

---

## 10. Preflight is part of the notebook change

When you edit a notebook, add/adjust checks in `scripts/preflight_kaggle.py`
so the forbidden string or missing required string fails closed.

Examples already enforced:

- No `/kaggle/working/Lilly` clone
- Speech half 1: `--keep-adapter`, no `evaluate_speech`
- Speech half 2: `--resume`, WER before zip path
- OCR: no `check=False`, no synthetic-only fallback string, harvest
  `SystemExit`, `AUTO_* =` assignments, `hv is None` guard in launcher
- `train_speech`: Encoder `SystemExit`, `FiniteLossCheck`

If preflight cannot express a rule, put it here and in the Cursor rule anyway —
then add a check when possible.

---

## 11. Checklist before you push a notebook change

```text
[ ] preflight exits 0
[ ] run() uses check=True everywhere train/data runs
[ ] Clone path is /kaggle/temp
[ ] Required datasets stated as required in markdown
[ ] Measure/gate cell index < shippable zip cell index
[ ] No zip of refused / unmeasured weights
[ ] Speech: half role clear (adapter vs app)
[ ] OCR: real before syn; harvest fail-closed; -u on heavy train
[ ] Committed and pushed (no secrets)
[ ] GPU slots free before launch (≤1 other batch GPU running if you need one slot)
```

---

## 12. Related files

| File | Role |
| :--- | :--- |
| `training/Lilly_Speech_Kaggle.ipynb` | Speech half 1 |
| `training/Lilly_Speech_Kaggle_Half2.ipynb` | Speech half 2 |
| `training/Lilly_OCR_Kaggle.ipynb` | OCR pass-7c |
| `scripts/preflight_kaggle.py` | Launch gate |
| `scripts/kaggle_train.py` | Push + harvest required |
| `scripts/kaggle_poll.py` | CANCEL/ERROR = crash |
| `training/train_speech.py` | `--keep-adapter`, `--resume`, 0-grad / NaN stop |
| `training/train_ocr.py` | Install gate; keep-trained after pass |
| `training/evaluate_speech.py` | AFTER WER (half 2 only on Kaggle) |
| `training/evaluate_ocr.py` | Local photo score; no holes |
| `docs/kaggle-fail-stop.md` | Short mistake list |
| `.cursor/rules/kaggle-fail-stop.mdc` | Always-apply summary |
