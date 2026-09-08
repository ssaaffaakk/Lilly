#!/usr/bin/env python3
"""Generate the Kaggle notebook for the ordinal-splitter re-measurement.

    python3 training/build_ordinals_notebook.py

Writes training/Lilly_Ordinals_Kaggle.ipynb and the staging copy. Every code
cell is ast.parse-checked before it is written (do-not-repeat #8: notebooks
are built, not hand-edited as JSON).

Pre-registered: training/PREREGISTRATION.md, "v4 -- translate -- ordinals".
The Mac was where this measurement lived; the owner moved it to Kaggle on
8 September 2026 (heavy compute goes to Kaggle, .claude/CLAUDE.md). Because the
old translations were made on the Mac's CPU, the notebook runs BOTH splitters
on the same box, so the splitter effect is measured with nothing else moving,
and reports this box against the Mac as a separate device-drift row.
"""
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "training" / "Lilly_Ordinals_Kaggle.ipynb"
STAGING = REPO_ROOT / "models" / "kaggle-staging" / "lilly-ordinals-remeasure"

# The two builds every served-path number compares. The scored fine-tune is
# named by training/RESULTS-product.md ("Scored build"); the base build is the
# Mac's models/lilly/translator-base, the untuned int8 build the fine-tune is
# measured against. Both are uploaded by scripts/kaggle_train.py and checked
# here by the same blake2b digest evaluate_app.py and publish_to_hf.py use.
SCORED_BUILD = "1aedcc11231cdf50817ff12f99ff0d1e"
BASE_BUILD = "348a984c324510cee218dfce8a7228e8"
# app/translate.py's SENTENCE_BREAK before commit 6c12bfc (8 September 2026):
# a period after a digit ended a sentence. Kept here, and only here, so the
# old rule can be measured beside the new one on the same machine.
OLD_RULE = r"(?<=[.!?])\s+|\s*\n+\s*"

MARKDOWN_TOP = """\
# Lilly — the ordinal splitter re-measurement (both splitters, same box)

**Job:** `ordinals-remeasure`. Pre-registered in `training/PREREGISTRATION.md`,
"v4 — translate — ordinals, written before the re-measurement".

**What this run does.** Scores the two served builds — the scored fine-tune
(`training/RESULTS-product.md`, build `1aedcc11…`) and the untuned base build
(`348a984c…`) — on all 2,009 FLORES-200 pairs through `app.translate.Engine`,
**twice on this machine**: once with the sentence splitter as it was before
8 September 2026 (a period after a digit ended a sentence), once with the rule
as it is now (an ordinal is not a full stop). Then `training/compare_hypotheses.py`
runs the paired bootstrap between the two passes, per build — the only thing
that differs between them is the splitter — and once more between this
machine's old-splitter translations and the Mac's (`training/app-hypotheses-armB.json`),
which is the device drift and is reported beside the numbers, never folded in.

**What it does NOT do.** It trains nothing and ships nothing. The artefact is a
small zip of JSON and markdown. Nothing here changes `models/lilly/`.

**Attach (the launcher does it):** datasets `lilly-translator-scored` and
`lilly-translator-base`, the two builds uploaded from the Mac and checked here
by content fingerprint before a sentence is translated. Internet **On**
(GitHub, PyPI, `dl.fbaipublicfiles.com` for FLORES). Accelerator: GPU —
CTranslate2 runs int8 on it, and the device-drift row is what says how far
that is from the Mac's CPU.
"""

MARKDOWN_END = """\
**Status ERROR or CANCEL → do not use these numbers.** Recovery is not success:
a zip recovered from a cancelled run is not this run's artefact. Read
`stdout.txt` in Output, fix the cause, and relaunch — do not soften a gate so
the next version reaches COMPLETE past the same hole.

**On COMPLETE:** `scripts/kaggle_train.py ordinals-remeasure --fetch`, unzip
`lilly-ordinals.zip` beside `training/`, commit the JSON and markdown, then
apply the pre-registered rule: the re-measured served-path figures replace
42.49 / 67.69 and 42.18 / 67.47 in `README.md`, `training/RESULTS-product.md`
and the model card, whichever way they moved, with the paired interval beside
the delta — and with the device-drift row beside them, because this run is on
a T4 and the numbers it replaces were made on a CPU.
"""

CELL_SETUP = '''\
# 1. Stop here unless the machine is actually set up
# Kaggle marks a version COMPLETE whenever no cell RAISES — a shell command that
# fails is not enough. Every check below is Python and every later step is a
# checked subprocess whose output is teed into Output, so a run that produced no
# numbers cannot be mistaken for one that did.
import hashlib, json, os, shutil, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

import torch
assert torch.cuda.is_available(), (
    "No GPU. Right panel -> Session options -> Accelerator -> GPU, then Save & Run All again.")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU = torch.cuda.get_device_name(0)
print(torch.cuda.device_count(), "GPU(s) visible, using:", GPU)

def reachable(url):
    try:
        urllib.request.urlopen(url, timeout=20).close()
    except urllib.error.HTTPError:
        pass          # a status code still proves we got out
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach {url} ({exc}). Right panel -> Session options -> Internet -> On.")

for host in ("https://github.com", "https://pypi.org", "https://dl.fbaipublicfiles.com"):
    reachable(host)
print("network ok")

# The child's stdout is NOT the Kaggle log. Tee everything into Output so a run
# that printed no BLEU line cannot be mistaken for one that did.
TEE = Path("/kaggle/working/stdout.txt")
TEE.parent.mkdir(parents=True, exist_ok=True)

def run(*cmd, quiet=False, env=None):
    line = "$ " + " ".join(str(c) for c in cmd)
    print(line, flush=True)
    with TEE.open("a", encoding="utf-8") as sink:
        sink.write(line + "\\n")
        child = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 env={**os.environ, **(env or {})})
        for out in child.stdout:
            if not quiet:
                print(out, end="", flush=True)
            sink.write(out)
        code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)
'''

CELL_CLONE = '''\
# 2. Get the Lilly code — into scratch, never into Output
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
CLONE = SCRATCH / "Lilly"
subprocess.run(["rm", "-rf", str(CLONE)], check=True)
os.chdir(SCRATCH)
run("git", "clone", "-q", "https://github.com/ssaaffaakk/Lilly.git")
assert (CLONE / "training").is_dir(), "clone produced nothing"
os.chdir(CLONE)
print("working in", os.getcwd())

sys.path.insert(0, str(CLONE / "training"))
from kaggle_offload import Offload
OFF = Offload("ordinals-remeasure", os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "manual"))
OFF.hardware(GPU)
print("offload log started:", OFF.body["git"])
'''

CELL_PIP = '''\
# 3. Install what we need (~2 min). Pins from requirements.txt, not a second list.
NEEDED = ["ctranslate2", "transformers", "sentencepiece", "sacrebleu", "sacremoses"]
pins = {}
for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
    line = line.split("#")[0].strip()
    if "==" in line:
        pins[line.split("==")[0].strip().lower()] = line
print("pinned here:", [pins[n] for n in NEEDED if n in pins])
print("no pin, taking latest:", [n for n in NEEDED if n not in pins] or "none")
run(sys.executable, "-m", "pip", "install", "-q", *[pins.get(n, n) for n in NEEDED])
import ctranslate2
print("ctranslate2", ctranslate2.__version__, "| cuda devices:", ctranslate2.get_cuda_device_count())
'''

CELL_FLORES = '''\
# 4. The data, and a hard count on it
run(sys.executable, "data/scripts/download_flores.py")
sizes = {p.name: sum(1 for _ in p.open(encoding="utf-8"))
         for p in sorted((CLONE / "data" / "flores").glob("*.??"))}
print(sizes)
pairs = sizes.get("devtest.bs", 0) + sizes.get("dev.bs", 0)
if pairs != 2009:
    raise SystemExit(f"FLORES came back as {pairs} pairs, not 2009 — "
                     "the pre-registered measurement is all 2,009 and this is not it")
OFF.metric("flores_pairs", pairs, stage="data")
print("FLORES ok:", pairs, "pairs")
'''

CELL_BUILDS = f'''\
# 5. The two builds, from the attached datasets, checked by content before use
SCORED = "{SCORED_BUILD}"   # training/RESULTS-product.md, "Scored build"
BASE = "{BASE_BUILD}"       # the Mac's models/lilly/translator-base

def build_fingerprint(build: Path) -> str:
    # identical to training/evaluate_app.py and scripts/publish_to_hf.py
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(f.name for f in build.iterdir() if f.is_file()):
        if name == "built.json":
            continue
        digest.update(name.encode("utf-8"))
        with open(build / name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()

INPUT = Path("/kaggle/input")
dirs = sorted(p.parent for p in INPUT.rglob("built.json")
              if (p.parent / "model.bin").is_file() and (p.parent / "source.spm").is_file()) if INPUT.is_dir() else []
found = {{}}
for d in dirs:
    found[build_fingerprint(d)] = d
print("attached builds:", {{k: str(v) for k, v in found.items()}})
for fp, name in ((SCORED, "translator"), (BASE, "translator-base")):
    if fp not in found:
        raise SystemExit(f"{{name}} (build {{fp}}) is not attached; found {{sorted(found)}}. "
                         "Relaunch: python3 scripts/kaggle_train.py ordinals-remeasure")
    dest = CLONE / "models" / "lilly" / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(found[fp], dest, ignore=shutil.ignore_patterns("dataset-metadata.json"))
    got = build_fingerprint(dest)
    if got != fp:
        raise SystemExit(f"{{name}} copied as {{got}}, expected {{fp}}")
    print(name, got, json.loads((dest / "built.json").read_text()))
'''

CELL_SMOKE = '''\
# 6. Smoke: eight pairs through the whole path, before the real run
run(sys.executable, "training/evaluate_app.py", "--fresh", "--limit", "8",
    "--saved", "/kaggle/temp/smoke.json", "--out", "/kaggle/temp/smoke.md")
smoke = json.loads(Path("/kaggle/temp/smoke.json").read_text(encoding="utf-8"))
assert smoke["n"] == 8, smoke["n"]
assert all(h.strip() for h in smoke["base"] + smoke["lilly"]), "empty translations"
assert smoke["lilly_build"] == SCORED and smoke["base_build"] == BASE, (smoke["lilly_build"], smoke["base_build"])
print("smoke ok — the path works; those numbers decide nothing")
'''

CELL_OLD = f'''\
# 7. PASS ONE — the splitter as it was before 8 September 2026, both builds, all 2,009
# The rule is patched onto app.translate for this process only; nothing in the
# repository changes. Everything else — builds, batching, tag stripping, the
# scorer — is the code in the clone.
OLD_RULE = {OLD_RULE!r}
OLD, NEW, MAC = ("training/app-hypotheses-oldsplit-kaggle.json",
                 "training/app-hypotheses-ordinals-kaggle.json",
                 "training/app-hypotheses-armB.json")
wrapper = SCRATCH / "evaluate_app_oldsplit.py"
wrapper.write_text(
    "import re, sys\\n"
    f"sys.path.insert(0, {{str(CLONE)!r}})\\n"
    "import app.translate as translate\\n"
    f"translate.SENTENCE_BREAK = re.compile({{OLD_RULE!r}})  # the rule before 8 September 2026\\n"
    "from training import evaluate_app\\n"
    f"sys.argv = ['evaluate_app.py', '--fresh', '--saved', {{OLD!r}}, "
    f"'--out', 'training/RESULTS-product-oldsplit-kaggle.md']\\n"
    "raise SystemExit(evaluate_app.main())\\n", encoding="utf-8")
run(sys.executable, str(wrapper))
'''

CELL_NEW = '''\
# 8. PASS TWO — the splitter as it is now, both builds, all 2,009, same machine
run(sys.executable, "training/evaluate_app.py", "--fresh",
    "--saved", NEW, "--out", "training/RESULTS-product-ordinals-kaggle.md")
'''

CELL_COMPARE = '''\
# 9. The comparisons: old against new on this box (the splitter), and the Mac's
#    old against this box's old (the device), per build
for side in ("lilly", "base"):
    run(sys.executable, "training/compare_hypotheses.py", OLD, NEW,
        "--side", side, "--json", f"training/compare-ordinals-{side}.json")
    run(sys.executable, "training/compare_hypotheses.py", MAC, OLD,
        "--side", side, "--json", f"training/compare-device-drift-{side}.json")
'''

CELL_GATE = '''\
# 10. The gate, then the package. Nothing is zipped before this cell passes.
OFF.check_trainproof(TEE)
for path in (OLD, NEW):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d["n"] != 2009 or len(d["base"]) != 2009 or len(d["lilly"]) != 2009:
        raise SystemExit(f"{path}: not all 2,009 pairs ({d['n']}, {len(d['base'])}, {len(d['lilly'])})")
    if any(not h.strip() for h in d["base"] + d["lilly"]):
        raise SystemExit(f"{path}: empty translations — a hole is not a score")
    if d.get("lilly_build") != SCORED or d.get("base_build") != BASE:
        raise SystemExit(f"{path}: wrong builds behind the numbers ({d.get('lilly_build')}, {d.get('base_build')})")
reports = []
for kind in ("ordinals", "device-drift"):
    for side in ("lilly", "base"):
        p = Path(f"training/compare-{kind}-{side}.json")
        if not p.is_file():
            raise SystemExit(f"{p} was never written")
        r = json.loads(p.read_text(encoding="utf-8"))
        a = r["splits"]["all"]
        OFF.metric(f"{kind}_{side}_chrf_old", a["old"]["chrf"], stage="compare")
        OFF.metric(f"{kind}_{side}_chrf_new", a["new"]["chrf"], stage="compare")
        OFF.metric(f"{kind}_{side}_bleu_old", a["old"]["bleu"], stage="compare")
        OFF.metric(f"{kind}_{side}_bleu_new", a["new"]["bleu"], stage="compare")
        OFF.metric(f"{kind}_{side}_changed", a["changed"], stage="compare")
        reports.append(str(p))
        print(kind, side, json.dumps(a))
ZIP = Path("/kaggle/working/lilly-ordinals.zip")
run("zip", "-j", str(ZIP), OLD, NEW,
    "training/RESULTS-product-oldsplit-kaggle.md", "training/RESULTS-product-ordinals-kaggle.md",
    *reports)
assert ZIP.is_file() and ZIP.stat().st_size > 100_000, "the package did not land"
out = sorted(p.name for p in Path("/kaggle/working").iterdir())
assert len(out) < 50, out
OFF.finish("complete", [ZIP.name])
print("packaged:", out)
'''


def build():
    cells = []
    for i, (kind, source) in enumerate([
        ("markdown", MARKDOWN_TOP), ("code", CELL_SETUP), ("code", CELL_CLONE),
        ("code", CELL_PIP), ("code", CELL_FLORES), ("code", CELL_BUILDS),
        ("code", CELL_SMOKE), ("code", CELL_OLD), ("code", CELL_NEW),
        ("code", CELL_COMPARE), ("code", CELL_GATE), ("markdown", MARKDOWN_END),
    ]):
        if kind == "code":
            ast.parse(source)
        cells.append({"cell_type": kind, "id": f"cell-{i}", "metadata": {},
                      **({"execution_count": None, "outputs": []} if kind == "code" else {}),
                      "source": source.splitlines(keepends=True)})
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12.0"}},
                "nbformat": 4, "nbformat_minor": 5}
    payload = json.dumps(notebook, indent=1) + "\n"
    STAGING.mkdir(parents=True, exist_ok=True)
    for target in (OUT, STAGING / OUT.name):
        target.write_text(payload, encoding="utf-8")
        written = json.loads(target.read_text(encoding="utf-8"))
        for cell in written["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        print("verified %d cells in %s" % (len(written["cells"]), target))


if __name__ == "__main__":
    build()
