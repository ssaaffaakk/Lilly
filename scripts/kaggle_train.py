#!/usr/bin/env python3
"""Run Phase 2 on Kaggle's GPU without opening a browser.

Kaggle's own API does everything the web page does: upload the base weights as a
dataset, push the notebook with a GPU and internet attached, run it, and fetch the
result. This machine takes about fifteen hours for the same job; a Kaggle T4 takes
two to three, and leaves the laptop alone.

    python3 scripts/kaggle_train.py translation   # the translation fine-tune
    python3 scripts/kaggle_train.py speech        # listener half 1 (adapter zip)
    python3 scripts/kaggle_train.py speech-half2  # listener half 2 (resume + convert)
    python3 scripts/kaggle_train.py ocr           # the photo reader fine-tune
    python3 scripts/kaggle_train.py speech --status
    python3 scripts/kaggle_train.py speech --fetch

Needs an API token once: kaggle.com/settings -> API -> Create New Token, which
downloads kaggle.json. Put it at ~/.kaggle/kaggle.json and chmod 600 it. Nothing
else about the account is touched.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state as repo_state

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(REPO_ROOT / ".venv" / "bin" / "kaggle")
WEIGHTS = REPO_ROOT / "models" / "lilly" / "translate"
READ_PASS1 = REPO_ROOT / "models" / "lilly" / "read" / "lilly.pth"
OCR_CROPS = REPO_ROOT / "data" / "ocr" / "crops"
# The extra corpus travels with the run instead of being re-harvested on Kaggle.
# data/extra/ is gitignored, so the notebook used to rebuild it up there with
# download_extra_data.py -- and a live web corpus does not come back
# byte-identical: the harvest that produced this file returned 38,279 pairs, two
# were removed afterwards leaving 38,277, and the Kaggle re-harvest returned
# 38,280. build_training_mix.py asserts roughly twenty per-step counts exactly,
# all of them measured on the 38,277-row file, so a three-row drift kills the run
# on its first assertion. Uploading the file removes the divergence; loosening
# those assertions would only hide it.
CORPUS = REPO_ROOT / "data" / "extra" / "extra-train.tsv"
# Which card to ask Kaggle for. The value has to come from Kaggle's own enum --
# NvidiaTeslaP100, NvidiaTeslaT4, NvidiaTeslaT4Highmem, NvidiaTeslaA100, NvidiaL4,
# NvidiaH100, TpuV38 and so on. The client forwards whatever string it is given
# without checking, and the server silently discards one it does not recognise
# and falls back to its default card. We spent four runs on a P100 that the
# image's PyTorch cannot use because "gpuT4x2" was an invented string: no error,
# no warning, just the wrong hardware.
ACCELERATOR = "NvidiaTeslaT4"
# Two jobs can go to Kaggle. Translation needs the base weights uploaded as a
# dataset; speech fetches its own audio, so it needs nothing but a GPU.
JOBS = {
    "translation": {"notebook": "Lilly_Translation_Kaggle.ipynb",
                    "slug": "lilly-translation", "title": "Lilly translation",
                    "needs_weights": True, "needs_corpus": True},
    "speech":      {"notebook": "Lilly_Speech_Kaggle.ipynb",
                    "slug": "lilly-speech", "title": "Lilly speech",
                    "needs_weights": False, "needs_corpus": False},
    "speech-half2": {"notebook": "Lilly_Speech_Kaggle_Half2.ipynb",
                    "slug": "lilly-speech-half2", "title": "Lilly speech half2",
                    "needs_weights": False, "needs_corpus": False,
                    "kernel_sources": ["lilly-speech"]},
    "ocr":         {"notebook": "Lilly_OCR_Kaggle.ipynb",
                    "slug": "lilly-ocr", "title": "Lilly ocr",
                    "needs_weights": False, "needs_corpus": False,
                    "needs_read_pass1": True, "needs_ocr_crops": True,
                    "needs_ocr_harvest": True},
}
STAGING = REPO_ROOT / "models" / "kaggle-staging"     # gitignored, under models/


def username() -> str:
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        raise SystemExit(
            "No API token. kaggle.com/settings -> API -> Create New Token, then:\n"
            "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "  chmod 600 ~/.kaggle/kaggle.json")
    return json.loads(token.read_text())["username"]


def run(*args, **kw):
    print("$", " ".join(str(a) for a in args), flush=True)
    return subprocess.run([str(a) for a in args], check=kw.pop("check", True),
                          text=True, capture_output=kw.pop("quiet", False))


def push_weights(user: str) -> str:
    """The 457 MB base model, as a dataset the notebook can attach."""
    slug = f"{user}/lilly-translate-base"
    stage = STAGING / "dataset"
    stage.mkdir(parents=True, exist_ok=True)
    for f in WEIGHTS.iterdir():
        if f.is_file():
            target = stage / f.name
            if not target.exists():
                target.write_bytes(f.read_bytes())
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "Lilly translate base", "id": slug,
        "licenses": [{"name": "other"}]}, indent=1))

    existing = subprocess.run([KAGGLE, "datasets", "status", slug],
                              text=True, capture_output=True)
    if "ready" in existing.stdout.lower():
        print(f"dataset already there: {slug}")
        return slug
    print(f"uploading {sum(f.stat().st_size for f in stage.iterdir()) / 1048576:.0f} MB "
          f"to {slug} — this is the slow part, once")
    run(KAGGLE, "datasets", "create", "-p", stage, "-r", "zip")
    wait_until_ready(slug)
    return slug


def wait_until_ready(slug: str, patience: int = 900) -> None:
    """Block until Kaggle has finished processing the dataset.

    `datasets create` returns as soon as the bytes are sent; Kaggle then
    processes them, and until that finishes the dataset cannot be attached to a
    notebook. Pushing straight afterwards printed

        The following are not valid dataset sources and could not be added

    and pushed the kernel anyway — so the run started with no base weights and
    would have died several cells in, having burned a GPU slot. The message is
    not an error the CLI exits on, which is why it went past unnoticed.
    """
    started = time.time()
    while time.time() - started < patience:
        state = subprocess.run([KAGGLE, "datasets", "status", slug],
                               text=True, capture_output=True).stdout.lower()
        if "ready" in state:
            print(f"  dataset ready after {time.time() - started:.0f}s")
            return
        if "error" in state:
            raise SystemExit(f"Kaggle could not process {slug}: {state.strip()}")
        time.sleep(15)
    raise SystemExit(
        f"{slug} was still not ready after {patience}s. Pushing now would attach "
        f"nothing and the run would die several cells in — check the dataset by "
        f"hand at kaggle.com/datasets/{slug}")


def push_corpus(user: str) -> str:
    """The 38,277 extra pairs, pinned so the mix recipe's counts still mean something.

    Why this is an upload and not a download: build_training_mix.py's twenty-odd
    per-step assertions were all measured on this exact file, and the run that
    re-harvested it on Kaggle died on the first one -- 390,172 rows measured
    against 351,889 expected. Half of that gap was a double-count in the notebook
    and is fixed there; the rest was the harvest itself returning 38,280 pairs
    where this file has 38,277. Three rows out of thirty-eight thousand is
    nothing to the model and fatal to an exact assertion, and the honest way to
    settle it is to send the measured corpus rather than to widen the guards that
    caught the drift.

    Re-uploads only when the local file's md5 differs from the one last sent, so
    the usual case costs a status call.
    """
    slug = f"{user}/lilly-extra-corpus"
    if not CORPUS.exists():
        raise SystemExit(
            f"no extra corpus at {CORPUS}.\n"
            f"Rebuild it with:  python3 data/scripts/download_extra_data.py\n"
            f"but note it will not come back byte-identical, and every count in "
            f"data/scripts/build_training_mix.py has to be re-measured with "
            f"--dry-run before a run can use it.")
    digest = hashlib.md5(CORPUS.read_bytes()).hexdigest()
    stage = STAGING / "corpus"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / CORPUS.name).write_bytes(CORPUS.read_bytes())
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "Lilly extra corpus", "id": slug,
        "licenses": [{"name": "other"}]}, indent=1))
    marker = STAGING / "corpus.md5"          # outside stage/, so it is not uploaded

    state = subprocess.run([KAGGLE, "datasets", "status", slug],
                           text=True, capture_output=True).stdout.lower()
    if "ready" in state:
        if marker.exists() and marker.read_text().strip() == digest:
            print(f"corpus already there: {slug} (md5 {digest[:8]}, {CORPUS.stat().st_size:,} bytes)")
            return slug
        print(f"corpus changed or never confirmed — pushing a new version of {slug}")
        run(KAGGLE, "datasets", "version", "-p", stage, "-r", "zip",
            "-m", f"extra-train.tsv md5 {digest}")
    else:
        print(f"uploading {CORPUS.stat().st_size / 1048576:.0f} MB to {slug}")
        run(KAGGLE, "datasets", "create", "-p", stage, "-r", "zip")
    wait_until_ready(slug)
    marker.write_text(digest)
    return slug


def push_read_pass1(user: str) -> str:
    """Pass-1 lilly.pth + user_network, so heavy pass-2 continues instead of restarting.

    Files sit at the dataset root. `datasets create -r zip` would otherwise pack
    a `read/` folder into `read.zip`, and the notebook would never see lilly.pth
    (v3 died on that in ~30s).
    """
    slug = f"{user}/lilly-read-pass1"
    net = READ_PASS1.parent / "user_network"
    for needed in (READ_PASS1, net / "lilly.yaml", net / "lilly.py"):
        if not needed.is_file():
            raise SystemExit(
                f"no pass-1 reader at {needed} — fetch lilly-read.zip from the "
                f"last OCR run first, then launch heavy pass-2.")
    stage = STAGING / "read-pass1"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / "lilly.pth").write_bytes(READ_PASS1.read_bytes())
    shutil.copy(net / "lilly.yaml", stage / "lilly.yaml")
    shutil.copy(net / "lilly.py", stage / "lilly.py")
    digest = hashlib.md5((stage / "lilly.pth").read_bytes()).hexdigest()
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "Lilly read pass-1", "id": slug,
        "licenses": [{"name": "other"}]}, indent=1))
    marker = STAGING / "read-pass1.md5"

    state = subprocess.run([KAGGLE, "datasets", "status", slug],
                              text=True, capture_output=True).stdout.lower()
    if "ready" in state and marker.exists() and marker.read_text().strip() == digest:
        print(f"read pass-1 already there: {slug} (md5 {digest[:8]})")
        return slug
    mb = (stage / "lilly.pth").stat().st_size / 1048576
    if "ready" in state:
        print(f"pass-1 reader changed — pushing a new version of {slug} ({mb:.0f} MB)")
        run(KAGGLE, "datasets", "version", "-p", stage, "-r", "zip",
            "-m", f"lilly.pth md5 {digest}")
    else:
        print(f"uploading {mb:.0f} MB pass-1 reader to {slug}")
        run(KAGGLE, "datasets", "create", "-p", stage, "-r", "zip")
    wait_until_ready(slug)
    marker.write_text(digest)
    return slug


def push_ocr_crops(user: str) -> str:
    """Hand-labelled real crop PNGs. They are not in git; the notebook copies them.

    One zip at the dataset root (same trap as pass-1: a nested `crops/` folder
    packed with `-r zip` becomes `crops.zip` and cell 5b never sees a PNG).
    """
    import zipfile

    slug = f"{user}/lilly-ocr-crops"
    labels = OCR_CROPS / "labels-human.tsv"
    pngs = sorted(OCR_CROPS.glob("*.png"))
    if not labels.is_file():
        raise SystemExit(f"no real-crop labels at {labels}")
    if len(pngs) < 500:
        raise SystemExit(
            f"only {len(pngs)} crop PNGs in {OCR_CROPS} — pass-3 needs the "
            f"hand-labelled set (about 1,900). They are gitignored.")
    stage = STAGING / "ocr-crops"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    zip_path = stage / "crops.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(labels, "labels-human.tsv")
        for png in pngs:
            zf.write(png, png.name)
    digest = hashlib.md5(zip_path.read_bytes()).hexdigest()
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "Lilly OCR real crops", "id": slug,
        "licenses": [{"name": "other"}]}, indent=1))
    marker = STAGING / "ocr-crops.md5"

    state = subprocess.run([KAGGLE, "datasets", "status", slug],
                           text=True, capture_output=True).stdout.lower()
    if "ready" in state and marker.exists() and marker.read_text().strip() == digest:
        print(f"real crops already there: {slug} ({len(pngs)} pngs, md5 {digest[:8]})")
        return slug
    mb = zip_path.stat().st_size / 1048576
    if "ready" in state:
        print(f"real crops changed — pushing a new version of {slug} ({mb:.0f} MB)")
        run(KAGGLE, "datasets", "version", "-p", stage, "-r", "zip",
            "-m", f"crops.zip md5 {digest} n={len(pngs)}")
    else:
        print(f"uploading {mb:.0f} MB real crops ({len(pngs)} pngs) to {slug}")
        run(KAGGLE, "datasets", "create", "-p", stage, "-r", "zip")
    wait_until_ready(slug)
    marker.write_text(digest)
    return slug


def push_ocr_harvest(user: str) -> "str | None":
    """Pass-7 Commons photographs (full scenes + CREDITS). Required for OCR."""
    import zipfile

    harvested = REPO_ROOT / "data" / "ocr" / "real-photos" / "harvested"
    credits = harvested / "CREDITS.tsv"
    photos = sorted(
        p for p in harvested.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not credits.is_file() or len(photos) < 50:
        print(f"harvest photos not ready ({len(photos)} files, need CREDITS.tsv + ≥50 photos)")
        return None
    slug = f"{user}/lilly-ocr-harvest"
    stage = STAGING / "ocr-harvest"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    zip_path = stage / "harvest.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(credits, "CREDITS.tsv")
        for photo in photos:
            zf.write(photo, f"harvest/{photo.name}")
    digest = hashlib.md5(zip_path.read_bytes()).hexdigest()
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "Lilly OCR harvest photos", "id": slug,
        "licenses": [{"name": "other"}]}, indent=1))
    marker = STAGING / "ocr-harvest.md5"
    state = subprocess.run([KAGGLE, "datasets", "status", slug],
                           text=True, capture_output=True).stdout.lower()
    if "ready" in state and marker.exists() and marker.read_text().strip() == digest:
        print(f"harvest photos already there: {slug} ({len(photos)} files, md5 {digest[:8]})")
        return slug
    mb = zip_path.stat().st_size / 1048576
    if "ready" in state:
        print(f"harvest photos changed — pushing {slug} ({mb:.0f} MB, {len(photos)} files)")
        run(KAGGLE, "datasets", "version", "-p", stage, "-r", "zip",
            "-m", f"harvest.zip md5 {digest} n={len(photos)}")
    else:
        print(f"uploading {mb:.0f} MB harvest photos ({len(photos)} files) to {slug}")
        run(KAGGLE, "datasets", "create", "-p", stage, "-r", "zip")
    wait_until_ready(slug)
    marker.write_text(digest)
    return slug


def require_github_matches_notebook() -> None:
    """Refuse to launch while GitHub and this machine disagree about the code.

    The notebook is uploaded from the working tree; the scripts it calls are
    `git clone`d from GitHub inside the run. Those are two different sources of
    the same codebase, and nothing used to check that they agreed.

    That gap is what killed the 27 August speech run. The notebook cell called
    `download_extra_speech.py --source fleurs_hr --hours 12`, which is the
    rewritten downloader's interface. The rewrite was still uncommitted, so the
    clone brought back the previous version -- `--lang`, `--max-clips` -- and
    argparse exited 2 about thirty-five minutes in, after the Bosnian audio had
    already been fetched. The notebook was from the future and the scripts were
    from the past, and the only symptom was an "unrecognized arguments" line.

    So the launcher asks the question before spending the GPU slot: is the tree
    clean, and does GitHub have this exact commit? Asked by SHA, because
    raw.githubusercontent.com's `/main/` is CDN-cached and has served a
    two-commit-stale file on this project already.

    There is deliberately no --force. A run launched past this check is a run
    whose result cannot be attributed to any particular version of the code,
    and that is worse than not launching.
    """
    head = repo_state.git("rev-parse", "HEAD")
    dirty = [l for l in repo_state.git("status", "--porcelain").splitlines()
             if not l.startswith("??")]
    unpushed = repo_state.git("log", "--oneline", "@{u}..HEAD").splitlines()

    if dirty or unpushed:
        raise SystemExit(
            "Not launching: this machine has code GitHub does not.\n"
            + (f"  {len(dirty)} file(s) changed and not committed\n" if dirty else "")
            + (f"  {len(unpushed)} commit(s) not pushed\n" if unpushed else "")
            + "\nThe notebook goes up from here, the scripts it calls come down "
              "from GitHub.\nLaunching now runs a new notebook against old "
              "scripts. Commit and push first,\nthen run scripts/state.py.")

    if not repo_state.on_github(head):
        raise SystemExit(
            f"Not launching: GitHub does not have {head[:8]}.\n"
            f"The push has not landed yet. Wait, re-check with scripts/state.py, "
            f"then launch.")

    print(f"code check: GitHub has {head[:8]}, tree is clean")


def push_notebook(user: str, job: dict, datasets: list) -> str:
    """The notebook, with a GPU and the internet switched on from here."""
    slug = f"{user}/{job['slug']}"
    notebook = REPO_ROOT / "training" / job["notebook"]
    stage = STAGING / job["slug"]
    stage.mkdir(parents=True, exist_ok=True)
    (stage / notebook.name).write_text(notebook.read_text())
    # Kaggle derives the notebook's address from its TITLE, not from the id in
    # this file. A title that does not match sends the push to whatever kernel
    # the title resolves to — which once put the speech notebook into the
    # translation kernel and ran it there under the wrong name.
    assert job["title"].lower().replace(" ", "-") == job["slug"], \
        f"title {job['title']!r} does not resolve to {job['slug']!r}"
    (stage / "kernel-metadata.json").write_text(json.dumps({
        "id": slug,
        "title": job["title"],
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        # enable_gpu alone is not enough: the request also carries a machine_shape,
        # and when that is empty the run lands on a CPU box however the flag reads.
        # The notebook's own assert then stops it in ten seconds — which is how this
        # was caught rather than discovered after an hour of CPU training.
        "machine_shape": ACCELERATOR,
        "enable_internet": True,
        "dataset_sources": list(datasets),
        "competition_sources": [],
        "kernel_sources": [
            src if "/" in src else f"{user}/{src}"
            for src in job.get("kernel_sources") or []
        ],
    }, indent=1))
    confirm_push(run(KAGGLE, "kernels", "push", "-p", stage, quiet=True))
    confirm_accelerator(slug, stage)
    return slug


def confirm_push(pushed) -> None:
    """Refuse to report a run that was never started.

    `kernels push` exits 0 when the server rejects it. "Maximum batch GPU
    session count of 2 reached" came back on stdout with a zero status, and the
    launcher went on to print the kernel's URL and "running" — so a run that
    does not exist looks exactly like one that does, and the watcher then polls
    the previous version's status and reports it as this one's.
    """
    output = (pushed.stdout or "") + (pushed.stderr or "")
    print(output.strip())
    if "successfully pushed" not in output.lower():
        raise SystemExit(
            "Not running: Kaggle refused the push (see the message above).\n"
            "A GPU session limit means a run has to finish or be stopped in the "
            "browser first;\nnothing was launched, so nothing is worth watching.")


def confirm_accelerator(slug: str, stage: Path) -> None:
    """Check that Kaggle kept the card we asked for.

    Reading it back is the whole point. An unrecognised machine_shape is not
    rejected — it is discarded, and the run lands on Kaggle's default card with
    no error anywhere. That is how four runs went to a P100 the image's PyTorch
    cannot use: the string we sent was invented, so the server threw it away and
    the reply said a bare "Gpu". A value from Kaggle's own enum comes back
    verbatim, which is exactly what makes this check worth making.
    """
    check = Path(tempfile.mkdtemp(prefix="kaggle-confirm-"))
    subprocess.run([KAGGLE, "kernels", "pull", slug, "-p", str(check), "-m"],
                   capture_output=True, text=True, timeout=180)
    meta = check / "kernel-metadata.json"
    if not meta.exists():
        print("  could not read the run's settings back")
        return
    got = json.loads(meta.read_text(encoding="utf-8")).get("machine_shape") or "(none)"
    if got == ACCELERATOR:
        print(f"  accelerator: {got} — Kaggle kept it")
        return
    print(f"  accelerator: asked for {ACCELERATOR!r}, Kaggle stored {got!r}. A reply "
          f"of 'Gpu' means the name was not recognised and was thrown away, so the "
          f"run is on whatever card Kaggle defaults to.")


def status(slug: str) -> str:
    out = subprocess.run([KAGGLE, "kernels", "status", slug],
                         text=True, capture_output=True).stdout.strip()
    print(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", nargs="?", default="translation", choices=sorted(JOBS),
                    help="which run to send up")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--watch", action="store_true", help="poll until it finishes")
    args = ap.parse_args()

    user = username()
    job = JOBS[args.job]
    slug = f"{user}/{job['slug']}"

    if args.status:
        return 0 if status(slug) else 1
    if args.fetch:
        out = REPO_ROOT / "models" / "kaggle-output"
        out.mkdir(parents=True, exist_ok=True)
        run(KAGGLE, "kernels", "output", slug, "-p", out)
        print(f"\nfetched to {out}")
        if args.job in ("speech", "speech-half2"):
            print("  python3 scripts/install_listen.py")
        elif args.job == "ocr":
            print("  unzip lilly-read.zip into models/lilly/read/")
        else:
            print("  unzip adapter to models/lilly/adapter/, then:")
            print("  python3 scripts/build_translator.py")
        return 0

    # Before anything is uploaded: the notebook and the cloned scripts have to
    # be the same version of this project. See require_github_matches_notebook.
    require_github_matches_notebook()
    run(sys.executable, str(REPO_ROOT / "scripts" / "preflight_kaggle.py"))

    datasets = []
    if job["needs_weights"]:
        if not WEIGHTS.exists():
            print(f"no weights at {WEIGHTS} — run scripts/fetch_models.py first",
                  file=sys.stderr)
            return 1
        datasets.append(push_weights(user))
    if job["needs_corpus"]:
        datasets.append(push_corpus(user))
    if job.get("needs_read_pass1"):
        datasets.append(push_read_pass1(user))
    if job.get("needs_ocr_crops"):
        datasets.append(push_ocr_crops(user))
    if job.get("needs_ocr_harvest"):
        hv = push_ocr_harvest(user)
        if hv is None:
            raise SystemExit(
                "OCR harvest photos are required for this pass but are not ready.\n"
                "Run the harvester (lilly-ocr-harvest) first, then relaunch.")
        datasets.append(hv)

    push_notebook(user, job, datasets)
    print(f"\nrunning: https://www.kaggle.com/code/{slug}")
    print("check on it with:  python3 scripts/kaggle_train.py --status")

    if args.watch:
        while True:
            time.sleep(120)
            state = status(slug).lower()
            if "complete" in state or "error" in state or "cancel" in state:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
