#!/bin/zsh
# Send the Space to Hugging Face. Run from the repository root:
#
#     zsh space/push.sh Safak11/Lilly-api                  # Gradio SDK, the default
#     zsh space/push.sh Safak11/Lilly-api --sdk docker     # Docker SDK
#     zsh space/push.sh Safak11/Lilly-api --dry-run        # stage, send nothing
#
# The token is the one `hf auth login` stored (huggingface_hub's get_token()),
# so nothing is pasted on a command line; HF_TOKEN in the environment overrides.
#
# Two SDKs, one app. Since mid-2026 a free personal account cannot CREATE a
# Gradio or Docker Space (HTTP 402 from the create call: "requires a PRO
# subscription"), but a Space that already exists keeps running on cpu-basic.
# Safak11/Lilly-api exists, with the Gradio SDK, so the default stages
# space/gradio/lilly_space.py (fetches the bundle at start, then runs the
# FastAPI server on 7860; named in the card's app_file) with packages.txt for
# apt; --sdk docker stages the
# Dockerfile instead, for an account that may create Docker Spaces. Either way
# the card is space/README.md with the SDK's own front matter, the app is app/,
# and the models are not copied: they are fetched from Safak11/lilly, where
# they already live.
#
# --dry-run does everything except the two network calls, so the staging, the
# file list and the interpreter can be checked without a token. Added because
# three separate defects in this script reached the owner instead of being
# caught here: it assumed a git push creates a Space, it shipped the whole of
# scripts/ including an overnight watchdog, and it called the system python for
# a library that lives in .venv. Each would have died on one local run.
set -eu
REPO=${1:?usage: push.sh <user>/<space-name> [--sdk gradio|docker] [--dry-run]}
shift
DRY=
SDK=gradio
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=--dry-run ;;
    --sdk) SDK=${2:?--sdk needs gradio or docker}; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
case "$SDK" in gradio|docker) ;; *) echo "--sdk must be gradio or docker, not $SDK" >&2; exit 2 ;; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# The repository's own interpreter: huggingface_hub lives in .venv, and the
# system python3 does not have it.
PY_BIN="$ROOT/.venv/bin/python"
[ -x "$PY_BIN" ] || PY_BIN=python3
# The token `hf auth login` stored, unless HF_TOKEN is already set. Never printed.
if [ "$DRY" != "--dry-run" ] && [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN=$("$PY_BIN" -c 'from huggingface_hub import get_token; print(get_token() or "")')
  export HF_TOKEN
fi
[ "$DRY" = "--dry-run" ] || : ${HF_TOKEN:?no token: run .venv/bin/hf auth login, or set HF_TOKEN}

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

cp "$ROOT/space/requirements.txt" "$STAGE/requirements.txt"
cp -R "$ROOT/app" "$STAGE/app"
if [ "$SDK" = docker ]; then
  cp "$ROOT/space/Dockerfile" "$STAGE/Dockerfile"
  cp "$ROOT/space/README.md"  "$STAGE/README.md"
else
  cp "$ROOT/space/gradio/lilly_space.py" "$STAGE/lilly_space.py"
  cp "$ROOT/space/gradio/packages.txt" "$STAGE/packages.txt"
  # The same card, under the Gradio SDK's front matter: the body of
  # space/README.md is everything after its own front matter. sdk_version is
  # the last Gradio 5: the SDK's own build step installs gradio==sdk_version,
  # and Gradio 6 needs huggingface-hub >= 1.2 while transformers 4.49.0 (the
  # tokeniser's version, pinned in requirements.txt) needs < 1.0. The first
  # build on 6.18.0 died on exactly that: ResolutionImpossible.
  {
    printf '%s\n' '---' 'title: Lilly' 'emoji: 🌉' 'colorFrom: blue' 'colorTo: gray' \
      'sdk: gradio' 'sdk_version: 5.50.0' 'python_version: "3.12"' 'app_file: lilly_space.py' \
      'pinned: false' 'license: other' 'models:' '  - Safak11/lilly' '---'
    awk 'seen >= 2 { print } /^---$/ { seen++ }' "$ROOT/space/README.md"
  } > "$STAGE/README.md"
fi

# Only the script the build actually runs. The rest — training launchers, the
# publisher, the overnight watcher — have no business in a public Space: they
# are noise at best and a reader wondering why a demo ships a Kaggle uploader at
# worst.
mkdir -p "$STAGE/scripts"
cp "$ROOT/scripts/fetch_models.py" "$STAGE/scripts/"

# Nothing generated, nothing heavy: a Space that carries weights in git is a
# Space that takes ten minutes to clone and breaks the 5 GB limit.
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true

echo "staging $(du -sh "$STAGE" | cut -f1) for $REPO ($SDK SDK)"
find "$STAGE" -type f | sed "s|$STAGE/|  |" | sort | head -20

# A git push does not create a Space; without this the remote simply is not
# there and the push fails with "Repository not found" after staging everything.
"$PY_BIN" - "$REPO" "$DRY" "$SDK" <<'PYTHON'
import os, sys
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
repo, dry, sdk = sys.argv[1], sys.argv[2] == "--dry-run", sys.argv[3]
if dry:
    # Prove the calls are well formed without making them: a wrong argument
    # name here is exactly the kind of thing that only shows up mid-push.
    import inspect
    params = inspect.signature(HfApi.create_repo).parameters
    for needed in ("repo_id", "repo_type", "space_sdk", "exist_ok"):
        assert needed in params, f"create_repo has no {needed}"
    assert "repo_type" in inspect.signature(HfApi.repo_exists).parameters
    print(f"dry run: repo_exists / create_repo({repo!r}, repo_type='space', space_sdk={sdk!r}) are valid")
    sys.exit(0)
api = HfApi(token=os.environ["HF_TOKEN"])
if api.repo_exists(repo, repo_type="space"):
    # No create call for a Space that exists: the create endpoint answers 402
    # on a free account even with exist_ok, and the existing Space keeps
    # running on cpu-basic. Its SDK is whatever the pushed README says.
    print(f"space exists: {repo}")
else:
    try:
        api.create_repo(repo, repo_type="space", space_sdk=sdk, exist_ok=True)
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 402:
            sys.exit(f"Hugging Face refused to create a {sdk} Space on this account (402): "
                     "since mid-2026 creating a Gradio or Docker Space needs a PRO plan. "
                     "A Space that already exists keeps running -- push to that one by name, "
                     "or subscribe and run this again.")
        raise
    print(f"space created: {repo} ({sdk})")
PYTHON

cd "$STAGE"
git init -q
git config user.email "noreply@huggingface.co"
git config user.name "Lilly"
git add -A
git commit -qm "Lilly: Bosnian to English, typed, spoken or photographed"
if [ "$DRY" = "--dry-run" ]; then
  echo "dry run: would push $(git rev-list --count HEAD) commit to spaces/${REPO}"
  echo "dry run: nothing was sent"
else
  git push -q --force "https://user:${HF_TOKEN}@huggingface.co/spaces/${REPO}" main
  echo "pushed — building at https://huggingface.co/spaces/${REPO}"
fi
