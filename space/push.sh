#!/bin/zsh
# Send the Space to Hugging Face. Run from the repository root:
#
#     HF_TOKEN='hf_...' zsh space/push.sh Safak11/lilly-demo
#
# A Space is a git repository, so this stages exactly the files it needs —
# the Space Dockerfile as ./Dockerfile, the card as ./README.md, and the app
# itself — and pushes. The models are not copied: the build fetches them from
# Safak11/lilly, which is where they already live.
#     zsh space/push.sh Safak11/lilly-demo --dry-run
#
# --dry-run does everything except the two network calls, so the staging, the
# file list and the interpreter can be checked without a token. Added because
# three separate defects in this script reached the owner instead of being
# caught here: it assumed a git push creates a Space, it shipped the whole of
# scripts/ including an overnight watchdog, and it called the system python for
# a library that lives in .venv. Each would have died on one local run.
set -eu
REPO=${1:?usage: push.sh <user>/<space-name> [--dry-run]}
DRY=${2:-}
[ "$DRY" = "--dry-run" ] || : ${HF_TOKEN:?HF_TOKEN is not set}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

cp "$ROOT/space/Dockerfile" "$STAGE/Dockerfile"
cp "$ROOT/space/README.md"  "$STAGE/README.md"
cp "$ROOT/space/requirements.txt" "$STAGE/requirements.txt"
cp -R "$ROOT/app" "$STAGE/app"

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

echo "staging $(du -sh "$STAGE" | cut -f1) for $REPO"
find "$STAGE" -type f | sed "s|$STAGE/|  |" | sort | head -20

# A git push does not create a Space; without this the remote simply is not
# there and the push fails with "Repository not found" after staging everything.
# The repository's own interpreter: huggingface_hub lives in .venv, and the
# system python3 does not have it.
PY_BIN="$ROOT/.venv/bin/python"
[ -x "$PY_BIN" ] || PY_BIN=python3
"$PY_BIN" - "$REPO" "$DRY" <<'PYTHON'
import os, sys
from huggingface_hub import HfApi
repo, dry = sys.argv[1], sys.argv[2] == "--dry-run"
if dry:
    # Prove the call is well formed without making it: a wrong argument name
    # here is exactly the kind of thing that only shows up mid-push.
    import inspect
    params = inspect.signature(HfApi.create_repo).parameters
    for needed in ("repo_id", "repo_type", "space_sdk", "exist_ok"):
        assert needed in params, f"create_repo has no {needed}"
    print(f"dry run: create_repo({repo!r}, repo_type='space', space_sdk='docker') is valid")
else:
    HfApi(token=os.environ["HF_TOKEN"]).create_repo(
        repo, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"space ready: {repo}")
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
