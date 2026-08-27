#!/bin/zsh
# Send the Space to Hugging Face. Run from the repository root:
#
#     HF_TOKEN='hf_...' zsh space/push.sh Safak11/lilly-demo
#
# A Space is a git repository, so this stages exactly the files it needs —
# the Space Dockerfile as ./Dockerfile, the card as ./README.md, and the app
# itself — and pushes. The models are not copied: the build fetches them from
# Safak11/lilly, which is where they already live.
set -eu
REPO=${1:?usage: push.sh <user>/<space-name>}
: ${HF_TOKEN:?HF_TOKEN is not set}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

cp "$ROOT/space/Dockerfile" "$STAGE/Dockerfile"
cp "$ROOT/space/README.md"  "$STAGE/README.md"
cp "$ROOT/requirements.txt" "$STAGE/requirements.txt"
cp -R "$ROOT/app"     "$STAGE/app"
cp -R "$ROOT/scripts" "$STAGE/scripts"

# Nothing generated, nothing heavy: a Space that carries weights in git is a
# Space that takes ten minutes to clone and breaks the 5 GB limit.
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true

echo "staging $(du -sh "$STAGE" | cut -f1) for $REPO"
find "$STAGE" -type f | sed "s|$STAGE/|  |" | sort | head -20

cd "$STAGE"
git init -q
git config user.email "noreply@huggingface.co"
git config user.name "Lilly"
git add -A
git commit -qm "Lilly: Bosnian to English, typed, spoken or photographed"
git push -q --force "https://user:${HF_TOKEN}@huggingface.co/spaces/${REPO}" main
echo "pushed — building at https://huggingface.co/spaces/${REPO}"
