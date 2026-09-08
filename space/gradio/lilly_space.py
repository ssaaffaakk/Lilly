"""Lilly on a Hugging Face Space with the Gradio SDK.

Why this file exists. Since mid-2026 a free personal account cannot CREATE a
Docker or Gradio Space (the create call answers HTTP 402, "requires a PRO
subscription"), but a Space that already exists keeps running on the free
cpu-basic hardware. Safak11/Lilly-api is such a Space, created 14 June 2026
with the Gradio SDK. A Gradio Space runs `python app.py` and proxies whatever
answers on port 7860, so this file fetches the bundle and starts the same
FastAPI server the Docker Space runs (space/Dockerfile). No Gradio interface
is built; the web UI is app/web/index.html, as everywhere else.

Not named app.py, although that is the Gradio default (the card names it in
`app_file`): a file called app.py beside the app/ package shadows the package,
because app/ has no __init__.py and a namespace package loses to a module of
the same name. `from app.server import app` then fails with "'app' is not a
package". Found by the smoke test, not on the Space.

What differs from the Docker Space:
  * there is no build step, so the weights (about 2.3 GB) are fetched every
    time the Space starts -- a few minutes after it wakes from sleep;
  * apt packages come from packages.txt, Python packages from requirements.txt;
  * corrections go to /tmp and are lost on restart, as on the Docker Space.

LILLY_FETCH=0 skips the fetch, for a machine that already has models/lilly
(the smoke test in the repository). Everything else is the server as shipped.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("LILLY_DB", "/tmp/feedback.db")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")


def fetch() -> None:
    if os.environ.get("LILLY_FETCH", "1") == "0":
        print("LILLY_FETCH=0: serving the models/lilly already here", flush=True)
        return
    # check=True on purpose. A Space that starts without its weights would show
    # "Running" and answer every request with a 500; exit 1 here shows as a
    # runtime error with the fetcher's own message at the end of the log.
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_models.py")],
                   check=True)


def main() -> None:
    fetch()
    sys.path.insert(0, str(ROOT))
    import uvicorn
    from app.server import app
    # One worker on purpose: a second would load its own copy of the weights.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
