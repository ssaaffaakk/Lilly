# Lilly on a Hugging Face Space.
#
# Not the repository's own Dockerfile. Three things differ, and each of them
# would break a Space:
#
#   1. The published bundle already contains a built translator/, and the
#      float32 base it would be built FROM is deliberately not published. So
#      the quantisation step the other Dockerfile runs has nothing to work on
#      here — the work was already done before upload.
#   2. A free Space has no persistent volume. Corrections written anywhere are
#      lost on the next restart, so they go to a temp path and the interface is
#      honest about it rather than pretending they are kept.
#   3. Spaces run as user 1000, not root. Anything the app writes has to be
#      somewhere that user owns, which is why HOME and the caches are set
#      explicitly instead of landing in /root and failing at first request.
FROM python:3.12-slim

# easyocr pulls in opencv (libGL, glib), soundfile needs libsndfile, and the
# voice needs espeak-ng for words its dictionary does not carry.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# The user a Space actually runs as. Created before anything is installed so
# every path below is owned by it.
RUN useradd -m -u 1000 lilly
ENV HOME=/home/lilly \
    HF_HOME=/home/lilly/.cache/huggingface \
    EASYOCR_MODULE_PATH=/home/lilly/.EasyOCR \
    LILLY_DB=/tmp/feedback.db \
    PORT=7860
# /app has to exist AND belong to lilly before WORKDIR touches it. WORKDIR
# creates a missing directory as root, and fetch_models.py then writes 900 MB of
# weights into /app/models as user 1000 — which fails on permission, during the
# build, after the dependencies have already installed.
RUN mkdir -p /app && chown lilly:lilly /app
WORKDIR /app

# The Space's own list: only what app/ imports. requirements.txt in the
# repository root also carries the training and publishing stack.
COPY --chown=lilly requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=lilly scripts/ scripts/
USER lilly

# 900 MB, fetched at build time so the first visitor does not wait for it.
ARG LILLY_MODELS_REPO=Safak11/lilly
RUN LILLY_MODELS_REPO=${LILLY_MODELS_REPO} python3 scripts/fetch_models.py

COPY --chown=lilly app/ app/

EXPOSE 7860
# One worker on purpose — a second would load its own copy of 900 MB of weights,
# and the free tier's 16 GB is generous but not free of consequence.
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
