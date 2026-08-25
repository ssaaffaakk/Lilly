# Lilly, ready to serve. Build:  docker build -t lilly .
#                        Run:    docker run -p 8000:8000 -v lilly-data:/data lilly
#
# The image carries the code and the dependencies; the 1.3 GB of weights are
# fetched during the build, and corrections are written to a volume so they
# survive the next deploy.
FROM python:3.12-slim

# easyocr pulls in opencv (libGL, glib) and soundfile needs libsndfile;
# the voice needs espeak-ng for words its dictionary does not carry.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not reinstall 2 GB of wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
ARG LILLY_MODELS_REPO=Safak11/lilly
RUN LILLY_MODELS_REPO=${LILLY_MODELS_REPO} python3 scripts/fetch_models.py

# Quantise the translator: 1.8 GB of float32 becomes 670 MB and runs about twice
# as fast, for a chrF2 that matches to two decimals. Folds in the fine-tuning if
# there is any. The float32 copy is only needed for training, so it goes.
RUN python3 scripts/build_translator.py && rm -rf models/lilly/translate

COPY app/ app/

# Corrections belong on a mounted volume: anywhere inside the image is wiped by
# the next deploy, taking every correction anyone submitted with it.
ENV LILLY_DB=/data/feedback.db
VOLUME ["/data"]

# One worker on purpose — each one would load its own full 1.3 GB of weights.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python3 -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health').read()"
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
