#!/usr/bin/env python3
"""Lilly web server — every feature behind one API.

    uvicorn app.server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                  the web app
    GET  /health            liveness, for whatever is watching the process
    POST /api/translate     {"text": "..."}            -> Bosnian text to English
    POST /api/reply         {"text": "..."}            -> English text to Bosnian
    POST /api/speech        audio file upload           -> transcribe Bosnian + translate
    POST /api/speak         {"text": "..."}            -> English speech (WAV)
    POST /api/photo         image file upload           -> OCR Bosnian + translate
    POST /api/feedback      correction report           -> saved to review database

Every ability comes from the one Lilly object (app/lilly.py), which reads its
weights from models/lilly/. Parts load lazily on first use, so startup is
instant and unused features cost nothing.

This is written to face the open internet, so every request is bounded before it
reaches a model: uploads by size, text by how much work it asks for, images by
pixels. The work itself runs off the event loop, one piece at a time, because
all of it is CPU-bound and the process has one set of weights to share.
"""
import os
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app import feedback
from app.lilly import BadInput, lilly
from app.ocr import ImageTooLarge
from app.translate import TextTooLong

APP_DIR = Path(__file__).resolve().parent

# Uploads are bounded before anything reads them. A photo of a sign is well
# under a megabyte; a minute of voice is a few hundred kilobytes.
MAX_UPLOAD = {"/api/photo": 12 * 1024 * 1024, "/api/speech": 25 * 1024 * 1024}
UPLOAD_CHUNK = 256 * 1024
SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,8}$")

app = FastAPI(title="Lilly")


class TranslateIn(BaseModel):
    # A cheap first gate. The real limit is in the engine and counts tokens,
    # because cost follows sentence count and length, not characters.
    text: str = Field(min_length=1, max_length=12_000)


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)


class FeedbackIn(BaseModel):
    source_text: str = Field(min_length=1, max_length=12_000)
    model_output: str = Field(max_length=12_000, default="")
    user_complaint: str = Field(max_length=2_000, default="")
    suggested_translation: str = Field(max_length=12_000, default="")
    direction: str = Field(default="bs-en", pattern="^(bs-en|en-bs)$")


@app.middleware("http")
async def refuse_oversized_bodies(request, call_next):
    """Turn a huge upload away on its declared size, before anything reads it."""
    cap = MAX_UPLOAD.get(request.url.path)
    declared = request.headers.get("content-length")
    if cap and declared and declared.isdigit() and int(declared) > cap:
        return JSONResponse(status_code=413,
                            content={"error": f"that file is over {cap // 1048576} MB"})
    return await call_next(request)


@app.exception_handler(TextTooLong)
@app.exception_handler(ImageTooLarge)
async def too_big(request, exc):
    return JSONResponse(status_code=413, content={"error": str(exc)})


@app.exception_handler(BadInput)
async def bad_input(request, exc):
    """Their side of the line: say what is wrong in a sentence, and mean 4xx."""
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(Exception)
async def anything_else(request, exc):
    """Never hand a stranger a stack trace, and never leave them without an answer."""
    print(f"error on {request.url.path}: {type(exc).__name__}: {exc}", flush=True)
    return JSONResponse(status_code=500,
                        content={"error": "something went wrong on our side"})


async def _save_upload(file: UploadFile, fallback_name: str, cap: int) -> str:
    """Write an upload to a temp file the models can read, return its path.

    Read in pieces and stop at the cap: a body that arrives without declaring
    its length gets past the middleware, and `.read()` with no argument would
    hand the whole thing to memory at once.
    """
    suffix = Path(file.filename or fallback_name).suffix
    if not SAFE_SUFFIX.match(suffix):
        suffix = Path(fallback_name).suffix
    written = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while chunk := await file.read(UPLOAD_CHUNK):
            written += len(chunk)
            if written > cap:
                Path(tmp.name).unlink(missing_ok=True)
                raise TextTooLong(f"that file is over {cap // 1048576} MB")
            tmp.write(chunk)
        if not written:
            Path(tmp.name).unlink(missing_ok=True)
            raise TextTooLong("that file is empty")
        return tmp.name


@app.get("/")
def index():
    return FileResponse(APP_DIR / "web" / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/translate")
async def translate(body: TranslateIn):
    english = await run_in_threadpool(lilly.translate, body.text)
    return {"bosnian": body.text, "english": english}


@app.post("/api/reply")
async def reply(body: TranslateIn):
    """The other direction: type English, get Bosnian to say back to somebody.

    Its own route rather than a flag on /api/translate, because the response
    keys mean the same thing in both — which side was typed is the difference,
    and a client reading `english` should not have to know how it was produced.
    """
    try:
        bosnian = await run_in_threadpool(lilly.reply, body.text)
    except FileNotFoundError as exc:
        # The reply model is a separate download. Missing weights is not a bug
        # in the request and not a crash, so it is neither 400 nor 500.
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return {"bosnian": bosnian, "english": body.text}


@app.post("/api/speech")
async def speech(file: UploadFile):
    tmp_path = await _save_upload(file, "a.webm", MAX_UPLOAD["/api/speech"])
    try:
        bosnian, english = await run_in_threadpool(lilly.translate_audio, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"bosnian": bosnian, "english": english}


@app.post("/api/speak")
async def speak(body: SpeakIn):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await run_in_threadpool(lilly.speak, body.text, tmp_path)
        data = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return Response(content=data, media_type="audio/wav")


@app.post("/api/photo")
async def photo(file: UploadFile):
    tmp_path = await _save_upload(file, "a.jpg", MAX_UPLOAD["/api/photo"])
    try:
        bosnian, english = await run_in_threadpool(lilly.translate_photo, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"bosnian": bosnian, "english": english}


@app.post("/api/feedback")
async def report(body: FeedbackIn):
    row_id = await run_in_threadpool(
        feedback.add_correction, body.source_text, body.model_output,
        body.user_complaint, body.suggested_translation, body.direction)
    return {"ok": True, "id": row_id}
