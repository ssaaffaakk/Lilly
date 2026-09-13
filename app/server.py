#!/usr/bin/env python3
"""Lilly web server — every feature behind one API.

    uvicorn app.server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                  the web app
    GET  /health            liveness, for whatever is watching the process
    POST /api/translate     {"text": "..."}            -> Bosnian text to English
    POST /api/reply         {"text": "..."}            -> English text to Bosnian
    POST /api/detect        {"text": "..."}            -> {"language": "bs"|"en"}
    POST /api/speech        audio upload [+ direction]  -> transcribe, then translate
    POST /api/speak         {"text": "...", "language": "en"|"bs"} -> speech (WAV)
    POST /api/photo         image upload [+ direction]  -> read, then translate
    POST /api/photo-boxes   image upload [+ direction]  -> regions with boxes
    POST /api/document      .docx/.pdf upload [+ direction] -> text, translated
    POST /api/feedback      correction report           -> saved to review database

Every ability comes from the one Lilly object (app/lilly.py), which reads its
weights from models/lilly/. Parts load lazily on first use, so startup is
instant and unused features cost nothing.

Every ability runs both ways. The uploads take an optional `direction` form
field, "bs-en" (the default: Bosnian heard or photographed, English back) or
"en-bs" (English heard or photographed, Bosnian back); the answer is always
{"bosnian": ..., "english": ...}, whichever side was the input. /api/speak takes
`language`, "en" (default) or "bs", for reading the answer aloud on either side.
/api/speech also takes direction="auto" (conversation mode): Whisper decides
which language the clip holds, and the answer says which side was heard.

This is written to face the open internet, so every request is bounded before it
reaches a model: uploads by size, text by how much work it asks for, images by
pixels. The work itself runs off the event loop, one piece at a time, because
all of it is CPU-bound and the process has one set of weights to share.
"""
import os
import re
import tempfile
from pathlib import Path

from typing import Annotated

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, StringConstraints
from starlette.concurrency import run_in_threadpool

from app import feedback
from app.detect import detect_language
from app.lilly import BadInput, lilly
from app.ocr import ImageTooLarge
from app.translate import TextTooLong

APP_DIR = Path(__file__).resolve().parent

# Uploads are bounded before anything reads them. A photo of a sign is well
# under a megabyte; a minute of voice is a few hundred kilobytes. A document
# may carry pictures of its own, so it gets the same room as speech.
MAX_UPLOAD = {"/api/photo": 12 * 1024 * 1024,
              "/api/photo-boxes": 12 * 1024 * 1024,
              "/api/speech": 25 * 1024 * 1024,
              "/api/document": 25 * 1024 * 1024}
UPLOAD_CHUNK = 256 * 1024
SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,8}$")

app = FastAPI(title="Lilly")


# Stripped before the length check: min_length counts spaces, so a body of
# nothing but whitespace used to reach the model, and the voice answered it
# with a 500. Blank is empty, and empty is a 422 like any other short field.
TranslateIn_text = Annotated[str, StringConstraints(strip_whitespace=True,
                                                     min_length=1, max_length=12_000)]
SpeakIn_text = Annotated[str, StringConstraints(strip_whitespace=True,
                                                 min_length=1, max_length=2_000)]


class TranslateIn(BaseModel):
    # A cheap first gate. The real limit is in the engine and counts tokens,
    # because cost follows sentence count and length, not characters.
    text: TranslateIn_text


class SpeakIn(BaseModel):
    text: SpeakIn_text
    language: str = Field(default="en", pattern="^(en|bs)$")


# The uploads' direction, a form field beside the file. Checked here so a typo
# is a 422 with the field named, never a 400 blamed on the recording. Speech
# is the one upload that also takes "auto": conversation mode hears either
# language and lets Whisper decide.
Direction = Annotated[str, Form(pattern="^(bs-en|en-bs)$")]
SpeechDirection = Annotated[str, Form(pattern="^(bs-en|en-bs|auto)$")]


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
    # no-cache means "ask before reusing", not "never cache": the browser sends
    # If-Modified-Since and gets a 304 while the page is unchanged. Without it,
    # a page served with only a Last-Modified header is kept on heuristics --
    # a tenth of its age -- and a browser that loaded Lilly a week ago goes on
    # running that week-old script against a server that has moved on. Seen
    # on 8 September 2026: the microphone flipped the arrow back to Bosnian
    # an hour after the page had stopped doing that.
    return FileResponse(APP_DIR / "web" / "index.html",
                        headers={"Cache-Control": "no-cache"})


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


@app.post("/api/detect")
async def detect(body: TranslateIn):
    """Which language a text is in, so the page can route it without asking.

    Same length bounds as /api/translate: a text too long to translate is too
    long to be worth classifying. The detector is a committed table of counts,
    not a model download, but a machine without the file still gets a 503
    rather than a crash, like any other missing part.
    """
    try:
        language = await run_in_threadpool(detect_language, body.text)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return {"language": language}


@app.post("/api/speech")
async def speech(file: UploadFile, direction: SpeechDirection = "bs-en"):
    tmp_path = await _save_upload(file, "a.webm", MAX_UPLOAD["/api/speech"])
    try:
        if direction == "auto":
            # Conversation mode: the listener decides the language, then the
            # detector routes the text. `heard` says which side the answer
            # belongs to, because both keys hold a language either way.
            bosnian, english, heard = await run_in_threadpool(lilly.converse, tmp_path)
            return {"bosnian": bosnian, "english": english, "heard": heard}
        bosnian, english = await run_in_threadpool(lilly.translate_audio, tmp_path, direction)
    except FileNotFoundError as exc:
        # No listener on this machine (or no reply model behind the answer).
        # Not the caller's recording and not a crash: the same 503 the reply
        # direction answers with.
        return JSONResponse(status_code=503, content={"error": str(exc)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"bosnian": bosnian, "english": english}


@app.post("/api/speak")
async def speak(body: SpeakIn):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await run_in_threadpool(lilly.speak, body.text, tmp_path, body.language)
        data = Path(tmp_path).read_bytes()
    except FileNotFoundError as exc:
        # The Bosnian voice is a separate download, like the reply model:
        # missing weights are neither the caller's text nor a crash.
        return JSONResponse(status_code=503, content={"error": str(exc)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return Response(content=data, media_type="audio/wav")


@app.post("/api/photo")
async def photo(file: UploadFile, direction: Direction = "bs-en"):
    tmp_path = await _save_upload(file, "a.jpg", MAX_UPLOAD["/api/photo"])
    try:
        bosnian, english = await run_in_threadpool(lilly.translate_photo, tmp_path, direction)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"bosnian": bosnian, "english": english}


@app.post("/api/photo-boxes")
async def photo_boxes(file: UploadFile, direction: Direction = "bs-en"):
    """The same read-plus-translate as /api/photo, plus a box per region.

    The page draws each region's translation over the photograph at the place
    the words were found, so this returns the regions with their boxes, their
    source text and their own translation, beside the full pair. Boxes are in
    the original upload's pixels and pass the reader's confidence floor like
    everything else the reader returns. Region-level, not word-level: one box
    per paragraph group.
    """
    tmp_path = await _save_upload(file, "a.jpg", MAX_UPLOAD["/api/photo-boxes"])
    try:
        bosnian, english, regions = await run_in_threadpool(
            lilly.translate_photo_regions, tmp_path, direction)
    except FileNotFoundError as exc:
        # The reader is a required part, but missing weights are reported the
        # way /api/reply reports them: a download to make, not a crash.
        return JSONResponse(status_code=503, content={"error": str(exc)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"regions": regions, "bosnian": bosnian, "english": english}


@app.post("/api/document")
async def document(file: UploadFile, direction: Direction = "bs-en"):
    """A .docx or .pdf upload: its text, read and translated.

    The document is extracted to text (app/document.py) and travels the same
    sentence-split path as anything typed, truncate=True — the caller never
    typed the document, so the beginning is translated rather than the request
    refused. The 503 covers a machine without the translation weights, the
    same way /api/reply answers.
    """
    tmp_path = await _save_upload(file, "a.pdf", MAX_UPLOAD["/api/document"])
    try:
        bosnian, english, original = await run_in_threadpool(
            lilly.translate_document, tmp_path, direction)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"bosnian": bosnian, "english": english, "original": original}


@app.post("/api/feedback")
async def report(body: FeedbackIn):
    row_id = await run_in_threadpool(
        feedback.add_correction, body.source_text, body.model_output,
        body.user_complaint, body.suggested_translation, body.direction)
    return {"ok": True, "id": row_id}
