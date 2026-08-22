#!/usr/bin/env python3
"""Lilly web server — every feature behind one API.

    uvicorn app.server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                  the web app
    POST /api/translate     {"text": "..."}            -> Bosnian text to English
    POST /api/speech        audio file upload           -> transcribe Bosnian + translate
    POST /api/speak         {"text": "..."}            -> English speech (WAV)
    POST /api/photo         image file upload           -> OCR Bosnian + translate
    POST /api/feedback      correction report           -> saved to review database

Every ability comes from the one Lilly object (app/lilly.py), which reads its
weights from models/lilly/. Parts load lazily on first use, so startup is
instant and unused features cost nothing.
"""
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app import feedback
from app.lilly import lilly

APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Lilly")


class TextIn(BaseModel):
    text: str


class FeedbackIn(BaseModel):
    source_text: str
    model_output: str
    user_complaint: str = ""
    suggested_translation: str = ""


async def _save_upload(file: UploadFile, fallback_name: str) -> str:
    """Write an upload to a temp file the models can read, return its path."""
    suffix = Path(file.filename or fallback_name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        return tmp.name


@app.get("/")
def index():
    return FileResponse(APP_DIR / "web" / "index.html")


@app.post("/api/translate")
def translate(body: TextIn):
    return {"bosnian": body.text, "english": lilly.translate(body.text)}


@app.post("/api/speech")
async def speech(file: UploadFile):
    tmp_path = await _save_upload(file, "a.webm")
    try:
        bosnian, english = lilly.translate_audio(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"bosnian": bosnian, "english": english}


@app.post("/api/speak")
def speak(body: TextIn):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    lilly.speak(body.text, tmp_path)
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return Response(content=data, media_type="audio/wav")


@app.post("/api/photo")
async def photo(file: UploadFile):
    tmp_path = await _save_upload(file, "a.jpg")
    try:
        bosnian, english = lilly.translate_photo(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"bosnian": bosnian, "english": english}


@app.post("/api/feedback")
def report(body: FeedbackIn):
    row_id = feedback.add_correction(body.source_text, body.model_output,
                                     body.user_complaint, body.suggested_translation)
    return {"ok": True, "id": row_id}
