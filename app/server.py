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

Models load lazily on first use, so startup is instant and unused features
cost nothing.
"""
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app import feedback
from app.translate import get_engine

APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Lilly")


class TextIn(BaseModel):
    text: str


class FeedbackIn(BaseModel):
    source_text: str
    model_output: str
    user_complaint: str = ""
    suggested_translation: str = ""


@app.get("/")
def index():
    return FileResponse(APP_DIR / "web" / "index.html")


@app.post("/api/translate")
def translate(body: TextIn):
    english = get_engine().translate(body.text)
    return {"bosnian": body.text, "english": english}


@app.post("/api/speech")
async def speech(file: UploadFile):
    from app.speech import transcribe
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "a.webm").suffix,
                                     delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        bosnian = transcribe(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    english = get_engine().translate(bosnian) if bosnian else ""
    return {"bosnian": bosnian, "english": english}


@app.post("/api/speak")
def speak(body: TextIn):
    from app.tts import speak_to_file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    speak_to_file(body.text, tmp_path)
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return Response(content=data, media_type="audio/wav")


@app.post("/api/photo")
async def photo(file: UploadFile):
    from app.ocr import scan
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "a.jpg").suffix,
                                     delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        bosnian = scan(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    english = get_engine().translate(bosnian) if bosnian else ""
    return {"bosnian": bosnian, "english": english}


@app.post("/api/feedback")
def report(body: FeedbackIn):
    row_id = feedback.add_correction(body.source_text, body.model_output,
                                     body.user_complaint, body.suggested_translation)
    return {"ok": True, "id": row_id}
