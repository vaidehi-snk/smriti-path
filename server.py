import os
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from extract import extract

app = FastAPI(title="SmritiPath")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def load_audio(path: str):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise HTTPException(400, "Could not read that audio file. Try .mp3, .m4a, .wav or .mpeg.")
    return np.frombuffer(result.stdout, dtype=np.float32)


class Text(BaseModel):
    transcript: str


@app.post("/api/transcribe")
def api_transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        path = tmp.name
    try:
        audio = load_audio(path)  # the temp file is deleted right after decoding
    finally:
        os.remove(path)
    segments, info = get_model().transcribe(audio, vad_filter=True)
    return {"language": info.language,
            "transcript": " ".join(s.text.strip() for s in segments)}


@app.post("/api/extract")
def api_extract(body: Text):
    try:
        return extract(body.transcript)
    except Exception as e:
        raise HTTPException(500, f"Ollama failed: {e}. Is Ollama running and the model pulled?")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
