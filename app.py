# app.py
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
import uvicorn
import tempfile

app = FastAPI()

# Load ASR model from Hugging Face Hub
pipe = pipeline("automatic-speech-recognition", model="clt013/whisper-large-v3-ft-malay-peft-epoch-20")

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = pipe(tmp_path)
    return {"text": result["text"]}
