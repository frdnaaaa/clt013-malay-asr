from fastapi import FastAPI, File, UploadFile
from transformers import AutoProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import torch
import torchaudio
import tempfile

# Initialize FastAPI
app = FastAPI()

# Load base Whisper model
print("Loading base Whisper model...")
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base_model, "clt013/whisper-large-v3-ft-malay-peft-epoch-20")
model.eval()

# Load processor
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    # Load and preprocess audio
    speech_array, sampling_rate = torchaudio.load(audio_path)
    inputs = processor(
        speech_array[0],
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )

    # Transcribe
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return {"text": transcription}
