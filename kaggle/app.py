import os
import uuid
import time
import asyncio
from pathlib import Path

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from voxcpm import VoxCPM


# ============================================================
# CONFIG
# ============================================================

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))

API_KEY = os.getenv("VOX_API_KEY", "CHANGE_THIS_KEY")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOICE_DIR = Path(
    "/kaggle/input/datasets/ohmyqueenmedusa/pyaesonevvipvoice"
)

VOICES = {
    "Phyo": VOICE_DIR / "Phyo_vvipvoice.wav",
    "Htun": VOICE_DIR / "Htun_vvipvoice.wav",
    "Pyae": VOICE_DIR / "Pyae_vvipvoice.wav",
    "Fangyung": VOICE_DIR / "Fangyung_vvipvoice.wav",
}

OUTPUT_DIR = Path("/kaggle/working/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VOXCPM SETTINGS
# ============================================================

CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 30
MAX_LEN = 2000
RETRY_BADCASE = False


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="VoxCPM2 Telegram GPU API",
    version="1.0.0"
)


model = None
model_lock = asyncio.Lock()


# ============================================================
# REQUEST MODEL
# ============================================================

class GenerateRequest(BaseModel):
    text: str
    voice: str


# ============================================================
# LOAD MODEL ONLY WHEN NEEDED
# ============================================================

async def get_model():

    global model

    if model is not None:
        return model

    async with model_lock:

        if model is not None:
            return model

        print("========================================")
        print("🚀 Loading VoxCPM2")
        print("========================================")
        print(f"PyTorch : {torch.__version__}")
        print(f"CUDA    : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU     : {torch.cuda.get_device_name(0)}")

        print(f"Device  : {DEVICE}")

        model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2",
            device=DEVICE
        )

        print("✅ VoxCPM2 loaded")

        return model


# ============================================================
# AUTH
# ============================================================

def check_key(key: str):

    if key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "service": "VoxCPM2 GPU API",
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "cuda": torch.cuda.is_available(),
        "model_loaded": model is not None,
    }


@app.get("/voices")
async def voices():

    return {
        "voices": list(VOICES.keys())
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/generate")
async def generate(
    request: GenerateRequest
):

    start_time = time.time()

    check_key(API_KEY)

    text = request.text.strip()
    voice = request.voice.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text is empty"
        )

    if len(text) > MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_LEN} characters allowed"
        )

    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unknown voice",
                "available": list(VOICES.keys())
            }
        )

    reference = VOICES[voice]

    if not reference.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Reference voice missing: {reference}"
        )

    print("----------------------------------------")
    print("🎙 NEW TTS REQUEST")
    print(f"Voice      : {voice}")
    print(f"Characters : {len(text)}")
    print(f"Steps      : {INFERENCE_TIMESTEPS}")
    print("----------------------------------------")

    # --------------------------------------------------------
    # Load model only after actual Telegram request
    # --------------------------------------------------------

    t0 = time.time()

    tts = await get_model()

    model_load_time = time.time() - t0

    # --------------------------------------------------------
    # GPU inference
    # --------------------------------------------------------

    t1 = time.time()

    try:

        with torch.inference_mode():

            wav = tts.generate(
                text=text,

                reference_wav_path=str(reference),

                cfg_value=CFG_VALUE,

                inference_timesteps=INFERENCE_TIMESTEPS,

                retry_badcase=RETRY_BADCASE,

                max_len=MAX_LEN,
            )

    except Exception as e:

        print("❌ GENERATION ERROR")
        print(repr(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    inference_time = time.time() - t1

    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    filename = (
        f"{uuid.uuid4().hex}.wav"
    )

    output_path = OUTPUT_DIR / filename

    sf.write(
        output_path,
        wav,
        tts.tts_model.sample_rate
    )

    total_time = time.time() - start_time

    print("----------------------------------------")
    print("✅ GENERATION COMPLETE")
    print(f"Model load : {model_load_time:.2f}s")
    print(f"Inference  : {inference_time:.2f}s")
    print(f"Total      : {total_time:.2f}s")
    print(f"Output     : {output_path}")
    print("----------------------------------------")

    return {
        "success": True,
        "voice": voice,
        "characters": len(text),
        "steps": INFERENCE_TIMESTEPS,
        "cfg": CFG_VALUE,
        "max_len": MAX_LEN,
        "inference_time": round(inference_time, 2),
        "total_time": round(total_time, 2),
        "file": str(output_path),
        "filename": filename,
    }


# ============================================================
# FILE DOWNLOAD
# ============================================================

from fastapi.responses import FileResponse


@app.get("/audio/{filename}")
async def audio(filename: str):

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Audio not found"
        )

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )
