import os
import time
import uuid
import asyncio
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from voxcpm import VoxCPM


# ============================================================
# VoxCPM2 Kaggle API Server
# ============================================================

APP_NAME = "VoxCPM2 Telegram Voice API"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

VOICE_DIR = Path(
    "/kaggle/input/datasets/ohmyqueenmedusa/pyaesonevvipvoice"
)

OUTPUT_DIR = Path("/kaggle/working/voxcpm_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API key
# IMPORTANT:
# Change this to a strong random secret.
API_KEY = os.getenv(
    "VOXCPM_API_KEY",
    "CHANGE_THIS_TO_A_LONG_RANDOM_SECRET"
)

# Maximum text length
MAX_TEXT_LENGTH = 2000

# VoxCPM2 generation settings
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 30
RETRY_BADCASE = False
MAX_LEN = 2000


# ------------------------------------------------------------
# Voice references
# ------------------------------------------------------------

VOICES = {
    "Fangyung": VOICE_DIR / "Fangyung_vvipvoice.wav",
    "Htun": VOICE_DIR / "Htun_vvipvoice.wav",
    "Phyo": VOICE_DIR / "Phyo_vvipvoice.wav",
    "Pyae": VOICE_DIR / "Pyae_vvipvoice.wav",
}


# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model: Optional[VoxCPM] = None

# Prevent multiple GPU inference jobs running at the same time.
generation_lock = asyncio.Lock()


# ------------------------------------------------------------
# Request model
# ------------------------------------------------------------

class GenerateRequest(BaseModel):
    voice: str = Field(..., description="Voice name")
    text: str = Field(..., min_length=1, max_length=2000)


# ------------------------------------------------------------
# Startup
# ------------------------------------------------------------

@app.on_event("startup")
async def startup_event():

    print("=" * 70)
    print("🚀 VoxCPM2 Telegram Voice API")
    print("=" * 70)

    print("\n🔍 Checking CUDA...")

    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available.")
        raise RuntimeError(
            "CUDA GPU is required for VoxCPM2."
        )

    print("✅ CUDA:", torch.cuda.is_available())
    print("✅ GPU:", torch.cuda.get_device_name(0))

    print("\n🎙️ Checking voice files...")

    for name, path in VOICES.items():

        if path.exists():
            print(f"✅ {name:10} -> {path}")
        else:
            print(f"❌ {name:10} -> {path}")

    print("\n✅ API server started.")
    print("ℹ️ VoxCPM2 model will load on first /generate request.")
    print("=" * 70)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

def load_model():

    global model

    if model is not None:
        return model

    print("\n" + "=" * 70)
    print("🚀 Loading VoxCPM2...")
    print("=" * 70)

    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        device="cuda",
        optimize=True,
    )

    print("\n✅ VoxCPM2 loaded!")

    print(
        "Sample rate:",
        model.tts_model.sample_rate
    )

    if torch.cuda.is_available():

        print(
            "GPU memory allocated:",
            round(
                torch.cuda.memory_allocated() / 1024**3,
                2
            ),
            "GB"
        )

    print("=" * 70)

    return model


# ------------------------------------------------------------
# API key validation
# ------------------------------------------------------------

def check_api_key(
    authorization: Optional[str],
    x_api_key: Optional[str],
):

    # Accept:
    #
    # Authorization: Bearer YOUR_KEY
    #
    # OR
    #
    # X-API-Key: YOUR_KEY

    provided_key = None

    if authorization:

        if authorization.startswith("Bearer "):
            provided_key = authorization[7:].strip()

    if not provided_key and x_api_key:
        provided_key = x_api_key.strip()

    if not provided_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key."
        )

    if provided_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key."
        )


# ------------------------------------------------------------
# Root
# ------------------------------------------------------------

@app.get("/")
async def root():

    return {
        "status": "online",
        "service": APP_NAME,
        "version": "1.0.0",
        "cuda": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "voices": list(VOICES.keys()),
        "max_text_length": MAX_TEXT_LENGTH,
        "inference_timesteps": INFERENCE_TIMESTEPS,
    }


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


# ------------------------------------------------------------
# Voices
# ------------------------------------------------------------

@app.get("/voices")
async def voices():

    return {
        "voices": [
            {
                "name": name,
                "available": path.exists(),
            }
            for name, path in VOICES.items()
        ]
    }


# ------------------------------------------------------------
# Generate
# ------------------------------------------------------------

@app.post("/generate")
async def generate(
    request: GenerateRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    check_api_key(
        authorization,
        x_api_key,
    )

    # --------------------------------------------------------
    # Validate voice
    # --------------------------------------------------------

    voice_name = request.voice.strip()

    if voice_name not in VOICES:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unknown voice.",
                "available_voices": list(VOICES.keys()),
            },
        )

    reference = VOICES[voice_name]

    if not reference.exists():

        raise HTTPException(
            status_code=500,
            detail=f"Reference voice not found: {reference}",
        )

    # --------------------------------------------------------
    # Validate text
    # --------------------------------------------------------

    text = request.text.strip()

    if not text:

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    if len(text) > MAX_TEXT_LENGTH:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Text is too long. "
                f"Maximum is {MAX_TEXT_LENGTH} characters."
            ),
        )

    # --------------------------------------------------------
    # Queue GPU generation
    # --------------------------------------------------------

    async with generation_lock:

        generation_id = uuid.uuid4().hex

        output_file = (
            OUTPUT_DIR
            / f"{voice_name}_{generation_id}.wav"
        )

        print("\n" + "=" * 70)
        print("🎙️ NEW GENERATION")
        print("=" * 70)

        print("Voice:", voice_name)
        print("Text length:", len(text))
        print("Reference:", reference)
        print("Output:", output_file)

        print("\n⚙️ Settings")
        print("CFG:", CFG_VALUE)
        print("Inference steps:", INFERENCE_TIMESTEPS)
        print("Retry badcase:", RETRY_BADCASE)
        print("Max length:", MAX_LEN)

        # ----------------------------------------------------
        # Load model only when needed
        # ----------------------------------------------------

        tts_model = load_model()

        # ----------------------------------------------------
        # GPU inference
        # ----------------------------------------------------

        print("\n🚀 Generating...")

        start = time.perf_counter()

        try:

            wav = tts_model.generate(
                text=text,
                reference_wav_path=str(reference),

                # Keep user's tested settings
                cfg_value=CFG_VALUE,
                inference_timesteps=INFERENCE_TIMESTEPS,
                retry_badcase=RETRY_BADCASE,
                max_len=MAX_LEN,
            )

        except Exception as e:

            print("\n❌ Generation error:")
            print(repr(e))

            if output_file.exists():
                try:
                    output_file.unlink()
                except Exception:
                    pass

            raise HTTPException(
                status_code=500,
                detail=f"Voice generation failed: {str(e)}",
            )

        elapsed = time.perf_counter() - start

        # ----------------------------------------------------
        # Save WAV
        # ----------------------------------------------------

        try:

            sf.write(
                str(output_file),
                wav,
                tts_model.tts_model.sample_rate,
            )

        except Exception as e:

            print("\n❌ WAV save error:")
            print(repr(e))

            raise HTTPException(
                status_code=500,
                detail=f"Failed to save WAV: {str(e)}",
            )

        # ----------------------------------------------------
        # Calculate statistics
        # ----------------------------------------------------

        sample_rate = tts_model.tts_model.sample_rate

        duration = (
            len(wav) / sample_rate
        )

        rtf = (
            elapsed / duration
            if duration > 0
            else 0
        )

        vram = None

        if torch.cuda.is_available():

            vram = round(
                torch.cuda.memory_allocated() / 1024**3,
                2
            )

        print("\n" + "=" * 70)
        print("✅ GENERATION COMPLETE")
        print("=" * 70)

        print("Voice:", voice_name)
        print("Duration:", round(duration, 2), "sec")
        print("Generation:", round(elapsed, 2), "sec")
        print("RTF:", round(rtf, 3))

        if vram is not None:
            print("VRAM:", vram, "GB")

        print("File:", output_file)

        print("=" * 70)

        # ----------------------------------------------------
        # Return metadata + download URL
        # ----------------------------------------------------

        return {
            "success": True,
            "generation_id": generation_id,
            "voice": voice_name,
            "duration": round(duration, 2),
            "generation_time": round(elapsed, 2),
            "rtf": round(rtf, 3),
            "sample_rate": sample_rate,
            "vram_gb": vram,
            "filename": output_file.name,
            "download_url": (
                f"/audio/{output_file.name}"
            ),
        }


# ------------------------------------------------------------
# Audio download
# ------------------------------------------------------------

@app.get("/audio/{filename}")
async def audio(
    filename: str,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):

    # Authentication
    check_api_key(
        authorization,
        x_api_key,
    )

    # Security:
    # Only allow the filename itself.
    safe_name = Path(filename).name

    if safe_name != filename:

        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    file_path = OUTPUT_DIR / safe_name

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Audio file not found.",
        )

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=safe_name,
    )


# ------------------------------------------------------------
# Run server
# ------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    print("\n🚀 Starting FastAPI server...")
    print("Host: 0.0.0.0")
    print("Port: 8000")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
