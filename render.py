import os
import tempfile
from pathlib import Path

import aiohttp

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    FSInputFile,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")

KAGGLE_API_URL = os.getenv(
    "KAGGLE_API_URL"
)

KAGGLE_API_KEY = os.getenv(
    "KAGGLE_API_KEY"
)

MAX_LEN = 2000


if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN is missing"
    )

if not KAGGLE_API_URL:
    raise RuntimeError(
        "KAGGLE_API_URL is missing"
    )

if not KAGGLE_API_KEY:
    raise RuntimeError(
        "KAGGLE_API_KEY is missing"
    )


# ============================================================
# BOT
# ============================================================

bot = Bot(
    token=BOT_TOKEN
)

dp = Dispatcher()


# ============================================================
# USER STATE
# ============================================================

user_voice = {}


# ============================================================
# VOICE KEYBOARD
# ============================================================

def voice_keyboard():

    builder = InlineKeyboardBuilder()

    builder.button(
        text="🎙 Phyo",
        callback_data="voice:Phyo"
    )

    builder.button(
        text="🎙 Htun",
        callback_data="voice:Htun"
    )

    builder.button(
        text="🎙 Pyae",
        callback_data="voice:Pyae"
    )

    builder.button(
        text="🎙 Fangyung",
        callback_data="voice:Fangyung"
    )

    builder.adjust(2)

    return builder.as_markup()


# ============================================================
# START
# ============================================================

@dp.message(CommandStart())
async def start(message: Message):

    user_voice[message.from_user.id] = "Phyo"

    await message.answer(
        "🎙 <b>VoxCPM2 Voice Bot</b>\n\n"
        "Voice ကိုရွေးပါ။\n\n"
        "ရွေးပြီးရင် စာသားပို့လိုက်ပါ။\n"
        "Maximum <b>2000 characters</b> ပါ။",
        reply_markup=voice_keyboard(),
        parse_mode="HTML"
    )


# ============================================================
# VOICE SELECT
# ============================================================

@dp.callback_query(F.data.startswith("voice:"))
async def select_voice(
    callback: CallbackQuery
):

    voice = callback.data.split(
        ":",
        1
    )[1]

    user_voice[
        callback.from_user.id
    ] = voice

    await callback.answer(
        f"{voice} selected"
    )

    await callback.message.edit_text(
        f"🎙 <b>Voice:</b> {voice}\n\n"
        f"စာသားကို ပို့လိုက်ပါ။\n"
        f"Maximum <b>{MAX_LEN}</b> characters.",
        reply_markup=voice_keyboard(),
        parse_mode="HTML"
    )


# ============================================================
# GENERATE
# ============================================================

async def generate_voice(
    text: str,
    voice: str
):

    headers = {
        "Authorization":
            f"Bearer {KAGGLE_API_KEY}",

        "X-API-Key":
            KAGGLE_API_KEY,

        "Content-Type":
            "application/json",
    }

    payload = {
        "text": text,
        "voice": voice,
    }

    timeout = aiohttp.ClientTimeout(
        total=900
    )

    async with aiohttp.ClientSession(
        timeout=timeout
    ) as session:

        async with session.post(
            f"{KAGGLE_API_URL}/generate",
            json=payload,
            headers=headers
        ) as response:

            data = await response.json()

            if response.status != 200:

                raise RuntimeError(
                    data.get(
                        "detail",
                        "Kaggle API error"
                    )
                )

            return data


# ============================================================
# TEXT HANDLER
# ============================================================

@dp.message(F.text)
async def text_handler(
    message: Message
):

    text = message.text.strip()

    if not text:
        return

    if len(text) > MAX_LEN:

        await message.answer(
            f"❌ စာသားက {MAX_LEN} characters "
            f"ထက်မကျော်ရပါဘူး။\n\n"
            f"လက်ရှိ: {len(text)}"
        )

        return

    voice = user_voice.get(
        message.from_user.id,
        "Phyo"
    )

    status = await message.answer(
        "⏳ <b>Generating...</b>\n\n"
        f"🎙 Voice: <b>{voice}</b>\n"
        f"📝 Characters: <b>{len(text)}</b>\n"
        f"⚙️ Steps: <b>30</b>\n\n"
        "GPU inference လုပ်နေပါတယ်...",
        parse_mode="HTML"
    )

    temp_file = None

    try:

        result = await generate_voice(
            text=text,
            voice=voice
        )

        if not result.get("success"):
            raise RuntimeError(
                "Generation failed"
            )

        filename = result["filename"]

        # ----------------------------------------------------
        # Download generated audio
        # ----------------------------------------------------

        headers = {
            "Authorization":
                f"Bearer {KAGGLE_API_KEY}",

            "X-API-Key":
                KAGGLE_API_KEY,
        }

        timeout = aiohttp.ClientTimeout(
            total=300
        )

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.get(
                f"{KAGGLE_API_URL}/audio/{filename}",
                headers=headers
            ) as response:

                if response.status != 200:

                    raise RuntimeError(
                        "Audio download failed"
                    )

                audio_data = await response.read()

        # ----------------------------------------------------
        # Save temporary WAV
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:

            f.write(audio_data)

            temp_file = f.name

        # ----------------------------------------------------
        # Send Telegram
        # ----------------------------------------------------

        await status.edit_text(
            "✅ <b>Voice Generated</b>\n\n"
            f"🎙 Voice: <b>{voice}</b>\n"
            f"📝 Characters: <b>{len(text)}</b>\n"
            f"⚙️ Steps: <b>30</b>\n"
            f"⏱ Inference: "
            f"<b>{result.get('inference_time', 0)}s</b>",
            parse_mode="HTML"
        )

        audio = FSInputFile(
            temp_file,
            filename=f"{voice}.wav"
        )

        await message.answer_audio(
            audio=audio,
            title=f"VoxCPM2 - {voice}",
            performer="VoxCPM2"
        )

    except Exception as e:

        print(
            "Generation error:",
            repr(e)
        )

        await status.edit_text(
            "❌ <b>Generation Failed</b>\n\n"
            f"<code>{str(e)}</code>",
            parse_mode="HTML"
        )

    finally:

        if temp_file:

            try:
                os.remove(temp_file)
            except OSError:
                pass


# ============================================================
# RUN
# ============================================================

async def main():

    print(
        "🚀 VoxCPM2 Telegram Bot starting..."
    )

    await dp.start_polling(
        bot
    )


if __name__ == "__main__":

    import asyncio

    asyncio.run(
        main()
    )
