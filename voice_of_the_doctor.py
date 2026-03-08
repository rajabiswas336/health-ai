# voice_of_the_doctor.py — v2.0
# Bengali: Microsoft edge-tts neural voice (much better than gTTS)
# English: ElevenLabs (high quality) or edge-tts fallback

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import subprocess
import platform
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"

# ── Microsoft Neural Bengali Voices (edge-tts) ────────────────────────────────
# bn-BD-NabanitaNeural  → Female, Bangladesh Bengali ✓ recommended
# bn-BD-PradeepNeural   → Male,   Bangladesh Bengali
# bn-IN-TanishaaNeural  → Female, India Bengali
# bn-IN-BashkarNeural   → Male,   India Bengali
EDGE_BENGALI_VOICE = "bn-BD-NabanitaNeural"
EDGE_ENGLISH_VOICE = "en-US-JennyNeural"


def list_voices():
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    voices = client.voices.get_all()
    print("Available ElevenLabs Voices:")
    for v in voices.voices:
        print(f"  Name: {v.name}, ID: {v.voice_id}")


def _autoplay(filepath):
    """Local playback only — not used on Streamlit Cloud."""
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(["afplay", filepath])
        elif os_name == "Windows":
            fixed = filepath.replace(".mp3", "_fixed.mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", filepath,
                 "-ar", "44100", "-ac", "2", "-b:a", "128k", fixed],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(["ffplay", "-nodisp", "-autoexit", fixed])
        elif os_name == "Linux":
            subprocess.run(["mpg123", filepath])
    except Exception as e:
        print(f"Autoplay error (safe to ignore on cloud): {e}")


# ── edge-tts (Microsoft Neural TTS) ──────────────────────────────────────────
async def _edge_tts_async(text: str, voice: str, output_filepath: str) -> bool:
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_filepath)
        return True
    except ImportError:
        print("edge-tts not installed. Run: pip install edge-tts")
        return False
    except Exception as e:
        print(f"edge-tts error: {e}")
        return False


def text_to_speech_with_edge(text: str, output_filepath: str,
                              language: str = "bn",
                              voice_id: str = None,
                              autoplay: bool = False):
    """
    Microsoft Neural TTS via edge-tts — free, no API key, high quality.
      language='bn' → Bengali neural voice (default: NabanitaNeural)
      language='en' → English neural voice (JennyNeural)
      voice_id       → override with any specific edge-tts voice name
    Falls back to gTTS if edge-tts is not installed.
    """
    if voice_id:
        voice = voice_id
    else:
        voice = EDGE_BENGALI_VOICE if language == "bn" else EDGE_ENGLISH_VOICE
    try:
        loop = asyncio.new_event_loop()
        success = loop.run_until_complete(_edge_tts_async(text, voice, output_filepath))
        loop.close()
    except Exception as e:
        print(f"edge-tts loop error: {e}")
        success = False

    if not success:
        # Graceful fallback to gTTS
        print("Falling back to gTTS...")
        _gtts_save(text, output_filepath, language)

    if autoplay:
        _autoplay(output_filepath)
    return output_filepath


# ── gTTS (Google TTS — fallback) ─────────────────────────────────────────────
def _gtts_save(text: str, filepath: str, language: str = "en"):
    gTTS(text=text, lang=language, slow=False).save(filepath)


def text_to_speech_with_gtts(input_text, output_filepath,
                              language="en", autoplay=False):
    _gtts_save(input_text, output_filepath, language)
    if autoplay:
        _autoplay(output_filepath)
    return output_filepath


# ── ElevenLabs (English, highest quality) ────────────────────────────────────
def text_to_speech_with_elevenlabs(input_text, output_filepath, autoplay=False):
    # Re-read key at call time so Streamlit Cloud secrets are picked up
    _key = os.environ.get("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)
    if not _key:
        raise ValueError("ELEVENLABS_API_KEY not set")
    client = ElevenLabs(api_key=_key)
    audio = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        model_id="eleven_turbo_v2",
        text=input_text,
    )
    elevenlabs.save(audio, output_filepath)
    if autoplay:
        _autoplay(output_filepath)
    return output_filepath
