# language_utils.py — v2.0
# Gap 5 fix: auto language detection + Hindi, Odia, Assamese support
# All free — no API key needed
# pip install deep-translator langdetect

from deep_translator import GoogleTranslator

# ── Language config — single source of truth ─────────────────────────────────
# Maps UI display name → (ISO code, Whisper code, edge-tts voice, gTTS code)
LANGUAGE_CONFIG = {
    "English":  ("en", "en", "en-US-JennyNeural",        "en"),
    "Bengali":  ("bn", "bn", "bn-IN-TanishaaNeural",      "bn"),
    "Hindi":    ("hi", "hi", "hi-IN-SwaraNeural",         "hi"),
    "Odia":     ("or", "or", "or-IN-SubhasiniNeural",     "or"),
    "Assamese": ("as", "as", "as-IN-YashicaNeural",       "as"),
}

# Readable names shown next to auto-detected language in UI
LANGUAGE_DISPLAY = {
    "en": "English",
    "bn": "Bengali",
    "hi": "Hindi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
}


def auto_detect_language(text: str) -> str:
    """
    Auto-detect language from text. Returns ISO 639-1 code e.g. 'bn', 'hi', 'en'.
    Uses langdetect (free, offline, no API key).
    Falls back to 'en' on any error.
    """
    if not text or len(text.strip()) < 3:
        return "en"
    try:
        from langdetect import detect
        return detect(text)
    except Exception as e:
        print(f"[langdetect] error: {e}")
        return "en"


def get_language_display_name(lang_code: str) -> str:
    """Returns human-readable name for a language code."""
    return LANGUAGE_DISPLAY.get(lang_code, lang_code.upper())


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """
    Translate any language text to English.
    source_lang: ISO code e.g. 'bn', 'hi', or 'auto' for auto-detect.
    """
    if not text or not text.strip():
        return text
    if source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception as e:
        print(f"Translation error ({source_lang}→en): {e}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate English text to any target language.
    target_lang: ISO code e.g. 'bn', 'hi', 'or', 'as'.
    """
    if not text or not text.strip():
        return text
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error (en→{target_lang}): {e}")
        return text


# ── Legacy wrappers — keeps existing app.py calls working ────────────────────
def translate_to_bengali(text: str) -> str:
    return translate_from_english(text, "bn")


def get_whisper_language_code(language: str) -> str:
    """Returns Whisper STT language code for a UI display name."""
    return LANGUAGE_CONFIG.get(language, ("en", "en", "", "en"))[1]


def get_gtts_language_code(language: str) -> str:
    """Returns gTTS language code for a UI display name."""
    return LANGUAGE_CONFIG.get(language, ("en", "en", "", "en"))[3]


def get_edge_tts_voice(language: str) -> str:
    """Returns edge-tts neural voice name for a UI display name."""
    return LANGUAGE_CONFIG.get(language, ("en", "en", "en-US-JennyNeural", "en"))[2]


def get_lang_iso(language: str) -> str:
    """Returns ISO code for a UI display name e.g. 'Bengali' → 'bn'."""
    return LANGUAGE_CONFIG.get(language, ("en",))[0]