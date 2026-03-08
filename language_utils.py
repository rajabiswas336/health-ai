# language_utils.py
# Handles Bengali <-> English translation using deep-translator (free, no API key needed)
# Install: pip install deep-translator

from deep_translator import GoogleTranslator


def translate_to_english(text: str) -> str:
    """Translate Bengali (or any language) text to English."""
    try:
        return GoogleTranslator(source="bn", target="en").translate(text)
    except Exception as e:
        print(f"Translation error (bn→en): {e}")
        return text  # Fall back to original if translation fails


def translate_to_bengali(text: str) -> str:
    """Translate English text to Bengali."""
    try:
        return GoogleTranslator(source="en", target="bn").translate(text)
    except Exception as e:
        print(f"Translation error (en→bn): {e}")
        return text  # Fall back to original if translation fails


def get_whisper_language_code(language: str) -> str:
    """Returns Whisper language code based on UI selection."""
    return "bn" if language == "Bengali" else "en"


def get_gtts_language_code(language: str) -> str:
    """Returns gTTS language code based on UI selection."""
    return "bn" if language == "Bengali" else "en"
