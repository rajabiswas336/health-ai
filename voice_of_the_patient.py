# voice_of_the_patient.py

from dotenv import load_dotenv
load_dotenv()

import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """Record audio from the microphone and save it as an MP3 file."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while recording: {e}")


def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY, language="en"):
    """
    Transcribe audio using Groq Whisper.
    - language="en" for English
    - language="bn" for Bengali
    BUG FIX: Removed duplicate empty function definition.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing! Please set it in your .env file.")

    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language=language   # Now supports "bn" for Bengali
        )

    if hasattr(transcription, "text"):
        return transcription.text
    elif isinstance(transcription, dict) and "text" in transcription:
        return transcription["text"]
    else:
        return None
