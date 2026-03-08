# brain_of_the_doctor.py

import os
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing! Please set it in a .env file.")


def encode_image(image_path: str) -> str:
    """Convert an image file to a base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image_with_query(query: str, model: str, encoded_image: str) -> str:
    """
    Send an image + text query to the Groq vision LLM.
    Returns the model's text response.
    """
    client = Groq(api_key=GROQ_API_KEY)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content
