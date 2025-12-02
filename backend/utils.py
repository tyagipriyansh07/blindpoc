from gtts import gTTS
import uuid
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_groq_llm(user_text, scene, rule_output):
    prompt = f"""
    You are a real-time assistant for a blind person.
    KEEP RESPONSES ULTRA SHORT (max 8–10 words).

    Scene: {scene}
    Risk analysis: {rule_output}

    Respond like:
    - “Car approaching from left”
    - “Path clear ahead”
    - “Person close in front”
    """

    response = client.chat.completions.create(
        # model="llama-3.2-3b-preview",
        model="llama-3.1-8b-instant",   # ← UPDATED MODEL
        messages=[{"role": "user", "content": prompt}]
    )

    # return response.choices[0].message["content"]
    return response.choices[0].message.content
