# backend/utils.py

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment or .env file")

client = Groq(api_key=API_KEY)

def call_groq_llm(user_text, scene, rule_output):
    """
    Call Groq Llama 3.1 8B to generate a short voice-friendly reply
    based on current scene, rule output, and user's question.
    """
    prompt = f"""
    You are a real-time assistant for a visually impaired user.
    You must be calm, concise, and safety-first.

    Scene detections (list of objects): {scene}
    Rule-based safety analysis: {rule_output}
    User asked: "{user_text}"

    Respond with ONE short sentence, suitable to be spoken aloud.
    Do NOT add extra explanations.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
