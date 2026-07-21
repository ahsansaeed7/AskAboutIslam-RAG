import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import requests
from groq import Groq
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER


def call_llm(prompt: str) -> str:
    """
    Single entry point for generation. Routes to Ollama (local) or Groq (cloud)
    based on the LLM_PROVIDER setting in config.py / .env.
    """
    if LLM_PROVIDER == "groq":
        return _call_groq(prompt)
    else:
        return _call_ollama(prompt)


def _call_ollama(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]


def _call_groq(prompt: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Quick test
    print(f"Using provider: {LLM_PROVIDER}")
    result = call_llm("Say hello in one sentence.")
    print(result)