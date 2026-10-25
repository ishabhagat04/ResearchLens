"""Thin Ollama client."""
import requests

OLLAMA_URL = "http://localhost:11434"

def generate(prompt, model="qwen2.5:1.5b", temperature=0.1):
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature},
    }, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

def health():
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False