# mcp_ai/providers/ollama_client.py
from __future__ import annotations
import requests, json, re, time

class OllamaClient:
    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "mistral:7b-instruct", timeout_s: int = 30):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.endpoint}{path}"
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def generate_json(self, system: str, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> dict:
        # Use /api/generate (non-stream) for simplicity
        payload = {
            "model": self.model,
            "prompt": f"{system}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        data = self._post("/api/generate", payload)
        text = data.get("response","").strip()
        # Extract JSON payload from response (model may wrap text)
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if m:
            text = m.group(0)
        try:
            return json.loads(text)
        except Exception:
            return {"error":"invalid_json", "raw": text}
