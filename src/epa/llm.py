"""Minimal client for any OpenAI-compatible chat-completions endpoint.

Works unchanged with a local Ollama server (default, free, offline), vLLM,
LM Studio, or a hosted API. Configure with environment variables:

    EPA_LLM_BASE_URL   default http://localhost:11434/v1   (Ollama)
    EPA_LLM_MODEL      default qwen2.5:7b                  (tool calling)
    EPA_VLM_MODEL      default qwen2.5vl:3b                (vision)
    EPA_LLM_API_KEY    default "ollama" (ignored by Ollama)
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx


class ChatClient:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 vlm_model: str | None = None, api_key: str | None = None,
                 timeout: float = 180.0):
        self.base_url = (base_url or os.getenv("EPA_LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.getenv("EPA_LLM_MODEL", "qwen2.5:7b")
        self.vlm_model = vlm_model or os.getenv("EPA_VLM_MODEL", "qwen2.5vl:3b")
        self.api_key = api_key or os.getenv("EPA_LLM_API_KEY", "ollama")
        self.http = httpx.Client(timeout=timeout)

    def _post(self, payload: dict) -> dict:
        r = self.http.post(f"{self.base_url}/chat/completions", json=payload,
                           headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code != 200:
            raise RuntimeError(f"LLM endpoint returned {r.status_code}: {r.text[:300]}")
        return r.json()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Return the assistant message (may contain `tool_calls`)."""
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
        return self._post(payload)["choices"][0]["message"]

    def describe_image(self, image_path: str | Path, question: str) -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        msg = [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}]
        out = self._post({"model": self.vlm_model, "messages": msg, "temperature": 0})
        return out["choices"][0]["message"]["content"]
