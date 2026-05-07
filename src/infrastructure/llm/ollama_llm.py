from __future__ import annotations

import json
import urllib.request

from src.domain.ports import LLMPort
from src.infrastructure.config_loader import load_config


class OllamaLLM(LLMPort):
    def __init__(self) -> None:
        cfg = load_config()["llm"]
        self._model = cfg["model"]
        self._base_url = cfg["base_url"]
        self._max_tokens = cfg["max_tokens"]
        # temperatura 0.1 reduz aleatoriedade na geração: em RAG o modelo deve
        # reproduzir fatos dos trechos recuperados, não criar conteúdo — valores
        # altos aumentam criatividade mas também a taxa de alucinação em QA factual
        self._temperature = cfg["temperature"]

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }).encode()

        req = urllib.request.Request(
            url=f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())

        return body["response"]
