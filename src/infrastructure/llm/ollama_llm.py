from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from src.domain.exceptions import OllamaUnavailableError
from src.domain.ports import LLMPort
from src.infrastructure.config_loader import load_config

_log = logging.getLogger(__name__)


class OllamaLLM(LLMPort):
    def __init__(self) -> None:
        cfg = load_config()["llm"]
        self._model: str = cfg["model"]
        self._base_url: str = cfg["base_url"]
        self._max_tokens: int = cfg["max_tokens"]
        # temperatura 0.1 reduz aleatoriedade na geração: em RAG o modelo deve
        # reproduzir fatos dos trechos recuperados, não criar conteúdo — valores
        # altos aumentam criatividade mas também a taxa de alucinação em QA factual
        self._temperature: float = cfg["temperature"]
        self._timeout: int = cfg["timeout_seconds"]
        self._max_retries: int = cfg["max_retries"]

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

        url = f"{self._base_url}/api/generate"
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            t0 = time.monotonic()
            try:
                req = urllib.request.Request(
                    url=url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    body = json.loads(resp.read().decode())

                elapsed = time.monotonic() - t0
                _log.info(
                    "ollama respondeu em %.2fs (tentativa %d/%d)",
                    elapsed, attempt, self._max_retries,
                )
                return body["response"]

            except (urllib.error.URLError, OSError) as exc:
                elapsed = time.monotonic() - t0
                last_exc = exc
                _log.warning(
                    "tentativa %d/%d falhou em %.2fs: %s",
                    attempt, self._max_retries, elapsed, exc,
                )
                if attempt < self._max_retries:
                    # backoff exponencial: 1s entre tentativas 1→2, 2s entre 2→3
                    # evita sobrecarregar o Ollama enquanto ainda está subindo
                    delay = 2 ** (attempt - 1)
                    time.sleep(delay)

        raise OllamaUnavailableError(
            f"Ollama não respondeu após {self._max_retries} tentativas"
        ) from last_exc
