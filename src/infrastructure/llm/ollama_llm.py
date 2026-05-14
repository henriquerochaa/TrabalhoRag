from __future__ import annotations

import json
import logging
import os
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
        # OLLAMA_BASE_URL sobrescreve config.yaml dentro do Docker: localhost:11434
        # apontaria para o próprio container; host.docker.internal alcança o host real.
        # Em execução nativa (sem Docker) a variável não é definida e usa o config.yaml.
        self._base_url: str = os.environ.get("OLLAMA_BASE_URL", cfg["base_url"])
        self._max_tokens: int = cfg["max_tokens"]
        # temperatura 0.1 reduz aleatoriedade na geração: em RAG o modelo deve
        # reproduzir fatos dos trechos recuperados, não criar conteúdo — valores
        # altos aumentam criatividade mas também a taxa de alucinação em QA factual
        self._temperature: float = cfg["temperature"]
        self._timeout: int = cfg["timeout_seconds"]
        self._max_retries: int = cfg["max_retries"]
        # num_ctx precisa ser explicitado: Ollama usa 2048 por padrão para llama3.2:3b,
        # mas nosso prompt com chunks do PDF pode ultrapassar esse limite → HTTP 500.
        self._context_window: int = cfg["context_window"]

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
                "num_ctx": self._context_window,
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
