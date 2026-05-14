from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.generate_answer import GenerateAnswer, _UNAVAILABLE_TEXT
from src.application.prompt_builder import PromptBuilder
from src.application.search_chunks import SearchChunks
from src.domain.entities import Chunk, SearchResult
from src.domain.exceptions import OllamaUnavailableError
from src.domain.ports import EmbedderPort, LLMPort, RerankerPort, VectorStorePort
from src.infrastructure.llm.ollama_llm import OllamaLLM


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_response(text: str) -> MagicMock:
    """Mock de context manager que simula resposta bem-sucedida do Ollama."""
    body = json.dumps({"response": text}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _url_error(msg: str = "connection refused") -> urllib.error.URLError:
    return urllib.error.URLError(msg)


@pytest.fixture()
def llm(monkeypatch) -> OllamaLLM:
    cfg = {
        "llm": {
            "model": "gemma2:9b",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 512,
            "timeout_seconds": 5,
            # 3 tentativas: equilibra disponibilidade vs latência total.
            # backoff 1s→2s→falha dá ~3s de janela para recuperação transitória
            "max_retries": 3,
            # context_window obrigatório: OllamaLLM envia num_ctx para evitar truncagem
            "context_window": 8192,
        }
    }
    monkeypatch.setattr("src.infrastructure.llm.ollama_llm.load_config", lambda: cfg)
    return OllamaLLM()


# ---------------------------------------------------------------------------
# Retry e backoff exponencial
# ---------------------------------------------------------------------------

class TestRetry:
    def test_timeout_triggers_retry(self, llm: OllamaLLM) -> None:
        # primeira chamada falha com URLError (simula timeout de rede);
        # segunda bem-sucedida — verifica que o retry ocorreu de fato
        with patch("urllib.request.urlopen", side_effect=[
            _url_error("timed out"),
            _make_response("Resposta válida após retry."),
        ]) as mock_open, patch("time.sleep"):
            result = llm.generate("Qual o PIB do Paraná?")

        assert result == "Resposta válida após retry."
        assert mock_open.call_count == 2, (
            "esperado 2 chamadas: 1 falha + 1 sucesso"
        )

    def test_success_on_second_attempt_returns_normal_response(self, llm: OllamaLLM) -> None:
        # verifica que o valor retornado é o da segunda tentativa, não um fallback
        with patch("urllib.request.urlopen", side_effect=[
            _url_error("connection refused"),
            _make_response("O agronegócio representa 30% do PIB paranaense."),
        ]), patch("time.sleep") as mock_sleep:
            result = llm.generate("fale sobre agronegócio no Paraná")

        assert result == "O agronegócio representa 30% do PIB paranaense."
        # exatamente 1 sleep entre tentativas 1 e 2 (backoff 2^0 = 1s)
        mock_sleep.assert_called_once_with(1)

    def test_three_failures_raises_ollama_unavailable(self, llm: OllamaLLM) -> None:
        # após esgotar todas as tentativas, OllamaUnavailableError deve ser lançada
        with patch("urllib.request.urlopen", side_effect=[
            _url_error("timeout"),
            _url_error("timeout"),
            _url_error("timeout"),
        ]) as mock_open, patch("time.sleep"):
            with pytest.raises(OllamaUnavailableError):
                llm.generate("pergunta qualquer")

        assert mock_open.call_count == 3, (
            "deve ter tentado exatamente max_retries=3 vezes antes de desistir"
        )

    def test_backoff_delays_between_attempts(self, llm: OllamaLLM) -> None:
        # verifica sequência exata de sleeps: 1s (2^0) e 2s (2^1)
        # a terceira falha não gera sleep — não faz sentido esperar antes de lançar
        with patch("urllib.request.urlopen", side_effect=[
            _url_error(), _url_error(), _url_error()
        ]), patch("time.sleep") as mock_sleep:
            with pytest.raises(OllamaUnavailableError):
                llm.generate("prompt")

        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [1, 2], (
            f"backoff esperado [1, 2] segundos; recebido {calls}"
        )

    def test_no_sleep_on_first_success(self, llm: OllamaLLM) -> None:
        # sucesso imediato não deve gerar nenhum sleep
        with patch("urllib.request.urlopen", return_value=_make_response("OK")), \
             patch("time.sleep") as mock_sleep:
            llm.generate("prompt")

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback em GenerateAnswer quando Ollama indisponível
# ---------------------------------------------------------------------------

def _chunk(cid: str, text: str = "Dado econômico sobre o Paraná.") -> Chunk:
    return Chunk(id=cid, document_id="doc", text=text, page=1, section="", position=0)


class _StubEmbedder(EmbedderPort):
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.eye(len(texts), dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


class _PassthroughReranker(RerankerPort):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return chunks


class _UnavailableLLM(LLMPort):
    # sempre lança OllamaUnavailableError — simula serviço offline
    def generate(self, prompt: str) -> str:
        raise OllamaUnavailableError("ollama não está respondendo")


def _make_store_with(pairs: list[tuple[Chunk, float]]) -> VectorStorePort:
    class _Store(VectorStorePort):
        def add(self, c, e): pass
        def search(self, emb, top_k): return [SearchResult(c, s) for c, s in pairs]
        def save(self, p): pass
        def load(self, p): pass
    return _Store()


class TestGenerateAnswerFallback:
    def _make_ga(self, pairs: list[tuple[Chunk, float]]) -> GenerateAnswer:
        search = SearchChunks(
            _StubEmbedder(),
            _make_store_with(pairs),
            _PassthroughReranker(),
        )
        return GenerateAnswer(search, PromptBuilder(), _UnavailableLLM())

    def test_returns_unavailable_text(self) -> None:
        # quando OllamaUnavailableError é capturada, o texto padrão deve ser retornado
        ans = self._make_ga([(_chunk("c1"), 0.90)]).execute("Qual o PIB do Paraná?")
        assert ans.text == _UNAVAILABLE_TEXT

    def test_out_of_scope_is_false(self) -> None:
        # a pergunta tem contexto nos docs — out_of_scope=False mesmo sem LLM
        ans = self._make_ga([(_chunk("c1"), 0.90)]).execute("Qual o PIB do Paraná?")
        assert ans.out_of_scope is False

    def test_prompt_used_is_filled(self) -> None:
        # prompt_used preenchido: usuário sabe o que seria enviado à LLM
        ans = self._make_ga([(_chunk("c1"), 0.90)]).execute("Qual o PIB do Paraná?")
        assert ans.prompt_used != ""

    def test_sources_are_filled(self) -> None:
        # sources preenchidos: usuário pode consultar as fontes mesmo sem resposta
        c = _chunk("c1", "O PIB do Paraná cresceu 4% em 2024.")
        ans = self._make_ga([(c, 0.90)]).execute("Qual o PIB do Paraná?")
        assert ans.sources, "sources deve conter os chunks usados no prompt"
        assert any(s.id == "c1" for s in ans.sources)

    def test_unavailable_does_not_mask_out_of_scope(self) -> None:
        # query fora do escopo deve retornar out_of_scope=True, não _UNAVAILABLE_TEXT —
        # a LLM nunca é chamada nesse caminho, então OllamaUnavailableError não ocorre
        ans = self._make_ga([(_chunk("low"), 0.10)]).execute("Capital da Alemanha?")
        assert ans.out_of_scope is True
        assert ans.text != _UNAVAILABLE_TEXT


# ---------------------------------------------------------------------------
# Respostas malformadas do Ollama
# ---------------------------------------------------------------------------

class TestMalformedResponse:
    def test_json_missing_response_key_raises(self, llm: OllamaLLM) -> None:
        # Ollama retorna JSON válido mas sem a chave "response" — KeyError deve propagar
        # sem silêncio: melhor falha explícita do que retornar string vazia
        body = json.dumps({"model": "llama3.2:3b", "done": True}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp), patch("time.sleep"):
            with pytest.raises((KeyError, OllamaUnavailableError)):
                llm.generate("prompt")

    def test_invalid_json_raises(self, llm: OllamaLLM) -> None:
        # corpo não é JSON válido — json.JSONDecodeError deve propagar sem tentar retry
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json {{{"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp), patch("time.sleep"):
            with pytest.raises((json.JSONDecodeError, ValueError, OllamaUnavailableError)):
                llm.generate("prompt")

    def test_successful_response_returns_string(self, llm: OllamaLLM) -> None:
        # caminho feliz: resposta completa retorna str com o texto gerado
        with patch("urllib.request.urlopen", return_value=_make_response("Resposta final.")), \
             patch("time.sleep"):
            result = llm.generate("prompt")
        assert isinstance(result, str) and result == "Resposta final."
