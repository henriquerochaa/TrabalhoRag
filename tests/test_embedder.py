from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder


def _make_normalized(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    raw = rng.random((n, dim)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


@pytest.fixture()
def embedder(monkeypatch):
    cfg = {
        "paths": {"models": "models/"},
        "embedding": {"model_name": "intfloat/multilingual-e5-large", "batch_size": 32},
    }
    monkeypatch.setattr(
        "src.infrastructure.embeddings.sentence_transformer_embedder.load_config",
        lambda: cfg,
    )
    mock_model = MagicMock()
    # simula encode retornando vetores L2-normalizados — mesmo contrato do modelo real
    mock_model.encode.side_effect = lambda texts, **kw: _make_normalized(len(texts))
    with patch(
        "src.infrastructure.embeddings.sentence_transformer_embedder.SentenceTransformer",
        return_value=mock_model,
    ):
        yield SentenceTransformerEmbedder()


# ---------------------------------------------------------------------------
# Formato de saída
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_embed_returns_2d_ndarray(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed(["texto de exemplo"])
        assert isinstance(result, np.ndarray) and result.ndim == 2

    def test_embed_queries_returns_2d_ndarray(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed_queries(["query de exemplo"])
        assert isinstance(result, np.ndarray) and result.ndim == 2

    def test_embed_first_dim_matches_input_count(self, embedder: SentenceTransformerEmbedder) -> None:
        texts = ["texto a", "texto b", "texto c"]
        result = embedder.embed(texts)
        assert result.shape[0] == len(texts)

    def test_embed_queries_first_dim_matches_input_count(self, embedder: SentenceTransformerEmbedder) -> None:
        queries = ["query 1", "query 2"]
        result = embedder.embed_queries(queries)
        assert result.shape[0] == len(queries)

    def test_embed_single_text_returns_one_row(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed(["único texto"])
        assert result.shape[0] == 1

    def test_embed_queries_single_query_returns_one_row(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed_queries(["única query"])
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# Normalização L2
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_embed_vectors_are_l2_normalized(self, embedder: SentenceTransformerEmbedder) -> None:
        # normalização L2 é pré-requisito para que produto interno == similaridade cosseno
        result = embedder.embed(["texto normalizado"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_queries_vectors_are_l2_normalized(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed_queries(["query normalizada"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Prefixos passage:/query: (requisito do multilingual-e5)
# ---------------------------------------------------------------------------

class TestPrefixes:
    def test_embed_prepends_passage_prefix(self, embedder: SentenceTransformerEmbedder) -> None:
        # multilingual-e5 requer "passage:" em chunks para distinguir de queries
        embedder.embed(["meu texto"])
        call_texts = embedder._model.encode.call_args[0][0]
        assert call_texts[0].startswith("passage: ")

    def test_embed_queries_prepends_query_prefix(self, embedder: SentenceTransformerEmbedder) -> None:
        # multilingual-e5 requer "query:" em queries de busca
        embedder.embed_queries(["minha query"])
        call_texts = embedder._model.encode.call_args[0][0]
        assert call_texts[0].startswith("query: ")

    def test_embed_and_embed_queries_use_distinct_prefixes(self, embedder: SentenceTransformerEmbedder) -> None:
        embedder.embed(["texto"])
        passage_text = embedder._model.encode.call_args[0][0][0]
        embedder.embed_queries(["texto"])
        query_text = embedder._model.encode.call_args[0][0][0]
        # prefixos distintos são obrigatórios — unificá-los quebraria a semântica do e5
        assert passage_text != query_text
