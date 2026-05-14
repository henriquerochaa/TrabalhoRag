from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.entities import Chunk
from src.infrastructure.reranking.cross_encoder_reranker import CrossEncoderReranker


def _chunk(cid: str, text: str = "texto de exemplo sobre o Paraná") -> Chunk:
    return Chunk(id=cid, document_id="doc", text=text, page=1, section="", position=0)


@pytest.fixture()
def reranker(monkeypatch):
    cfg = {
        "paths": {"models": "models/"},
        "reranker": {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    }
    monkeypatch.setattr(
        "src.infrastructure.reranking.cross_encoder_reranker.load_config",
        lambda: cfg,
    )
    mock_model = MagicMock()
    with patch(
        "src.infrastructure.reranking.cross_encoder_reranker.CrossEncoder",
        return_value=mock_model,
    ):
        rr = CrossEncoderReranker()
    return rr, mock_model


# ---------------------------------------------------------------------------
# Lista vazia
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_chunks_returns_empty_list(self, reranker) -> None:
        rr, _ = reranker
        assert rr.rerank("query", []) == []


# ---------------------------------------------------------------------------
# Ordenação por score
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_highest_score_is_first(self, reranker) -> None:
        rr, mock_model = reranker
        c1, c2, c3 = _chunk("c1"), _chunk("c2"), _chunk("c3")
        mock_model.predict.return_value = np.array([0.2, 0.9, 0.5])
        result = rr.rerank("query", [c1, c2, c3])
        assert result[0].id == "c2"

    def test_lowest_score_is_last(self, reranker) -> None:
        rr, mock_model = reranker
        c1, c2, c3 = _chunk("c1"), _chunk("c2"), _chunk("c3")
        mock_model.predict.return_value = np.array([0.7, 0.9, 0.1])
        result = rr.rerank("query", [c1, c2, c3])
        assert result[-1].id == "c3"

    def test_single_chunk_returned_unchanged(self, reranker) -> None:
        rr, mock_model = reranker
        chunk = _chunk("solo")
        mock_model.predict.return_value = np.array([0.75])
        result = rr.rerank("query", [chunk])
        assert len(result) == 1 and result[0].id == "solo"


# ---------------------------------------------------------------------------
# Preservação de chunks
# ---------------------------------------------------------------------------

class TestPreservation:
    def test_returns_same_count_as_input(self, reranker) -> None:
        rr, mock_model = reranker
        chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
        mock_model.predict.return_value = np.array([0.5, 0.9, 0.3])
        result = rr.rerank("query", chunks)
        assert len(result) == 3

    def test_all_original_chunks_preserved(self, reranker) -> None:
        rr, mock_model = reranker
        chunks = [_chunk(f"c{i}") for i in range(5)]
        mock_model.predict.return_value = np.array([0.5, 0.4, 0.9, 0.1, 0.7])
        result = rr.rerank("query", chunks)
        assert {c.id for c in result} == {c.id for c in chunks}

    def test_returns_list_of_chunk_instances(self, reranker) -> None:
        rr, mock_model = reranker
        chunks = [_chunk("c1"), _chunk("c2")]
        mock_model.predict.return_value = np.array([0.6, 0.8])
        result = rr.rerank("query", chunks)
        assert isinstance(result, list) and all(isinstance(c, Chunk) for c in result)


# ---------------------------------------------------------------------------
# Pares enviados ao cross-encoder
# ---------------------------------------------------------------------------

class TestPairs:
    def test_pairs_contain_the_query(self, reranker) -> None:
        rr, mock_model = reranker
        mock_model.predict.return_value = np.array([0.8])
        rr.rerank("minha query", [_chunk("c1", "texto relevante")])
        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0][0] == "minha query"

    def test_pairs_contain_chunk_text(self, reranker) -> None:
        rr, mock_model = reranker
        texto = "texto específico do chunk sobre Paraná"
        mock_model.predict.return_value = np.array([0.8])
        rr.rerank("query", [_chunk("c1", texto)])
        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0][1] == texto
