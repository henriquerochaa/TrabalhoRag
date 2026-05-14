from __future__ import annotations

import numpy as np
import pytest

from src.application.search_chunks import SearchChunks
from src.domain.entities import Chunk, SearchResult
from src.domain.ports import EmbedderPort, RerankerPort, VectorStorePort


def _chunk(cid: str, doc_id: str = "doc") -> Chunk:
    return Chunk(id=cid, document_id=doc_id, text=f"texto {cid}", page=1, section="", position=0)


def _result(cid: str, score: float, doc_id: str = "doc") -> SearchResult:
    return SearchResult(chunk=_chunk(cid, doc_id), score=score)


class _OrthEmbedder(EmbedderPort):
    # vetores ortonormais: dot(e_i, e_j) = 0 para i≠j → diversidade = 0 após 1ª seleção
    # consequência: MMR reduz a pura relevância (ordem por score), facilitando asserções
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.eye(len(texts), dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 1), dtype=np.float32)


class _PassReranker(RerankerPort):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return chunks


class _EmptyStore(VectorStorePort):
    def add(self, c, e): pass
    def search(self, emb, top_k): return []
    def save(self, p): pass
    def load(self, p): pass


def _make_sc() -> SearchChunks:
    return SearchChunks(_OrthEmbedder(), _EmptyStore(), _PassReranker())


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------

class TestTopK:
    def test_returns_exactly_top_k(self) -> None:
        sc = _make_sc()
        results = [_result(f"c{i}", 0.9 - i * 0.05) for i in range(10)]
        selected = sc._mmr(np.array([1.0], dtype=np.float32), results, top_k=3, lam=0.5)
        assert len(selected) == 3

    def test_returns_all_when_k_greater_than_candidates(self) -> None:
        sc = _make_sc()
        results = [_result(f"c{i}", 0.9 - i * 0.05) for i in range(2)]
        selected = sc._mmr(np.array([1.0], dtype=np.float32), results, top_k=5, lam=0.5)
        assert len(selected) == 2

    def test_single_result_returns_single(self) -> None:
        sc = _make_sc()
        results = [_result("solo", 0.90)]
        selected = sc._mmr(np.array([1.0], dtype=np.float32), results, top_k=3, lam=0.5)
        assert len(selected) == 1 and selected[0].chunk.id == "solo"


# ---------------------------------------------------------------------------
# Relevância pura (λ=1)
# ---------------------------------------------------------------------------

class TestRelevance:
    def test_lambda_one_selects_highest_score_first(self) -> None:
        sc = _make_sc()
        results = [
            _result("high", 0.95),
            _result("mid", 0.80),
            _result("low", 0.60),
        ]
        selected = sc._mmr(np.array([1.0], dtype=np.float32), results, top_k=3, lam=1.0)
        assert selected[0].chunk.id == "high"

    def test_output_is_subset_of_input(self) -> None:
        sc = _make_sc()
        results = [_result(f"c{i}", 0.9 - i * 0.05) for i in range(6)]
        selected = sc._mmr(np.array([1.0], dtype=np.float32), results, top_k=3, lam=0.5)
        input_ids = {r.chunk.id for r in results}
        assert all(r.chunk.id in input_ids for r in selected)

    def test_no_duplicates_in_output(self) -> None:
        sc = _make_sc()
        results = [_result(f"c{i}", 0.9 - i * 0.05) for i in range(6)]
        selected = sc._mmr(np.array([1.0], dtype=np.float32), results, top_k=4, lam=0.5)
        ids = [r.chunk.id for r in selected]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Cap por documento (diversidade multi-doc)
# ---------------------------------------------------------------------------

class TestDocCap:
    def test_cap_forces_second_document_into_results(self) -> None:
        # 3 chunks do doc1 com scores altos + 1 chunk do doc2 com score menor.
        # cap por documento: max(1, top_k-1) = 2 → doc1 pode ter no máx 2 chunks.
        # com top_k=3: o 3º slot obrigatoriamente vai para doc2.
        sc = _make_sc()
        results = [
            _result("a1", 0.95, doc_id="doc1"),
            _result("a2", 0.93, doc_id="doc1"),
            _result("a3", 0.91, doc_id="doc1"),
            _result("b1", 0.80, doc_id="doc2"),
        ]
        selected = sc._mmr(np.array([1.0], dtype=np.float32), results, top_k=3, lam=0.5)
        doc_ids = {r.chunk.document_id for r in selected}
        assert "doc2" in doc_ids, (
            "MMR deve incluir doc2 — cap por documento impede que doc1 ocupe todos os slots"
        )

    def test_cap_relaxes_when_all_from_same_doc(self) -> None:
        # corpus com apenas 1 documento: fallback remove o cap e preenche top_k
        sc = _make_sc()
        results = [_result(f"c{i}", 0.9 - i * 0.05, doc_id="único") for i in range(5)]
        selected = sc._mmr(np.array([1.0], dtype=np.float32), results, top_k=3, lam=0.5)
        assert len(selected) == 3

    def test_two_docs_both_appear_in_results(self) -> None:
        sc = _make_sc()
        results = [
            _result("d1c1", 0.90, doc_id="doc1"),
            _result("d1c2", 0.85, doc_id="doc1"),
            _result("d2c1", 0.80, doc_id="doc2"),
        ]
        selected = sc._mmr(np.array([1.0], dtype=np.float32), results, top_k=3, lam=0.5)
        doc_ids = {r.chunk.document_id for r in selected}
        assert "doc1" in doc_ids and "doc2" in doc_ids
