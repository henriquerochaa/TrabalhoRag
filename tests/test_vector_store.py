from __future__ import annotations

import numpy as np
import pytest

from src.domain.entities import Chunk, SearchResult
from src.infrastructure.storage.faiss_vector_store import FAISSVectorStore
from src.infrastructure.storage.sqlite_metadata_store import SQLiteMetadataStore


def _make_chunks(n: int, doc_id: str = "doc1") -> list[Chunk]:
    return [
        Chunk(
            id=f"{doc_id}_1_{i}",
            document_id=doc_id,
            text=f"Texto do chunk número {i} sobre o Paraná.",
            page=i + 1,
            section=f"Seção {i}",
            position=i,
        )
        for i in range(n)
    ]


def _make_embeddings(n: int, dim: int = 16, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((n, dim)).astype(np.float32)
    # L2-normalizado para compatibilidade com METRIC_INNER_PRODUCT
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """FAISSVectorStore com SQLiteMetadataStore real apontando para tmp_path."""
    cfg = {"paths": {"processed": str(tmp_path)}}
    monkeypatch.setattr("src.infrastructure.storage.sqlite_metadata_store.load_config", lambda: cfg)
    monkeypatch.setattr("src.infrastructure.storage.faiss_vector_store.load_config", lambda: cfg)
    meta = SQLiteMetadataStore()
    return FAISSVectorStore(meta), meta, tmp_path


# ---------------------------------------------------------------------------
# top-k
# ---------------------------------------------------------------------------

class TestTopK:
    def test_returns_exactly_top_k(self, store) -> None:
        vs, _, _ = store
        n, dim, k = 20, 16, 5
        vs.add(_make_chunks(n), _make_embeddings(n, dim))
        query = _make_embeddings(1, dim, seed=99)
        results = vs.search(query[0], top_k=k)
        assert len(results) == k

    def test_returns_all_when_k_exceeds_index_size(self, store) -> None:
        vs, _, _ = store
        n, dim = 3, 16
        vs.add(_make_chunks(n), _make_embeddings(n, dim))
        query = _make_embeddings(1, dim, seed=99)
        results = vs.search(query[0], top_k=10)
        assert len(results) == n

    def test_empty_index_returns_empty(self, store) -> None:
        vs, _, _ = store
        query = _make_embeddings(1, 16, seed=1)
        assert vs.search(query[0], top_k=5) == []

    def test_top_1_returns_single_result(self, store) -> None:
        vs, _, _ = store
        vs.add(_make_chunks(10), _make_embeddings(10, 16))
        query = _make_embeddings(1, 16, seed=77)
        results = vs.search(query[0], top_k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# ordenação de scores
# ---------------------------------------------------------------------------

class TestScoreOrdering:
    def test_scores_are_descending(self, store) -> None:
        vs, _, _ = store
        vs.add(_make_chunks(20), _make_embeddings(20, 16))
        query = _make_embeddings(1, 16, seed=7)
        results = vs.search(query[0], top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), f"scores fora de ordem: {scores}"

    def test_self_query_scores_near_one(self, store) -> None:
        # busca pelo próprio vetor deve retornar score ~1.0 (produto interno de vetores L2-normalizados)
        vs, _, _ = store
        embeddings = _make_embeddings(5, 16)
        vs.add(_make_chunks(5), embeddings)
        results = vs.search(embeddings[2], top_k=1)
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_all_scores_are_float(self, store) -> None:
        vs, _, _ = store
        vs.add(_make_chunks(5), _make_embeddings(5, 16))
        query = _make_embeddings(1, 16, seed=3)
        for r in vs.search(query[0], top_k=5):
            assert isinstance(r.score, float)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_load_same_top1(self, store, monkeypatch) -> None:
        vs, meta, tmp_path = store

        embeddings = _make_embeddings(10, 16)
        chunks = _make_chunks(10)
        vs.add(chunks, embeddings)

        vs.save(str(tmp_path))

        # novo store carregado do disco
        cfg = {"paths": {"processed": str(tmp_path)}}
        monkeypatch.setattr("src.infrastructure.storage.sqlite_metadata_store.load_config", lambda: cfg)
        monkeypatch.setattr("src.infrastructure.storage.faiss_vector_store.load_config", lambda: cfg)
        meta2 = SQLiteMetadataStore()
        vs2 = FAISSVectorStore(meta2)
        vs2.load(str(tmp_path))

        query = embeddings[3:4]
        r1 = vs.search(query[0], top_k=3)
        r2 = vs2.search(query[0], top_k=3)

        assert [r.chunk.id for r in r1] == [r.chunk.id for r in r2]
        assert [r.score for r in r1] == pytest.approx([r.score for r in r2], abs=1e-5)

    def test_save_creates_required_files(self, store) -> None:
        vs, _, tmp_path = store
        vs.add(_make_chunks(5), _make_embeddings(5, 16))
        vs.save(str(tmp_path))
        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "id_map.json").exists()
        assert (tmp_path / "metadata.db").exists()


# ---------------------------------------------------------------------------
# metadados via SQLiteMetadataStore
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_returned_chunks_have_correct_text(self, store) -> None:
        vs, _, _ = store
        embeddings = _make_embeddings(5, 16)
        chunks = _make_chunks(5)
        vs.add(chunks, embeddings)
        query = embeddings[1:2]
        results = vs.search(query[0], top_k=3)
        for r in results:
            original = next(c for c in chunks if c.id == r.chunk.id)
            assert r.chunk.text == original.text

    def test_returned_chunks_have_correct_page(self, store) -> None:
        vs, _, _ = store
        embeddings = _make_embeddings(5, 16)
        chunks = _make_chunks(5)
        vs.add(chunks, embeddings)
        results = vs.search(embeddings[0], top_k=5)
        for r in results:
            original = next(c for c in chunks if c.id == r.chunk.id)
            assert r.chunk.page == original.page

    def test_returned_chunks_have_correct_section(self, store) -> None:
        vs, _, _ = store
        embeddings = _make_embeddings(5, 16)
        chunks = _make_chunks(5)
        vs.add(chunks, embeddings)
        results = vs.search(embeddings[0], top_k=5)
        for r in results:
            original = next(c for c in chunks if c.id == r.chunk.id)
            assert r.chunk.section == original.section

    def test_returned_chunks_have_correct_document_id(self, store) -> None:
        vs, _, _ = store
        embeddings = _make_embeddings(5, 16)
        chunks = _make_chunks(5, doc_id="ipardes_doc")
        vs.add(chunks, embeddings)
        results = vs.search(embeddings[0], top_k=5)
        for r in results:
            assert r.chunk.document_id == "ipardes_doc"

    def test_results_are_search_result_instances(self, store) -> None:
        vs, _, _ = store
        vs.add(_make_chunks(3), _make_embeddings(3, 16))
        results = vs.search(_make_embeddings(1, 16, seed=5)[0], top_k=3)
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.chunk, Chunk)
