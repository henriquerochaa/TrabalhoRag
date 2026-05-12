from __future__ import annotations

import numpy as np
import pytest

from src.application.search_chunks import SearchChunks
from src.domain.entities import Chunk, SearchResult
from src.domain.ports import EmbedderPort, RerankerPort, VectorStorePort


# ---------------------------------------------------------------------------
# helpers compartilhados
# ---------------------------------------------------------------------------

def _chunk(cid: str, text: str = "texto de exemplo sobre o Paraná") -> Chunk:
    return Chunk(id=cid, document_id="doc", text=text, page=1, section="", position=0)


class _OrthogonalEmbedder(EmbedderPort):
    # Vetores ortonormais entre chunks: dot(chunk_i, chunk_j) = 0 para i ≠ j.
    # Consequência: no _mmr, diversity = 0 para todos os candidatos após a 1ª
    # seleção → MMR score = λ * relevância, preservando a ordem do FAISS.
    # Qualquer mudança de ordem após _mmr só pode vir do reranker.
    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        return np.eye(n, dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 1), dtype=np.float32)


class _PassthroughReranker(RerankerPort):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return chunks


class _ReversingReranker(RerankerPort):
    """Inverte a lista recebida — simula reranker que discorda completamente do FAISS."""
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return list(reversed(chunks))


class _EmptyStore(VectorStorePort):
    def add(self, c, e): pass
    def search(self, emb, top_k): return []
    def save(self, p): pass
    def load(self, p): pass


def _make_empty_store() -> _EmptyStore:
    return _EmptyStore()


def _make_sc(store: VectorStorePort, reranker: RerankerPort | None = None) -> SearchChunks:
    return SearchChunks(
        _OrthogonalEmbedder(),
        store,
        reranker if reranker is not None else _PassthroughReranker(),
    )


# ---------------------------------------------------------------------------
# reranker muda a ordem em relação ao FAISS
# ---------------------------------------------------------------------------

class TestRerankerChangesOrder:
    def test_reranker_overrides_faiss_order(self) -> None:
        # FAISS retorna A(0.90) > B(0.80) > C(0.70).
        # Com embeddings ortogonais, MMR preserva essa ordem (diversity = 0).
        # _ReversingReranker inverte para [C, B, A].
        # Se o reranker fosse ignorado, resultado seria [A, B, C].
        chunks = [_chunk("A"), _chunk("B"), _chunk("C")]
        results = [
            SearchResult(chunks[0], score=0.90),
            SearchResult(chunks[1], score=0.80),
            SearchResult(chunks[2], score=0.70),
        ]

        class FixedStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k): return results
            def save(self, p): pass
            def load(self, p): pass

        returned, out_of_scope = _make_sc(FixedStore(), _ReversingReranker()).execute("query")

        assert out_of_scope is False
        assert [c.id for c in returned] == ["C", "B", "A"], (
            f"esperado ordem do reranker [C, B, A], obtido {[c.id for c in returned]}"
        )

    def test_passthrough_reranker_preserves_mmr_order(self) -> None:
        # Controle: sem reordenação pelo reranker, ordem do FAISS é mantida.
        chunks = [_chunk("X"), _chunk("Y"), _chunk("Z")]
        results = [
            SearchResult(chunks[0], score=0.90),
            SearchResult(chunks[1], score=0.80),
            SearchResult(chunks[2], score=0.70),
        ]

        class FixedStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k): return results
            def save(self, p): pass
            def load(self, p): pass

        returned, _ = _make_sc(FixedStore(), _PassthroughReranker()).execute("query")

        assert [c.id for c in returned] == ["X", "Y", "Z"]

    def test_reranker_receives_mmr_subset_not_full_faiss(self) -> None:
        # Reranker deve receber apenas top_k_final chunks do MMR, não todos os
        # top_k_initial do FAISS — store retorna 3, reranker deve ver 3.
        received: list[Chunk] = []

        class CapturingReranker(RerankerPort):
            def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
                received.extend(chunks)
                return chunks

        three_chunks = [_chunk(f"c{i}") for i in range(3)]

        class SmallStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k):
                return [SearchResult(c, 0.9 - i * 0.05) for i, c in enumerate(three_chunks)]
            def save(self, p): pass
            def load(self, p): pass

        _make_sc(SmallStore(), CapturingReranker()).execute("query")
        assert len(received) == 3


# ---------------------------------------------------------------------------
# query fora do escopo → out_of_scope=True
# ---------------------------------------------------------------------------

class TestOutOfScope:
    def test_score_below_threshold_returns_empty_and_flag(self) -> None:
        # 0.30 < 0.84 (min_score_threshold do config.yaml — calibrado para multilingual-e5-large)
        class LowStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k):
                return [SearchResult(_chunk("low"), score=0.30)]
            def save(self, p): pass
            def load(self, p): pass

        chunks, out_of_scope = _make_sc(LowStore()).execute("Qual a capital da França?")

        assert chunks == []
        assert out_of_scope is True

    def test_empty_store_returns_out_of_scope(self) -> None:
        chunks, out_of_scope = _make_sc(_make_empty_store()).execute("qualquer coisa")

        assert chunks == []
        assert out_of_scope is True

    def test_score_just_below_threshold(self) -> None:
        # 0.83 < 0.84 — borda inferior: deve ser out_of_scope
        class BorderStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k):
                return [SearchResult(_chunk("border"), score=0.83)]
            def save(self, p): pass
            def load(self, p): pass

        _, out_of_scope = _make_sc(BorderStore()).execute("query")

        assert out_of_scope is True


# ---------------------------------------------------------------------------
# query dentro do escopo → chunks retornados, out_of_scope=False
# ---------------------------------------------------------------------------

class TestInScope:
    def test_score_above_threshold_returns_chunks(self) -> None:
        # 0.85 >= 0.84 → in scope, lista não vazia
        target = _chunk("alvo", "O Paraná é a quinta maior economia do Brasil.")

        class AboveStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k):
                return [SearchResult(target, score=0.85)]
            def save(self, p): pass
            def load(self, p): pass

        chunks, out_of_scope = _make_sc(AboveStore()).execute("economia do Paraná")

        assert out_of_scope is False
        assert len(chunks) > 0
        assert chunks[0].id == "alvo"

    def test_score_exactly_at_threshold_is_in_scope(self) -> None:
        # código usa `< threshold` (estrito): score == 0.84 é in scope
        class AtStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k):
                return [SearchResult(_chunk("at"), score=0.84)]
            def save(self, p): pass
            def load(self, p): pass

        chunks, out_of_scope = _make_sc(AtStore()).execute("query")

        assert out_of_scope is False
        assert len(chunks) > 0

    def test_returned_chunk_ids_are_subset_of_store_results(self) -> None:
        # chunks retornados devem ter IDs que existem nos resultados do store
        source_chunks = [_chunk(f"src{i}", f"conteúdo relevante {i}") for i in range(3)]

        class KnownStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k):
                return [SearchResult(c, 0.9 - i * 0.05) for i, c in enumerate(source_chunks)]
            def save(self, p): pass
            def load(self, p): pass

        chunks, _ = _make_sc(KnownStore()).execute("query")

        source_ids = {c.id for c in source_chunks}
        assert all(c.id in source_ids for c in chunks)


# ---------------------------------------------------------------------------
# compressão não trunca chunks no meio
# ---------------------------------------------------------------------------

class TestCompression:
    def test_no_chunk_text_is_truncated(self) -> None:
        # 4 chunks de 100 chars → 25 tokens cada (chars // 4).
        # token_budget = 75 → exatamente 3 cabem (3 × 25 = 75).
        # O 4º deve ser descartado inteiro; nenhum chunk deve ter texto cortado.
        text = "P" * 100
        chunks = [_chunk(f"c{i}", text) for i in range(4)]
        original_texts = {c.text for c in chunks}

        class FourStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k):
                return [SearchResult(c, 0.9 - i * 0.05) for i, c in enumerate(chunks)]
            def save(self, p): pass
            def load(self, p): pass

        sc = _make_sc(FourStore())
        sc._token_budget = 75  # sobrescreve budget para isolar o teste do config real

        returned, _ = sc.execute("query")

        for c in returned:
            assert c.text in original_texts, (
                f"texto do chunk '{c.id}' não corresponde a nenhum original — "
                f"possível truncagem: '{c.text[:40]}'"
            )

    def test_compression_stays_within_token_budget(self) -> None:
        # total de tokens estimados dos chunks retornados ≤ token_budget
        text = "Q" * 100  # 25 tokens estimados
        chunks = [_chunk(f"d{i}", text) for i in range(4)]

        class FourStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k):
                return [SearchResult(c, 0.9 - i * 0.05) for i, c in enumerate(chunks)]
            def save(self, p): pass
            def load(self, p): pass

        sc = _make_sc(FourStore())
        sc._token_budget = 75

        returned, _ = sc.execute("query")

        total_tokens = sum(max(1, len(c.text) // 4) for c in returned)
        assert total_tokens <= 75

    def test_token_budget_respects_max_tokens_reservation(self) -> None:
        # token_budget deve ser context_window - max_tokens, garantindo que a
        # LLM tem espaço para gerar a resposta sem truncar o contexto.
        sc = _make_sc(_make_empty_store())
        assert sc._token_budget == sc._context_window - sc._max_tokens

    def test_compression_discards_lowest_score_first(self) -> None:
        # reranker passthrough mantém ordem por score; compressão deve remover
        # os do final da lista (menor score), preservando os de maior score.
        text = "R" * 100  # 25 tokens cada
        chunks = [_chunk(f"rank{i}", text) for i in range(4)]

        class OrderedStore(VectorStorePort):
            def add(self, c, e): pass
            def search(self, emb, top_k):
                return [SearchResult(c, 0.9 - i * 0.05) for i, c in enumerate(chunks)]
            def save(self, p): pass
            def load(self, p): pass

        sc = _make_sc(OrderedStore())
        sc._token_budget = 50  # 50 // 25 = 2 chunks cabem

        returned, _ = sc.execute("query")

        assert len(returned) == 2
        assert returned[0].id == "rank0"
        assert returned[1].id == "rank1"
