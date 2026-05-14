from __future__ import annotations

import numpy as np
import pytest

from src.application.ingest_documents import IngestDocuments
from src.domain.entities import Chunk, Document
from src.domain.ports import (
    ChunkerPort,
    EmbedderPort,
    MetadataStorePort,
    PDFExtractorPort,
    VectorStorePort,
)


# ---------------------------------------------------------------------------
# Stubs mínimos — sem dependências externas
# ---------------------------------------------------------------------------

def _chunk(doc_id: str, position: int = 0) -> Chunk:
    return Chunk(
        id=f"{doc_id}_1_{position}",
        document_id=doc_id,
        text="Texto de exemplo sobre o Paraná.",
        page=1,
        section="Seção",
        position=position,
    )


class _FakeExtractor(PDFExtractorPort):
    def extract(self, path: str) -> list[tuple[int, str]]:
        return [(1, "Conteúdo da página 1."), (2, "Conteúdo da página 2.")]


class _FakeChunker(ChunkerPort):
    def chunk(self, pages: list[tuple[int, str]], document: Document) -> list[Chunk]:
        return [_chunk(document.id, i) for i in range(len(pages))]


class _FakeEmbedder(EmbedderPort):
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.eye(len(texts), dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 1), dtype=np.float32)


class _FakeVectorStore(VectorStorePort):
    def __init__(self):
        self.added_chunks: list[Chunk] = []
        self.saved = False

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.added_chunks.extend(chunks)

    def search(self, q, top_k): return []
    def save(self, path: str) -> None: self.saved = True
    def load(self, path: str) -> None: pass


class _FakeMetadataStore(MetadataStorePort):
    def __init__(self):
        self._chunks: dict[str, Chunk] = {}
        self._docs: set[str] = set()
        self.saved = False

    def save_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c
            self._docs.add(c.document_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def document_exists(self, document_id: str) -> bool:
        return document_id in self._docs

    def save(self, path: str) -> None: self.saved = True
    def load(self, path: str) -> None: pass


@pytest.fixture()
def use_case(monkeypatch):
    cfg = {
        "paths": {"processed": "data/processed/"},
        "pdfs": [
            {"filename": "doc_a.pdf"},
            {"filename": "doc_b.pdf"},
        ],
    }
    monkeypatch.setattr("src.application.ingest_documents.load_config", lambda: cfg)
    vs = _FakeVectorStore()
    meta = _FakeMetadataStore()
    uc = IngestDocuments(
        extractor=_FakeExtractor(),
        chunker=_FakeChunker(),
        embedder=_FakeEmbedder(),
        vector_store=vs,
        metadata_store=meta,
    )
    return uc, vs, meta


# ---------------------------------------------------------------------------
# Resultado do execute()
# ---------------------------------------------------------------------------

class TestExecuteResult:
    def test_returns_dict_with_all_filenames(self, use_case) -> None:
        uc, _, _ = use_case
        result = uc.execute()
        assert set(result.keys()) == {"doc_a.pdf", "doc_b.pdf"}

    def test_first_run_marks_all_as_ingested(self, use_case) -> None:
        uc, _, _ = use_case
        result = uc.execute()
        assert all(v == "ingested" for v in result.values())

    def test_second_run_marks_all_as_skipped(self, use_case) -> None:
        # idempotência: segunda execução não reindexa documentos já presentes
        uc, _, _ = use_case
        uc.execute()
        result = uc.execute()
        assert all(v == "skipped" for v in result.values())

    def test_new_doc_ingested_existing_skipped(self, use_case, monkeypatch) -> None:
        # configura apenas doc_a na primeira rodada
        cfg_one = {
            "paths": {"processed": "data/processed/"},
            "pdfs": [{"filename": "doc_a.pdf"}],
        }
        cfg_two = {
            "paths": {"processed": "data/processed/"},
            "pdfs": [{"filename": "doc_a.pdf"}, {"filename": "doc_b.pdf"}],
        }
        monkeypatch.setattr("src.application.ingest_documents.load_config", lambda: cfg_one)
        uc, _, _ = use_case
        uc.execute()

        monkeypatch.setattr("src.application.ingest_documents.load_config", lambda: cfg_two)
        result = uc.execute()
        assert result["doc_a.pdf"] == "skipped"
        assert result["doc_b.pdf"] == "ingested"


# ---------------------------------------------------------------------------
# Persistência chamada após ingestão
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_vector_store_save_called(self, use_case) -> None:
        uc, vs, _ = use_case
        uc.execute()
        assert vs.saved is True

    def test_metadata_store_save_called(self, use_case) -> None:
        uc, _, meta = use_case
        uc.execute()
        assert meta.saved is True


# ---------------------------------------------------------------------------
# Chunks chegam ao vector store e metadata store
# ---------------------------------------------------------------------------

class TestChunkFlow:
    def test_chunks_added_to_vector_store(self, use_case) -> None:
        uc, vs, _ = use_case
        uc.execute()
        # 2 documentos × 2 páginas cada = 4 chunks
        assert len(vs.added_chunks) == 4

    def test_chunks_saved_to_metadata_store(self, use_case) -> None:
        uc, _, meta = use_case
        uc.execute()
        assert len(meta._chunks) == 4

    def test_document_ids_registered_in_metadata(self, use_case) -> None:
        uc, _, meta = use_case
        uc.execute()
        assert meta.document_exists(next(iter(meta._docs)))

    def test_skipped_doc_adds_no_chunks(self, use_case) -> None:
        # na segunda rodada, nenhum chunk novo deve ser adicionado ao vector store
        uc, vs, _ = use_case
        uc.execute()
        count_after_first = len(vs.added_chunks)
        uc.execute()
        assert len(vs.added_chunks) == count_after_first
