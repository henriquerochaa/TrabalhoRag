from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from src.domain.entities import Chunk, Document, SearchResult


class PDFExtractorPort(ABC):
    @abstractmethod
    def extract(self, path: str) -> list[tuple[int, str]]:
        """Retorna lista de (número_da_página, texto_da_página)."""
        ...


class ChunkerPort(ABC):
    @abstractmethod
    def chunk(self, pages: list[tuple[int, str]], document: Document) -> list[Chunk]:
        ...


class EmbedderPort(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        # retorna matriz (N, dim) para passagens (chunks)
        ...

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> np.ndarray:
        # retorna matriz (N, dim) para queries de busca
        # separado de embed() porque modelos como e5 exigem prefixos distintos
        # ("query:" vs "passage:") — unificar os dois quebraria a semântica
        ...


class VectorStorePort(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...


class MetadataStorePort(ABC):
    @abstractmethod
    def save_chunks(self, chunks: list[Chunk]) -> None:
        ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        ...

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        # verifica se ao menos um chunk do documento já foi indexado —
        # usado pelo IngestDocuments para garantir idempotência por documento
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...


class RerankerPort(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        # retorna os mesmos chunks reordenados do mais ao menos relevante
        ...


class LLMPort(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...
