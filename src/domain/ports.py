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
        # retorna matriz (N, dim) — cada linha é o embedding de um texto
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
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...


class LLMPort(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...
