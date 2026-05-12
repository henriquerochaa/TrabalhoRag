from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Document:
    id: str       # md5 do filename — garante idempotência na ingestão
    filename: str


@dataclass
class Chunk:
    id: str       # f"{document_id}_{page}_{position}"
    document_id: str
    text: str
    page: int
    section: str
    position: int
    # score padrão 0.0: chunks criados no ingest não têm score de retrieval;
    # o valor real é atribuído por SearchChunks.execute() após a busca FAISS
    score: float = 0.0


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    prompt_used: str
    # exigido pelo enunciado para sinalizar perguntas fora do escopo dos PDFs
    out_of_scope: bool = False
