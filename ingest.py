"""Script raiz de ingestão — executa uma única vez (ou quantas vezes necessário,
idempotente por document_id) para construir o índice FAISS + SQLite em data/processed/.

Uso:
    python ingest.py
"""
from __future__ import annotations

import os

# TRANSFORMERS_OFFLINE antes de qualquer import de modelo HuggingFace
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.application.ingest_documents import IngestDocuments
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker
from src.infrastructure.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.infrastructure.pdf.pymupdf_extractor import PyMuPDFExtractor
from src.infrastructure.reranking.cross_encoder_reranker import CrossEncoderReranker  # noqa: F401
from src.infrastructure.storage.faiss_vector_store import FAISSVectorStore
from src.infrastructure.storage.sqlite_metadata_store import SQLiteMetadataStore

if __name__ == "__main__":
    metadata_store = SQLiteMetadataStore()
    vector_store = FAISSVectorStore(metadata_store)

    use_case = IngestDocuments(
        extractor=PyMuPDFExtractor(),
        chunker=RecursiveChunker(),
        embedder=SentenceTransformerEmbedder(),
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    results = use_case.execute()

    for filename, status in results.items():
        print(f"  {status:>8}  {filename}")
