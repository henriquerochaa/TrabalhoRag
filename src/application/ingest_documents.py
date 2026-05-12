from __future__ import annotations

import hashlib

from src.config_loader import load_config
from src.domain.entities import Document
from src.domain.ports import (
    ChunkerPort,
    EmbedderPort,
    MetadataStorePort,
    PDFExtractorPort,
    VectorStorePort,
)



class IngestDocuments:
    """Ingere os PDFs do IPARDES no índice vetorial.

    Idempotente: documentos cujo document_id já está no SQLite são ignorados,
    permitindo reexecutar o ingest sem duplicar dados ou recalcular embeddings.
    Nunca faz download — lê apenas de data/raw/ conforme config.yaml.
    """

    def __init__(
        self,
        extractor: PDFExtractorPort,
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        metadata_store: MetadataStorePort,
    ) -> None:
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._metadata_store = metadata_store

    def execute(self) -> dict[str, str]:
        """Processa todos os PDFs listados em config.yaml.

        Retorna {filename: "ingested" | "skipped"} para cada documento.
        """
        cfg = load_config()
        processed_dir = cfg["paths"]["processed"]
        filenames = [entry["filename"] for entry in cfg["pdfs"]]

        results: dict[str, str] = {}

        for filename in filenames:
            # document_id derivado do filename garante estabilidade entre execuções:
            # o mesmo arquivo sempre gera o mesmo id sem depender de conteúdo ou mtime
            document_id = hashlib.md5(filename.encode()).hexdigest()

            if self._metadata_store.document_exists(document_id):
                results[filename] = "skipped"
                continue

            document = Document(id=document_id, filename=filename)

            # extrator recebe apenas o filename — PyMuPDFExtractor monta o path completo
            # internamente usando config["paths"]["raw"], evitando duplicação
            pages = self._extractor.extract(filename)
            chunks = self._chunker.chunk(pages, document)
            embeddings = self._embedder.embed([c.text for c in chunks])

            self._metadata_store.save_chunks(chunks)
            self._vector_store.add(chunks, embeddings)

            results[filename] = "ingested"

        # salva após processar todos os documentos: uma única operação de I/O
        # em vez de um save por documento — reduz risco de estado inconsistente
        # entre FAISS e SQLite se o processo for interrompido no meio
        self._metadata_store.save(processed_dir)
        self._vector_store.save(processed_dir)

        return results
