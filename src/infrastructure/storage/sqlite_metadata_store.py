from __future__ import annotations

import sqlite3
from pathlib import Path

from src.domain.entities import Chunk
from src.domain.ports import MetadataStorePort
from src.infrastructure.config_loader import load_config

_DB_FILENAME = "metadata.db"

# SQLite escolhido sobre pickle/JSON por três razões:
# 1. Buscas por id são O(1) via PRIMARY KEY INDEX — get_many_by_ids com IN (?)
#    evita N queries; qualquer alternativa em arquivo plano exigiria varredura
# 2. ACID nativo garante consistência mesmo se o ingest for interrompido
# 3. Zero dependências externas — stdlib pura, sem ORM, sem servidor


class SQLiteMetadataStore(MetadataStorePort):
    def __init__(self) -> None:
        processed = Path(load_config()["paths"]["processed"])
        processed.mkdir(parents=True, exist_ok=True)
        self._db_path = processed / _DB_FILENAME
        # check_same_thread=False necessário para uso com FastAPI (threads distintas)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text        TEXT NOT NULL,
                page        INTEGER NOT NULL,
                section     TEXT NOT NULL,
                position    INTEGER NOT NULL
            )
        """)
        self._conn.commit()

    def _row_to_chunk(self, row: tuple) -> Chunk:
        return Chunk(
            id=row[0],
            document_id=row[1],
            text=row[2],
            page=row[3],
            section=row[4],
            position=row[5],
        )

    # ------------------------------------------------------------------
    # métodos específicos da implementação
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: Chunk) -> None:
        # INSERT OR IGNORE garante idempotência — reingestão não duplica dados
        self._conn.execute(
            "INSERT OR IGNORE INTO chunks (id, document_id, text, page, section, position) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chunk.id, chunk.document_id, chunk.text, chunk.page, chunk.section, chunk.position),
        )
        self._conn.commit()

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        cur = self._conn.execute(
            "SELECT id, document_id, text, page, section, position FROM chunks WHERE id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        return self._row_to_chunk(row) if row else None

    def get_many_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cur = self._conn.execute(
            f"SELECT id, document_id, text, page, section, position "
            f"FROM chunks WHERE id IN ({placeholders})",
            ids,
        )
        return [self._row_to_chunk(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # MetadataStorePort
    # ------------------------------------------------------------------

    def save_chunks(self, chunks: list[Chunk]) -> None:
        # executemany reduz round-trips ao SQLite comparado a N inserts individuais
        self._conn.executemany(
            "INSERT OR IGNORE INTO chunks (id, document_id, text, page, section, position) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(c.id, c.document_id, c.text, c.page, c.section, c.position) for c in chunks],
        )
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.get_by_id(chunk_id)

    def save(self, path: str) -> None:
        # SQLite persiste em disco automaticamente — commit garante flush
        self._conn.commit()

    def load(self, path: str) -> None:
        self._conn.close()
        self._conn = sqlite3.connect(
            str(Path(path) / _DB_FILENAME), check_same_thread=False
        )
        self._create_table()
