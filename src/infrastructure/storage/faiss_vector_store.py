from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.domain.entities import Chunk, SearchResult
from src.domain.ports import MetadataStorePort, VectorStorePort
from src.infrastructure.config_loader import load_config

# HNSW (Hierarchical Navigable Small World) escolhido sobre IndexFlatIP por dois motivos:
# 1. Complexidade de busca O(log N) vs O(N) do Flat — para o corpus dos 3 PDFs
#    (~3 000–5 000 chunks) a diferença é pequena, mas HNSW escala sem reindexar
#    caso o corpus cresça; IndexFlatIP exigiria varredura completa a cada query.
# 2. Recall ajustável em runtime via ef_search sem rebuildar o índice —
#    permite trocar precisão por latência em produção sem alterar código.
# M=32 é o ponto de equilíbrio padrão: recall >99% com uso de memória razoável.
# METRIC_INNER_PRODUCT exige vetores L2-normalizados (garantido pelo embedder),
# tornando produto interno equivalente à similaridade de cosseno.

_M = 32                  # conexões por camada — mais conexões = maior recall e maior RAM
_EF_CONSTRUCTION = 200   # qualidade do grafo na indexação — valor padrão recomendado
_EF_SEARCH = 64          # janela de busca — aumentar melhora recall a custo de latência

_INDEX_FILE = "index.faiss"
_IDMAP_FILE = "id_map.json"


class FAISSVectorStore(VectorStorePort):
    def __init__(self, metadata_store: MetadataStorePort) -> None:
        self._processed_dir = Path(load_config()["paths"]["processed"])
        self._metadata_store = metadata_store
        self._index: faiss.IndexHNSWFlat | None = None
        # mapeamento posição FAISS (int) → chunk_id (str), necessário pois
        # FAISS retorna índices inteiros e os metadados ficam no SQLite
        self._id_map: list[str] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        if self._index is None:
            self._index = faiss.IndexHNSWFlat(dim, _M, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = _EF_CONSTRUCTION

        self._index.hnsw.efSearch = _EF_SEARCH
        self._index.add(embeddings.astype(np.float32))
        self._id_map.extend(c.id for c in chunks)
        self._metadata_store.save_chunks(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        self._index.hnsw.efSearch = _EF_SEARCH
        scores, indices = self._index.search(query, top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._metadata_store.get_chunk(self._id_map[idx])
            if chunk is not None:
                results.append(SearchResult(chunk=chunk, score=float(score)))

        return results

    def save(self, path: str) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(out_dir / _INDEX_FILE))
        with open(out_dir / _IDMAP_FILE, "w", encoding="utf-8") as f:
            json.dump(self._id_map, f)
        self._metadata_store.save(path)

    def load(self, path: str) -> None:
        in_dir = Path(path)
        self._index = faiss.read_index(str(in_dir / _INDEX_FILE))
        self._index.hnsw.efSearch = _EF_SEARCH
        with open(in_dir / _IDMAP_FILE, encoding="utf-8") as f:
            self._id_map = json.load(f)
        self._metadata_store.load(path)
