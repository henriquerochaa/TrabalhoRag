from __future__ import annotations

import os

# TRANSFORMERS_OFFLINE e HF_DATASETS_OFFLINE definidos antes de qualquer import
# de modelo — qualquer tentativa de download em runtime gera erro imediato
# em vez de fazer chamada de rede silenciosa, garantindo execução 100% offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Cross-encoder escolhido sobre bi-encoder para reranking por duas razões:
# 1. Atenção cruzada entre query e passagem — o modelo vê os dois textos juntos
#    numa única forward pass, capturando interações token-a-token que um
#    bi-encoder perde ao codificá-los separadamente. Isso produz scores de
#    relevância mais precisos ao custo de O(k) inferências (uma por chunk).
# 2. Na pipeline RAG, o bi-encoder já fez a filtragem de larga escala (top-20
#    via FAISS). O cross-encoder opera apenas sobre esse conjunto pequeno, então
#    a latência extra é aceitável enquanto a ganho de precisão é substancial —
#    especialmente para perguntas com negação ou nuance que enganam embeddings.
# Referência: Nogueira & Cho (2019) mostraram +10 MRR@10 do cross-encoder
# sobre bi-encoder no MS MARCO; ms-marco-MiniLM-L-6-v2 é a variante distilada
# que preserva ~95% desse ganho com 6x menos parâmetros.

from sentence_transformers import CrossEncoder

from src.domain.entities import Chunk
from src.domain.ports import RerankerPort
from src.infrastructure.config_loader import load_config


class CrossEncoderReranker(RerankerPort):
    def __init__(self) -> None:
        cfg = load_config()
        models_path: str = cfg["paths"]["models"]
        model_name: str = cfg["reranker"]["model_name"]

        os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_path
        os.environ["TRANSFORMERS_CACHE"] = models_path

        self._model = CrossEncoder(model_name, cache_folder=models_path)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked]
