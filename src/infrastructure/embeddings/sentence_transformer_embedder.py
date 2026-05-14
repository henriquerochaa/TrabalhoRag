from __future__ import annotations

import os

# TRANSFORMERS_OFFLINE e HF_DATASETS_OFFLINE definidos antes de qualquer import
# de modelo — qualquer tentativa de download em runtime gera erro imediato
# em vez de fazer chamada de rede silenciosa, garantindo execução 100% offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# CUDA_VISIBLE_DEVICES NÃO é definido aqui — a política de GPU é decidida pelo chamador:
#   api.py  → define CUDA_VISIBLE_DEVICES="" antes de qualquer import para manter a
#             VRAM livre para o Ollama (llama3.2:3b) quando rodando em produção.
#   ingest.py → não define CUDA_VISIBLE_DEVICES, então PyTorch usa GPU automaticamente,
#               acelerando a geração de embeddings (~4× mais rápido que CPU).

import numpy as np
from sentence_transformers import SentenceTransformer

from src.domain.ports import EmbedderPort
from src.infrastructure.config_loader import load_config

# multilingual-e5-large escolhido sobre alternativas por três razões:
# 1. Suporte nativo a português sem fine-tuning adicional — treinado em 100 línguas
#    incluindo pt-BR, crítico para os PDFs do IPARDES
# 2. Melhor recall@10 em benchmarks MTEB multilíngues que BGE-M3, com menor
#    uso de VRAM (560 MB vs ~1.5 GB do BGE-M3 em float32)
# 3. Requisito de prefixo "query:"/"passage:" força distinção semântica entre
#    embedding de busca e embedding de indexação, reduzindo falsos positivos


class SentenceTransformerEmbedder(EmbedderPort):
    def __init__(self) -> None:
        cfg = load_config()
        models_path = cfg["paths"]["models"]
        emb_cfg = cfg["embedding"]
        model_name: str = emb_cfg["model_name"]
        self._batch_size: int = emb_cfg["batch_size"]

        os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_path
        os.environ["TRANSFORMERS_CACHE"] = models_path

        # device=None: SentenceTransformer auto-detecta GPU/CPU via CUDA_VISIBLE_DEVICES.
        # Quando api.py define CUDA_VISIBLE_DEVICES="" antes dos imports, PyTorch não enxerga
        # nenhuma GPU e cai em CPU — mantendo a VRAM livre para o Ollama (llama3.2:3b).
        # Quando ingest.py roda (sem CUDA_VISIBLE_DEVICES), a GPU é usada automaticamente.
        self._model = SentenceTransformer(model_name, cache_folder=models_path)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed passages (chunks) com prefixo exigido pelo e5."""
        return self._encode_with_prefix(texts, prefix="passage: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed queries de busca com prefixo exigido pelo e5."""
        return self._encode_with_prefix(texts, prefix="query: ")

    def _encode_with_prefix(self, texts: list[str], prefix: str) -> np.ndarray:
        prefixed = [prefix + t for t in texts]
        # normalize_embeddings=True aplica normalização L2 inline — evita passo
        # extra e garante que similaridade de cosseno == produto interno,
        # requisito para o índice FAISS Inner Product que usaremos
        return self._model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
