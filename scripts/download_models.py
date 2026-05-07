"""
Script de setup — executar UMA VEZ com internet antes do runtime.
Define os caches de modelos HuggingFace para models/ antes de qualquer import,
garantindo que o runtime nunca precise de conexão para carregar modelos.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# env vars definidos ANTES de qualquer import de modelo para que as libs
# de transformers usem models/ como cache tanto no download quanto no runtime
_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = str(_ROOT / "models")

os.environ["TRANSFORMERS_CACHE"] = _MODELS_DIR
os.environ["HF_HOME"] = _MODELS_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _MODELS_DIR

sys.path.insert(0, str(_ROOT))

from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

from src.infrastructure.config_loader import load_config  # noqa: E402


def download_models() -> None:
    cfg = load_config()
    embedding_name: str = cfg["embedding"]["model_name"]
    reranker_name: str = cfg["reranker"]["model_name"]

    print(f"[embedding] baixando {embedding_name} ...")
    embedder = SentenceTransformer(embedding_name, cache_folder=_MODELS_DIR)
    # validação mínima: encoda uma frase e verifica shape do vetor
    vec = embedder.encode(["teste"])
    print(f"[embedding] OK — dim={vec.shape[1]}")

    print(f"[reranker]  baixando {reranker_name} ...")
    reranker = CrossEncoder(reranker_name, cache_dir=_MODELS_DIR)
    # validação mínima: pontua um par e verifica que retorna float
    score = reranker.predict([("query", "passagem")])
    print(f"[reranker]  OK — score de exemplo={score[0]:.4f}")

    print("Concluído. Modelos salvos em models/")


if __name__ == "__main__":
    download_models()
