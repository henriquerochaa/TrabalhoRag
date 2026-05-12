#!/usr/bin/env python3
"""
scripts/benchmark_chunk_size.py — encontra o menor truncamento de chunk que
ainda retorna resposta correta para a pergunta Cat. A de referência.

Roda no host (não no Docker): conecta ao Ollama em localhost:11434 diretamente,
sem passar pela API FastAPI. Isso evita o overhead do Docker e isola a variável
"tamanho do chunk" como única diferença entre as três execuções.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config_loader import load_config
from src.domain.entities import Chunk
from src.infrastructure.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.infrastructure.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.storage.faiss_vector_store import FAISSVectorStore
from src.infrastructure.storage.sqlite_metadata_store import SQLiteMetadataStore
from src.application.search_chunks import SearchChunks

# ── Pergunta de referência Cat. A ─────────────────────────────────────────────
# Escolhida por exigir um número específico ("22,27%") que só existe em
# analise_conjuntural_2025 — facilita avaliar se a informação foi preservada
# no chunk truncado.
QUESTION = (
    "Qual foi a variação percentual da carteira de crédito a pessoas "
    "físicas no Paraná entre junho de 2024 e junho de 2025?"
)
# Substring obrigatória para considerar a resposta "correta" neste benchmark.
# O valor exato está na p.4 de analise_conjuntural_2025 — se aparecer na
# resposta, o chunk com a informação sobreviveu à truncagem.
EXPECTED_SUBSTRING = "22,27"

TRUNCATION_SIZES = [200, 300, 400]

_INSTRUCTION = """\
Você é um assistente especializado nos documentos do IPARDES sobre o estado do Paraná.
Responda EXCLUSIVAMENTE com base nos trechos fornecidos abaixo.
Se a resposta não estiver nos trechos, responda: "Não encontrei essa informação nos documentos fornecidos."
Não suponha, não complete e não extrapole informações que não estejam explicitamente nos trechos."""

_CONTEXT_HEADER = "\n\n### TRECHOS DOS DOCUMENTOS\n"
_QUESTION_HEADER = "\n\n### PERGUNTA\n"
_ANSWER_LEAD = "\n\n### RESPOSTA\n"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _format_chunk(index: int, chunk: Chunk, max_chars: int) -> str:
    section_part = f" | Seção: {chunk.section}" if chunk.section else ""
    text = chunk.text[:max_chars]
    return (
        f"\n--- Trecho {index} ---\n"
        f"Documento: {chunk.document_id} | Página: {chunk.page}{section_part}\n\n"
        f"{text}\n"
    )


def _build_prompt(query: str, chunks: list[Chunk], max_chars: int) -> tuple[str, int]:
    context_block = "".join(_format_chunk(i + 1, c, max_chars) for i, c in enumerate(chunks))
    prompt = (
        _INSTRUCTION
        + _CONTEXT_HEADER
        + context_block
        + _QUESTION_HEADER
        + query
        + _ANSWER_LEAD
    )
    return prompt, _estimate_tokens(prompt)


def _call_ollama(prompt: str, cfg: dict) -> tuple[str, float]:
    # Rodando no host: usa localhost em vez de host.docker.internal
    base_url = cfg["llm"]["base_url"].replace("host.docker.internal", "localhost")
    url = f"{base_url}/api/generate"
    payload = json.dumps({
        "model": cfg["llm"]["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg["llm"]["temperature"],
            "num_predict": cfg["llm"]["max_tokens"],
        },
    }).encode()

    req = urllib.request.Request(
        url=url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=cfg["llm"]["timeout_seconds"]) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.monotonic() - t0
    return body["response"], elapsed


def main() -> None:
    cfg = load_config()

    print("Inicializando embedder, vector store e reranker…")
    embedder = SentenceTransformerEmbedder()
    meta_store = SQLiteMetadataStore()
    vector_store = FAISSVectorStore(meta_store)
    reranker = CrossEncoderReranker()

    # Carrega índice do disco — mesmo fluxo do lazy loading em api.py
    processed = str(_PROJECT_ROOT / cfg["paths"]["processed"])
    meta_store.load(processed)
    vector_store.load(processed)

    search = SearchChunks(embedder, vector_store, reranker)

    print(f"Buscando chunks para a pergunta de referência…\n  → {QUESTION[:80]}…\n")
    chunks, out_of_scope = search.execute(QUESTION)

    if out_of_scope or not chunks:
        print("ERRO: nenhum chunk retornado — verifique o índice FAISS e o threshold.")
        sys.exit(1)

    print(f"Chunks recuperados: {len(chunks)}\n")

    sep = "─" * 72
    results: list[dict] = []

    for max_chars in TRUNCATION_SIZES:
        print(sep)
        print(f"  max_chars = {max_chars}")

        prompt, token_est = _build_prompt(QUESTION, chunks, max_chars)

        try:
            answer, elapsed = _call_ollama(prompt, cfg)
            correct = EXPECTED_SUBSTRING in answer
        except Exception as exc:
            answer = f"[ERRO: {exc}]"
            elapsed = -1.0
            correct = False

        print(f"  Tempo:          {elapsed:.1f}s")
        print(f"  Tokens estimados no prompt: {token_est}")
        print(f"  Contém '{EXPECTED_SUBSTRING}': {'SIM ✓' if correct else 'NÃO ✗'}")
        print(f"  Resposta: {answer[:200]}{'…' if len(answer) > 200 else ''}\n")

        results.append({
            "max_chars": max_chars,
            "token_est": token_est,
            "elapsed": round(elapsed, 1),
            "correct": correct,
            "answer": answer,
        })

    # ── Escolha automática do menor valor correto ──────────────────────────────
    print("═" * 72)
    print("RESULTADO")
    print("═" * 72)
    correct_results = [r for r in results if r["correct"]]

    if not correct_results:
        print("Nenhum tamanho retornou resposta correta.")
        print("Sugestão: aumentar timeout_seconds ou usar modelo maior.")
        sys.exit(1)

    best = min(correct_results, key=lambda r: r["max_chars"])
    print(f"Menor tamanho com resposta correta: {best['max_chars']} chars")
    print(f"  → Tokens estimados: {best['token_est']}")
    print(f"  → Tempo de resposta: {best['elapsed']}s")
    print()

    for r in results:
        status = "CORRETO ✓" if r["correct"] else "FALHOU  ✗"
        timeout_risk = " ⚠ próximo do limite" if r["elapsed"] > cfg["llm"]["timeout_seconds"] * 0.85 else ""
        print(f"  {r['max_chars']} chars | {r['elapsed']:6.1f}s | {r['token_est']:4d} tokens | {status}{timeout_risk}")

    print()
    print(f"Recomendação: definir max_chunk_chars: {best['max_chars']} em config.yaml")


if __name__ == "__main__":
    main()
