#!/usr/bin/env python3
"""
scripts/run_evaluation.py — avaliação end-to-end do pipeline RAG via HTTP.

Separação entre ingest e evaluation (justificativa):
  - ingest (ingest.py): constrói o índice vetorial a partir dos PDFs —
    operação cara (~minutos), executada uma vez ou quando os documentos
    mudam. Misturar indexação com avaliação forçaria re-indexar a cada
    iteração de qualidade, inviável em produção.
  - evaluation (este script): consulta o índice existente medindo qualidade
    end-to-end via HTTP — operação barata (~segundos), executada repetidamente
    após mudanças em prompts, threshold de retrieval ou chunking. Medir antes
    e depois de cada ajuste é o único jeito de saber se o RAG melhorou.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# garante imports do src/ independente do cwd
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constante espelhada de src/application/generate_answer.py — texto exato
# que a API retorna para perguntas fora do escopo
# ---------------------------------------------------------------------------
_OUT_OF_SCOPE_TEXT = "O assunto não está coberto pelos documentos disponíveis."


def _doc_id(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Dataset — 11 perguntas em 3 categorias (espelho de tests/test_evaluation.py)
# ---------------------------------------------------------------------------
DATASET: list[dict] = [
    # ── Categoria A — por documento ─────────────────────────────────────────
    {
        "id": 1,
        "category": "A",
        "text": (
            "Qual foi a variação percentual da carteira de crédito a pessoas "
            "físicas no Paraná entre junho de 2024 e junho de 2025?"
        ),
        "justification": (
            '"22,27% em termos reais" é estatística única em '
            "analise_conjuntural_2025 p.4"
        ),
        "expected_doc_id": _doc_id("analise_conjuntural_2025.pdf"),
        "expected_doc_label": "analise_conjuntural_2025.pdf",
        "expected_out_of_scope": False,
    },
    {
        "id": 2,
        "category": "A",
        "text": (
            "Desde qual trimestre o Paraná registra taxa de desocupação "
            "inferior a 5%?"
        ),
        "justification": (
            '"segundo trimestre de 2023" está exclusivamente em '
            "analise_conjuntural_2025 p.5"
        ),
        "expected_doc_id": _doc_id("analise_conjuntural_2025.pdf"),
        "expected_doc_label": "analise_conjuntural_2025.pdf",
        "expected_out_of_scope": False,
    },
    {
        "id": 3,
        "category": "A",
        "text": (
            "Quais são os componentes da decomposição da variação de "
            "produtividade do trabalho analisados no estudo sobre o Paraná?"
        ),
        "justification": (
            "Metodologia de decomposição em 3 efeitos está nas pp.18-21 de "
            "desenvolvimento_paranaense, ausente nos outros PDFs"
        ),
        "expected_doc_id": _doc_id("desenvolvimento_paranaense.pdf"),
        "expected_doc_label": "desenvolvimento_paranaense.pdf",
        "expected_out_of_scope": False,
    },
    {
        "id": 4,
        "category": "A",
        "text": (
            "Qual protocolo foi adotado para garantir a transparência e "
            "reprodutibilidade da revisão de escopo de políticas públicas "
            "brasileiras?"
        ),
        "justification": (
            '"PRISMA ScR" mencionado exclusivamente em '
            "avaliacoes_politicas_publicas p.7"
        ),
        "expected_doc_id": _doc_id("avaliacoes_politicas_publicas.pdf"),
        "expected_doc_label": "avaliacoes_politicas_publicas.pdf",
        "expected_out_of_scope": False,
    },
    # ── Categoria B — fora do escopo ────────────────────────────────────────
    {
        "id": 5,
        "category": "B",
        "text": "Qual é a distância em quilômetros entre São Paulo e Buenos Aires?",
        "justification": "geografia internacional, ausente de todos os PDFs IPARDES",
        "expected_out_of_scope": True,
    },
    {
        "id": 6,
        "category": "B",
        "text": (
            "Como funciona o protocolo de consenso Proof of Work utilizado "
            "no Bitcoin?"
        ),
        "justification": "tecnologia blockchain, sem ocorrência nos 3 documentos",
        "expected_out_of_scope": True,
    },
    {
        "id": 7,
        "category": "B",
        "text": (
            "Quais são as regras do campeonato mundial de xadrez segundo a FIDE?"
        ),
        "justification": "esporte, zero interseção temática com os PDFs",
        "expected_out_of_scope": True,
    },
    {
        "id": 8,
        "category": "B",
        "text": (
            "Qual é a estrutura eletrônica do átomo de carbono segundo a "
            "tabela periódica?"
        ),
        "justification": "química/física, fora do escopo socioeconômico do IPARDES",
        "expected_out_of_scope": True,
    },
    # ── Categoria C — multi-documento ───────────────────────────────────────
    {
        "id": 9,
        "category": "C",
        "text": (
            "Qual é a situação do mercado de trabalho paranaense e como ele "
            "se relaciona com a produtividade do trabalho no estado?"
        ),
        "justification": (
            "mercado de trabalho em analise_conjuntural; "
            "produtividade do trabalho em desenvolvimento_paranaense"
        ),
        "expected_out_of_scope": False,
    },
    {
        "id": 10,
        "category": "C",
        "text": (
            "Quais são as principais exportações agrícolas do Paraná e qual "
            "é a importância do agronegócio para a economia estadual?"
        ),
        "justification": (
            "exportações (frango/suíno) em analise_conjuntural; "
            "importância econômica do agronegócio em desenvolvimento_paranaense"
        ),
        "expected_out_of_scope": False,
    },
    {
        "id": 11,
        "category": "C",
        "text": (
            "Quais metodologias são usadas para avaliar políticas públicas e "
            "como essas avaliações contribuem para o desenvolvimento econômico "
            "do Paraná?"
        ),
        "justification": (
            "metodologias de avaliação em avaliacoes_politicas_publicas; "
            "desenvolvimento econômico em desenvolvimento_paranaense"
        ),
        "expected_out_of_scope": False,
    },
]


# ---------------------------------------------------------------------------
# Verificação de pré-condições
# ---------------------------------------------------------------------------

def _check_pdfs(cfg: dict) -> list[str]:
    raw_dir = _PROJECT_ROOT / cfg["paths"]["raw"]
    filenames = [e["filename"] for e in cfg["pdfs"]]
    missing = [f for f in filenames if not (raw_dir / f).exists()]
    return missing


def _check_index(cfg: dict) -> bool:
    processed = _PROJECT_ROOT / cfg["paths"]["processed"]
    return (processed / "index.faiss").exists() and (processed / "metadata.db").exists()


def _check_http(url: str, timeout: int = 8) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def _verify_preconditions(cfg: dict) -> bool:
    ok = True

    # 1. PDFs
    missing_pdfs = _check_pdfs(cfg)
    if missing_pdfs:
        _log.error(
            "PDFs ausentes em %s: %s\n"
            "  → execute: python scripts/download_pdfs.py",
            cfg["paths"]["raw"], missing_pdfs,
        )
        ok = False
    else:
        _log.info("PDFs: OK (%d arquivos encontrados)", len(cfg["pdfs"]))

    # 2. Índice FAISS + SQLite
    if not _check_index(cfg):
        _log.info(
            "Índice FAISS ausente em %s — executando ingest automaticamente...",
            cfg["paths"]["processed"],
        )
        _run_ingest()
        if not _check_index(cfg):
            _log.error(
                "Ingest falhou — index.faiss não encontrado em %s\n"
                "  → execute manualmente: python ingest.py",
                cfg["paths"]["processed"],
            )
            ok = False
        else:
            _log.info("Índice criado com sucesso.")
    else:
        _log.info("Índice FAISS + SQLite: OK")

    # 3. Ollama — este script roda no host, não dentro do Docker,
    # por isso usa localhost mesmo que config.yaml aponte para host.docker.internal
    ollama_host_url = "http://localhost:11434/api/tags"
    if not _check_http(ollama_host_url):
        _log.error(
            "Ollama não responde em %s\n"
            "  → verifique: ollama serve  (ou systemctl status ollama)",
            ollama_host_url,
        )
        ok = False
    else:
        _log.info("Ollama: OK (%s)", ollama_host_url)

    # 4. API FastAPI
    api_base = cfg["api"]["base_url"]
    if not _check_http(f"{api_base}/docs"):
        _log.error(
            "API FastAPI não responde em %s\n"
            "  → execute: docker-compose up api  (ou uvicorn src.interface.api:app --port 8000)",
            api_base,
        )
        ok = False
    else:
        _log.info("API FastAPI: OK (%s)", api_base)

    return ok


def _warmup_pipeline(api_base: str) -> None:
    """Força o carregamento do pipeline (embedder + FAISS + reranker + Ollama) antes
    da avaliação cronometrada.

    Sem este aquecimento, a primeira pergunta absorve o tempo de inicialização a frio
    (~60-90s para carregar multilingual-e5-large, ms-marco e o índice FAISS),
    somado ao carregamento do llama3.2:3b no Ollama (~20s), facilmente excedendo
    o timeout de 190s e causando falha espúria na primeira pergunta.
    """
    _log.info("Aquecendo pipeline (primeira carga pode levar até 120s)…")
    t0 = time.monotonic()
    try:
        _call_chat(api_base, "Qual é a situação econômica do Paraná?", timeout=300)
        elapsed = time.monotonic() - t0
        _log.info("Pipeline pronto em %.1fs.", elapsed)
    except (urllib.error.URLError, OSError) as exc:
        elapsed = time.monotonic() - t0
        _log.warning("Warmup falhou em %.1fs (%s) — avaliação pode ter timeout na pergunta 1.", elapsed, exc)


def _run_ingest() -> None:
    ingest_path = _PROJECT_ROOT / "ingest.py"
    _log.info("Iniciando ingest.py...")
    result = subprocess.run(
        [sys.executable, str(ingest_path)],
        cwd=str(_PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        _log.error("ingest.py terminou com código %d", result.returncode)


# ---------------------------------------------------------------------------
# Chamada ao endpoint /chat
# ---------------------------------------------------------------------------

def _call_chat(api_base: str, question: str, timeout: int = 180) -> dict:
    payload = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        url=f"{api_base}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Avaliação por categoria
# ---------------------------------------------------------------------------

def _evaluate(q: dict, response: dict) -> tuple[bool, str]:
    """Retorna (passou, motivo_falha)."""
    cat = q["category"]
    sources: list[dict] = response.get("sources", [])
    out_of_scope: bool = response.get("out_of_scope", False)
    answer: str = response.get("answer", "")

    if cat == "A":
        if out_of_scope:
            return False, "out_of_scope=True inesperado para pergunta coberta pelo corpus"
        doc_ids = {s["document_id"] for s in sources}
        if q["expected_doc_id"] not in doc_ids:
            return False, (
                f"source de '{q['expected_doc_label']}' não encontrado; "
                f"docs retornados: {doc_ids or '(nenhum)'}"
            )
        return True, ""

    if cat == "B":
        if not out_of_scope:
            return False, "out_of_scope=False — modelo deveria reconhecer que pergunta está fora do escopo"
        if answer != _OUT_OF_SCOPE_TEXT:
            return False, f"texto diferente do padrão; recebido: {answer[:80]!r}"
        if sources:
            return False, f"sources não vazio ({len(sources)} chunks) para pergunta fora do escopo"
        return True, ""

    if cat == "C":
        if out_of_scope:
            return False, "out_of_scope=True inesperado — pergunta exige chunks de múltiplos docs"
        doc_ids = {s["document_id"] for s in sources}
        if len(doc_ids) < 2:
            return False, (
                f"apenas {len(doc_ids)} documento(s) nos sources — "
                "esperado >= 2 para pergunta multi-documento"
            )
        return True, ""

    return False, f"categoria desconhecida: {cat!r}"


# ---------------------------------------------------------------------------
# Formatação de output
# ---------------------------------------------------------------------------

def _print_result(q: dict, response: dict, passed: bool, reason: str, elapsed: float) -> None:
    sep = "─" * 72
    status_label = "✓ PASSOU" if passed else "✗ FALHOU"

    print(f"\n{sep}")
    print(f"[{q['id']:02d}] Categoria {q['category']}  |  {status_label}  |  {elapsed:.1f}s")
    print(f"Pergunta:      {q['text']}")
    print(f"Justificativa: {q['justification']}")
    if not passed:
        print(f"Motivo:        {reason}")

    print(f"out_of_scope:  {response.get('out_of_scope')}")

    sources = response.get("sources", [])
    scores = [s["score"] for s in sources if "score" in s]
    if scores:
        print(f"Score médio:   {sum(scores) / len(scores):.4f}  (min={min(scores):.4f}  max={max(scores):.4f})")
    else:
        print("Score médio:   N/A (sem sources)")
    if sources:
        print("Sources:")
        for s in sources:
            sec = f" | seção: {s['section']}" if s.get("section") else ""
            sc = f"  score={s['score']:.4f}" if "score" in s else ""
            print(f"  • doc: {s['document_id'][:8]}…  p.{s['page']}{sec}{sc}")
    else:
        print("Sources:       (nenhum)")

    prompt = response.get("prompt_used", "")
    if prompt:
        print(f"prompt_used:   {prompt[:200]}{'…' if len(prompt) > 200 else ''}")
    else:
        print("prompt_used:   (vazio — LLM não foi consultada)")

    answer = response.get("answer", "")
    print(f"Resposta:      {answer[:300]}{'…' if len(answer) > 300 else ''}")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    api_base = cfg["api"]["base_url"]

    _log.info("=== Avaliação RAG IPARDES — %d perguntas ===", len(DATASET))

    if not _verify_preconditions(cfg):
        _log.error("Pré-condições não satisfeitas — abortando.")
        sys.exit(1)

    _warmup_pipeline(api_base)

    results: list[dict] = []
    passed_total = 0
    b_correct = 0
    b_total = 0
    c_correct = 0
    c_total = 0

    for q in DATASET:
        _log.info("Executando pergunta %d/%d (Cat. %s)…", q["id"], len(DATASET), q["category"])
        t0 = time.monotonic()

        try:
            response = _call_chat(api_base, q["text"], timeout=cfg["llm"]["timeout_seconds"] + 10)
            elapsed = time.monotonic() - t0
        except (urllib.error.URLError, OSError) as exc:
            elapsed = time.monotonic() - t0
            _log.error("Pergunta %d falhou (rede): %s", q["id"], exc)
            response = {"answer": "", "prompt_used": "", "sources": [], "out_of_scope": False}
            passed, reason = False, f"erro de rede: {exc}"
        else:
            passed, reason = _evaluate(q, response)

        if passed:
            passed_total += 1
        if q["category"] == "B":
            b_total += 1
            if passed:
                b_correct += 1
        if q["category"] == "C":
            c_total += 1
            if passed:
                c_correct += 1

        _print_result(q, response, passed, reason, elapsed)

        q_sources = response.get("sources", [])
        q_scores = [s["score"] for s in q_sources if "score" in s]
        q_score_medio = round(sum(q_scores) / len(q_scores), 4) if q_scores else None

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["text"],
            "justification": q["justification"],
            "status": "PASSOU" if passed else "FALHOU",
            "failure_reason": reason if not passed else None,
            "elapsed_seconds": round(elapsed, 2),
            "out_of_scope": response.get("out_of_scope"),
            "score_medio": q_score_medio,
            "sources": q_sources,
            "prompt_used": response.get("prompt_used", ""),
            "answer": response.get("answer", ""),
        })

    # ── Resumo ───────────────────────────────────────────────────────────────
    failed_total = len(DATASET) - passed_total

    all_scores = [
        s["score"]
        for r in results
        for s in r.get("sources", [])
        if "score" in s
    ]
    score_medio_geral = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    score_label = f"{score_medio_geral:.4f}" if score_medio_geral is not None else "N/A (sem sources)"

    sep = "═" * 72
    print(f"\n{sep}")
    print("RESUMO")
    print(sep)
    print(f"Total de perguntas:              {len(DATASET)}")
    print(f"Passaram:                        {passed_total}")
    print(f"Falharam:                        {failed_total}")
    print(f"Score médio geral:               {score_label}")
    print(f"Categoria B — out_of_scope ok:   {b_correct}/{b_total}")
    print(f"Categoria C — multi-doc ok:      {c_correct}/{c_total}")
    print(sep)

    # ── Salva JSON ───────────────────────────────────────────────────────────
    processed_dir = _PROJECT_ROOT / cfg["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "evaluation_results.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(DATASET),
            "passed": passed_total,
            "failed": failed_total,
            "score_medio": score_medio_geral,
            "categoria_b_out_of_scope_rate": f"{b_correct}/{b_total}",
            "categoria_c_multi_doc_rate": f"{c_correct}/{c_total}",
        },
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _log.info("Resultados salvos em %s", output_path)


if __name__ == "__main__":
    main()
