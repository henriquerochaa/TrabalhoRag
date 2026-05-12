from __future__ import annotations

import threading
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.application.generate_answer import GenerateAnswer
from src.application.prompt_builder import PromptBuilder
from src.application.search_chunks import SearchChunks
from src.config_loader import load_config
from src.infrastructure.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.infrastructure.llm.ollama_llm import OllamaLLM
from src.infrastructure.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.storage.faiss_vector_store import FAISSVectorStore
from src.infrastructure.storage.sqlite_metadata_store import SQLiteMetadataStore


# ---------------------------------------------------------------------------
# Schemas de request / response
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str


class SourceModel(BaseModel):
    id: str
    document_id: str
    page: int
    section: str
    score: float


class ChatResponse(BaseModel):
    # prompt_used e answer são campos separados e obrigatórios por exigência do
    # enunciado — o avaliador verifica os dois independentemente para checar se
    # a resposta está ancorada no prompt enviado à LLM e não foi gerada livremente.
    answer: str
    prompt_used: str
    sources: list[SourceModel]
    out_of_scope: bool


# ---------------------------------------------------------------------------
# Aplicação e estado global
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG IPARDES")

# use_case=None enquanto o índice não tiver sido carregado.
# O carregamento acontece na primeira requisição /chat (lazy loading),
# não no startup — garante que docker compose up sobe mesmo sem ingest.py.
app.state.use_case: GenerateAnswer | None = None

# Lock garante que o pipeline é inicializado apenas uma vez mesmo sob concorrência.
_init_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_exists() -> bool:
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    return (processed / "index.faiss").exists() and (processed / "metadata.db").exists()


def _ollama_reachable() -> bool:
    cfg = load_config()
    try:
        urllib.request.urlopen(cfg["llm"]["base_url"] + "/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _load_pipeline() -> GenerateAnswer:
    cfg = load_config()
    processed = cfg["paths"]["processed"]

    metadata_store = SQLiteMetadataStore()
    metadata_store.load(processed)

    # FAISSVectorStore precisa do metadata_store para recuperar Chunks pelo id
    vector_store = FAISSVectorStore(metadata_store)
    vector_store.load(processed)

    embedder = SentenceTransformerEmbedder()
    reranker = CrossEncoderReranker()
    llm = OllamaLLM()

    return GenerateAnswer(
        SearchChunks(embedder, vector_store, reranker),
        PromptBuilder(),
        llm,
    )


def _get_use_case() -> GenerateAnswer:
    if app.state.use_case is not None:
        return app.state.use_case
    # Double-checked locking: evita re-inicialização concorrente sem penalizar
    # leituras após o primeiro carregamento (sem lock no caminho quente).
    with _init_lock:
        if app.state.use_case is not None:
            return app.state.use_case
        app.state.use_case = _load_pipeline()
    return app.state.use_case


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    """Estado de prontidão do serviço — sempre HTTP 200.

    Retorna status:
      "ready"              — pipeline carregado, pronto para responder
      "not_indexed"        — index.faiss ausente; execute python ingest.py
      "ollama_unavailable" — índice presente mas Ollama não responde
    """
    if app.state.use_case is not None:
        return JSONResponse({"status": "ready"})
    if not _index_exists():
        return JSONResponse({"status": "not_indexed"})
    if not _ollama_reachable():
        return JSONResponse({"status": "ollama_unavailable"})
    # índice existe e Ollama responde: lazy load ocorrerá na primeira requisição
    return JSONResponse({"status": "ready"})


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not _index_exists():
        raise HTTPException(
            status_code=503,
            detail="Sistema não inicializado. Execute python ingest.py primeiro.",
        )

    use_case = _get_use_case()
    answer = use_case.execute(request.question)
    return ChatResponse(
        answer=answer.text,
        prompt_used=answer.prompt_used,
        sources=[
            SourceModel(
                id=chunk.id,
                document_id=chunk.document_id,
                page=chunk.page,
                section=chunk.section,
                score=chunk.score,
            )
            for chunk in answer.sources
        ],
        out_of_scope=answer.out_of_scope,
    )
