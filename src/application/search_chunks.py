from __future__ import annotations

from collections import Counter

import numpy as np

from src.config_loader import load_config
from src.domain.entities import Chunk, SearchResult
from src.domain.ports import EmbedderPort, RerankerPort, VectorStorePort

# MMR (Maximal Marginal Relevance) escolhido sobre deduplicação simples por três razões:
# 1. Dedup remove apenas duplicatas textuais/quase-idênticas dentro do mesmo doc.
#    MMR seleciona ativamente chunks que cobrem facetas diferentes da query —
#    fundamental num corpus multi-documento (3 PDFs) onde respostas completas
#    podem exigir evidências de fontes distintas.
# 2. O parâmetro λ controla explicitamente o trade-off relevância/diversidade:
#    λ=1 equivale ao ranking FAISS original; λ=0 maximiza diversidade pura.
#    λ=0.5 (default) balanceia os dois, ajustável sem tocar no código.
# 3. Dedup opera no espaço de texto (hash/Jaccard), MMR opera no espaço de
#    embeddings — permite detectar paráfrases semânticas que dedup textual perde.
# Referência: Carbonell & Goldstein (1998), ACM SIGIR — proposta original do MMR.


class SearchChunks:
    def __init__(
        self,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        reranker: RerankerPort,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        full_cfg = load_config()
        cfg = full_cfg["retrieval"]
        self._top_k_initial: int = cfg["top_k_initial"]
        self._top_k_final: int = cfg["top_k_final"]
        self._threshold: float = cfg["min_score_threshold"]
        self._lambda_mmr: float = cfg["lambda_mmr"]
        self._context_window: int = full_cfg["llm"]["context_window"]
        self._max_tokens: int = full_cfg["llm"]["max_tokens"]
        # Budget de tokens disponível para os chunks: descontamos max_tokens porque
        # esse espaço fica reservado para a geração da resposta pela LLM. Ultrapassar
        # esse limite causaria truncagem silenciosa do prompt pelo Ollama — que corta
        # pelo início, descartando exatamente o contexto que o RAG forneceu.
        self._token_budget: int = self._context_window - self._max_tokens

    def execute(self, query: str) -> tuple[list[Chunk], bool]:
        """Retorna (chunks, out_of_scope).

        out_of_scope=True quando nenhum resultado ultrapassa min_score_threshold,
        indicando que a query está fora do escopo dos PDFs do IPARDES.
        """
        query_emb: np.ndarray = self._embedder.embed_queries([query])[0]

        results: list[SearchResult] = self._vector_store.search(
            query_emb, top_k=self._top_k_initial
        )

        if not results or results[0].score < self._threshold:
            return [], True

        diverse = self._mmr(query_emb, results, self._top_k_final, self._lambda_mmr)

        # score_map preserva a similaridade cosine do FAISS por chunk_id.
        # O reranker (cross-encoder) apenas reordena — não recalcula scores FAISS;
        # usar o score do reranker seria incomparável entre queries diferentes
        # porque a escala do cross-encoder não é normalizada entre 0 e 1.
        score_map: dict[str, float] = {r.chunk.id: r.score for r in diverse}

        reranked: list[Chunk] = self._reranker.rerank(query, [r.chunk for r in diverse])
        compressed = self._compress_to_context(reranked[: self._top_k_final])

        for chunk in compressed:
            chunk.score = score_map.get(chunk.id, 0.0)

        return compressed, False

    # ------------------------------------------------------------------

    def _mmr(
        self,
        query_emb: np.ndarray,
        results: list[SearchResult],
        top_k: int,
        lam: float,
    ) -> list[SearchResult]:
        # Re-embeda os chunks como passagens para calcular similaridade inter-chunk.
        # Não reutiliza vetores do FAISS porque VectorStorePort não os expõe —
        # expor vetores internos quebraria o encapsulamento da camada de storage.
        texts = [r.chunk.text for r in results]
        chunk_embs: np.ndarray = self._embedder.embed(texts)  # (N, dim), L2-norm

        selected: list[int] = []
        candidates = list(range(len(results)))
        doc_count: Counter[str] = Counter()

        # Cap por documento: no máximo (top_k - 1) chunks do mesmo documento.
        # Garante que quando o pool contém 2+ documentos, ao menos 2 aparecem
        # no resultado — necessário porque o PDF maior (desenvolvimento, 3.8 MB)
        # domina o top-20 FAISS com 19/20 chunks, bloqueando diversidade multi-doc.
        # Fallback sem cap (linha abaixo do loop) preserva corretude quando
        # todos os candidatos elegíveis se esgotam antes de preencher top_k.
        max_per_doc: int = max(1, top_k - 1)

        while len(selected) < top_k and candidates:
            best_idx: int = candidates[0]
            best_score = float("-inf")

            for i in candidates:
                if doc_count[results[i].chunk.document_id] >= max_per_doc:
                    continue  # documento já atingiu o limite; tenta próximo

                relevance = results[i].score  # cosine sim via FAISS inner product

                if not selected:
                    diversity = 0.0
                else:
                    # máxima similaridade com qualquer chunk já selecionado
                    sims = chunk_embs[i] @ chunk_embs[selected].T
                    diversity = float(np.max(sims))

                mmr_score = lam * relevance - (1.0 - lam) * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_score == float("-inf"):
                # todos os candidatos restantes atingiram o cap — relaxa e pega
                # o de maior relevância para não desperdiçar slots de resultado
                best_idx = max(candidates, key=lambda i: results[i].score)

            doc_count[results[best_idx].chunk.document_id] += 1
            selected.append(best_idx)
            candidates.remove(best_idx)

        return [results[i] for i in selected]

    def _compress_to_context(self, chunks: list[Chunk]) -> list[Chunk]:
        # Estratégia: descartar chunks de menor score (posições finais da lista,
        # já ordenada pelo reranker de melhor para pior) até o total estimado de
        # tokens caber em _token_budget = context_window - max_tokens.
        #
        # Por que descarte por score e não truncagem no meio de um chunk:
        # Modelos <= 9.9B têm janela de atenção menor e são mais sensíveis a
        # fragmentos incompletos no contexto — uma sentença cortada no meio tende
        # a ser completada pela LLM com alucinação, pois o modelo "fecha" o raciocínio
        # com seus pesos em vez de se limitar à evidência. Descartar o chunk inteiro
        # preserva a integridade semântica de cada evidência fornecida; a perda de
        # recall é preferível à introdução de ruído estrutural que degrada a
        # confiabilidade do output — que é o critério de avaliação de anti-alucinação.
        #
        # Por que descartar por score e não por posição ou tamanho:
        # O reranker já ordenou os chunks pela pertinência à query; os últimos da
        # lista têm a menor contribuição marginal para a resposta. Remover por score
        # minimiza a perda de informação por token descartado.
        #
        # Estimativa de tokens: chars // 4 é conservador para português (média real
        # ~4.2 chars/token). Arredondar para baixo superestima tokens marginalmente —
        # preferível a subestimar e ultrapassar a janela da LLM.
        def _estimate_tokens(c: Chunk) -> int:
            return max(1, len(c.text) // 4)

        if sum(_estimate_tokens(c) for c in chunks) <= self._token_budget:
            return chunks

        result = list(chunks)
        while result and sum(_estimate_tokens(c) for c in result) > self._token_budget:
            result.pop()
        return result
