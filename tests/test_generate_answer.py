from __future__ import annotations

import numpy as np

from src.application.generate_answer import GenerateAnswer, _OUT_OF_SCOPE_TEXT
from src.application.prompt_builder import PromptBuilder
from src.application.search_chunks import SearchChunks
from src.domain.entities import Chunk, SearchResult
from src.domain.ports import EmbedderPort, LLMPort, RerankerPort, VectorStorePort


# ---------------------------------------------------------------------------
# helpers compartilhados
# ---------------------------------------------------------------------------

def _chunk(cid: str, text: str = "conteúdo relevante sobre o Paraná") -> Chunk:
    return Chunk(id=cid, document_id="doc", text=text, page=1, section="", position=0)


class _StubEmbedder(EmbedderPort):
    # Vetores ortogonais entre chunks e query constante: MMR preserva ordem do store.
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.eye(len(texts), dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


class _PassthroughReranker(RerankerPort):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return chunks


class _FixedLLM(LLMPort):
    def __init__(self, response: str = "Resposta gerada.") -> None:
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


def _make_store(pairs: list[tuple[Chunk, float]]) -> VectorStorePort:
    class _Store(VectorStorePort):
        def add(self, c, e): pass
        def search(self, emb, top_k): return [SearchResult(c, s) for c, s in pairs]
        def save(self, p): pass
        def load(self, p): pass
    return _Store()


def _make_ga(
    pairs: list[tuple[Chunk, float]],
    llm: LLMPort | None = None,
    pb: PromptBuilder | None = None,
) -> GenerateAnswer:
    search = SearchChunks(_StubEmbedder(), _make_store(pairs), _PassthroughReranker())
    return GenerateAnswer(
        search,
        pb if pb is not None else PromptBuilder(),
        llm if llm is not None else _FixedLLM(),
    )


# ---------------------------------------------------------------------------
# resposta fora do escopo retorna exatamente o texto padrão definido
# ---------------------------------------------------------------------------

class TestOutOfScope:
    def test_returns_exact_standard_text(self) -> None:
        # 0.10 < 0.65 (min_score_threshold) — qualquer score abaixo dispara out_of_scope
        ans = _make_ga([(_chunk("low"), 0.10)]).execute("Qual a capital da França?")
        assert ans.text == _OUT_OF_SCOPE_TEXT

    def test_sources_is_empty(self) -> None:
        ans = _make_ga([(_chunk("low"), 0.10)]).execute("query irrelevante")
        assert ans.sources == []

    def test_prompt_used_is_empty_string(self) -> None:
        # prompt_used="" sinaliza que a LLM não foi consultada — detectável downstream
        ans = _make_ga([(_chunk("low"), 0.10)]).execute("query irrelevante")
        assert ans.prompt_used == ""

    def test_out_of_scope_flag_is_true(self) -> None:
        ans = _make_ga([(_chunk("low"), 0.10)]).execute("query irrelevante")
        assert ans.out_of_scope is True

    def test_llm_not_called_when_out_of_scope(self) -> None:
        # LLM que levanta exceção garante que não houve chamada à inferência.
        # Importante: evitar custo de inferência e alucinação para queries sem cobertura.
        class _ExplodingLLM(LLMPort):
            def generate(self, prompt: str) -> str:
                raise AssertionError("LLM não deve ser chamada para query fora do escopo")

        ans = _make_ga([(_chunk("low"), 0.10)], llm=_ExplodingLLM()).execute("fora do escopo")
        assert ans.out_of_scope is True


# ---------------------------------------------------------------------------
# prompt_used está preenchido em todas as respostas dentro do escopo
# ---------------------------------------------------------------------------

class TestPromptUsed:
    def test_prompt_used_is_nonempty(self) -> None:
        ans = _make_ga([(_chunk("ok"), 0.80)]).execute("Qual o PIB do Paraná?")
        assert ans.prompt_used != ""

    def test_prompt_used_contains_query(self) -> None:
        # A query deve aparecer literalmente no prompt para que a LLM responda
        # à pergunta correta — modelos pequenos seguem o texto do prompt, não inferem.
        query = "Qual o crescimento do PIB do Paraná em 2024?"
        ans = _make_ga([(_chunk("ok", "O PIB cresceu 4%."), 0.80)]).execute(query)
        assert query in ans.prompt_used

    def test_prompt_used_contains_chunk_text(self) -> None:
        chunk_text = "A taxa de desemprego caiu para 6% em 2024, menor que a média nacional."
        ans = _make_ga([(_chunk("ok", chunk_text), 0.80)]).execute("Desemprego no Paraná?")
        assert chunk_text in ans.prompt_used

    def test_out_of_scope_false_for_covered_question(self) -> None:
        ans = _make_ga([(_chunk("ok"), 0.80)]).execute("Qual o PIB do Paraná?")
        assert ans.out_of_scope is False


# ---------------------------------------------------------------------------
# sources contém apenas chunks realmente usados no prompt
# ---------------------------------------------------------------------------

class TestSources:
    def test_every_source_text_appears_in_prompt_used(self) -> None:
        # Invariante central: sources jamais pode incluir chunk ausente do prompt.
        # Viola o contrato de Answer se fonte citada não embasou a resposta gerada.
        c1 = _chunk("c1", "Exportações paranaenses cresceram 12% em 2024.")
        c2 = _chunk("c2", "Indústria automotiva representa 30% do PIB industrial.")
        ans = _make_ga([(c1, 0.90), (c2, 0.80)]).execute("exportações do Paraná")
        for src in ans.sources:
            assert src.text in ans.prompt_used, (
                f"chunk '{src.id}' listado em sources mas ausente do prompt_used"
            )

    def test_chunk_dropped_by_tight_budget_not_in_sources(self) -> None:
        # 3 chunks de 2000 chars → 511 tokens cada (medido: block=2047 chars).
        # skeleton ≈ 105 tokens; budget=650 → chunk_budget=545.
        # 1 chunk (511) cabe; 2 (1022) não cabem → rank1 e rank2 devem ser descartados.
        text = "P" * 2000
        c0 = _chunk("rank0", text)
        c1 = _chunk("rank1", text)
        c2 = _chunk("rank2", text)

        pb = PromptBuilder()
        pb._input_budget = 650

        ans = _make_ga([(c0, 0.90), (c1, 0.80), (c2, 0.70)], pb=pb).execute("query")

        source_ids = {c.id for c in ans.sources}
        assert "rank0" in source_ids, "chunk de maior score deve ser preservado"
        assert "rank1" not in source_ids
        assert "rank2" not in source_ids

    def test_sources_matches_prompt_builder_output_exactly(self) -> None:
        # Captura os chunks que PromptBuilder efetivamente usou e compara com
        # sources da Answer — garantem que GenerateAnswer não manipula a lista.
        captured: list[Chunk] = []

        class _CapturingPB(PromptBuilder):
            def build(self, query: str, chunks: list[Chunk]) -> tuple[str, list[Chunk]]:
                prompt, used = super().build(query, chunks)
                captured.extend(used)
                return prompt, used

        c1 = _chunk("c1", "Dado A sobre agronegócio paranaense.")
        c2 = _chunk("c2", "Dado B sobre produção de soja.")

        ans = _make_ga([(c1, 0.90), (c2, 0.80)], pb=_CapturingPB()).execute("agronegócio")

        assert [c.id for c in ans.sources] == [c.id for c in captured]
