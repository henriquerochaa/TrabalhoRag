from __future__ import annotations

import pytest

from src.application.prompt_builder import PromptBuilder, _MAX_CHUNK_CHARS
from src.domain.entities import Chunk


def _chunk(cid: str, text: str = "Conteúdo relevante sobre o Paraná.") -> Chunk:
    return Chunk(id=cid, document_id="doc", text=text, page=1, section="", position=0)


def _chunk_with(cid: str, *, page: int = 1, section: str = "", text: str = "texto") -> Chunk:
    return Chunk(id=cid, document_id="doc", text=text, page=page, section=section, position=0)


# ---------------------------------------------------------------------------
# Tipo de retorno
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_tuple(self) -> None:
        pb = PromptBuilder()
        result = pb.build("pergunta?", [_chunk("c1")])
        assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_is_str(self) -> None:
        pb = PromptBuilder()
        prompt, _ = pb.build("pergunta?", [_chunk("c1")])
        assert isinstance(prompt, str)

    def test_used_chunks_is_list_of_chunk(self) -> None:
        pb = PromptBuilder()
        _, used = pb.build("pergunta?", [_chunk("c1")])
        assert isinstance(used, list)
        assert all(isinstance(c, Chunk) for c in used)


# ---------------------------------------------------------------------------
# Conteúdo do prompt
# ---------------------------------------------------------------------------

class TestPromptContent:
    def test_query_present_in_prompt(self) -> None:
        pb = PromptBuilder()
        query = "Qual o PIB do Paraná em 2024?"
        prompt, _ = pb.build(query, [_chunk("c1")])
        assert query in prompt

    def test_instruction_present_in_prompt(self) -> None:
        # palavra âncora da instrução anti-alucinação — garante que o template foi montado
        pb = PromptBuilder()
        prompt, _ = pb.build("query?", [_chunk("c1")])
        assert "EXCLUSIVAMENTE" in prompt

    def test_chunk_text_truncated_at_max_chars(self) -> None:
        # PromptBuilder trunca o texto do chunk em _MAX_CHUNK_CHARS
        long_text = "X" * 1000
        pb = PromptBuilder()
        prompt, _ = pb.build("query?", [_chunk("c1", long_text)])
        assert "X" * (_MAX_CHUNK_CHARS + 1) not in prompt
        assert "X" * _MAX_CHUNK_CHARS in prompt

    def test_chunk_page_number_in_prompt(self) -> None:
        pb = PromptBuilder()
        chunk = _chunk_with("c1", page=42)
        prompt, _ = pb.build("query?", [chunk])
        assert "42" in prompt

    def test_section_in_prompt_when_non_empty(self) -> None:
        pb = PromptBuilder()
        chunk = _chunk_with("c1", section="DESENVOLVIMENTO ECONÔMICO", text="dado")
        prompt, _ = pb.build("query?", [chunk])
        assert "DESENVOLVIMENTO ECONÔMICO" in prompt

    def test_section_absent_when_empty_string(self) -> None:
        # chunk sem seção não deve injetar "Seção:" no prompt
        pb = PromptBuilder()
        chunk = _chunk("c1")  # section=""
        prompt, _ = pb.build("query?", [chunk])
        assert "Seção:" not in prompt

    def test_multiple_chunks_all_appear_in_prompt(self) -> None:
        pb = PromptBuilder()
        c1 = _chunk("c1", "Dado A sobre o agronegócio paranaense.")
        c2 = _chunk("c2", "Dado B sobre a taxa de desemprego.")
        prompt, used = pb.build("query?", [c1, c2])
        for c in used:
            assert c.text[:_MAX_CHUNK_CHARS] in prompt

    def test_prompt_nonempty_for_covered_query(self) -> None:
        pb = PromptBuilder()
        prompt, _ = pb.build("Qual o crescimento do Paraná?", [_chunk("c1")])
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# Budget de tokens
# ---------------------------------------------------------------------------

class TestBudget:
    def test_empty_chunks_returns_empty_used(self) -> None:
        pb = PromptBuilder()
        _, used = pb.build("pergunta?", [])
        assert used == []

    def test_empty_chunks_prompt_still_contains_query(self) -> None:
        # prompt sem contexto ainda deve conter a query e a instrução
        pb = PromptBuilder()
        query = "Pergunta sem contexto disponível?"
        prompt, _ = pb.build(query, [])
        assert query in prompt

    def test_chunk_dropped_when_budget_too_small(self) -> None:
        # budget forçado para valor mínimo — nenhum chunk cabe
        pb = PromptBuilder()
        pb._input_budget = 5
        _, used = pb.build("q?", [_chunk("c1", "texto qualquer")])
        assert used == []

    def test_first_chunk_preserved_within_budget(self) -> None:
        pb = PromptBuilder()
        c1 = _chunk("c1", "A" * 10)
        c2 = _chunk("c2", "B" * 10)
        _, used = pb.build("q?", [c1, c2])
        assert any(c.id == "c1" for c in used)

    def test_used_chunks_are_subset_of_input_chunks(self) -> None:
        pb = PromptBuilder()
        chunks = [_chunk(f"c{i}", f"texto relevante {i}" * 5) for i in range(5)]
        _, used = pb.build("pergunta?", chunks)
        input_ids = {c.id for c in chunks}
        assert all(c.id in input_ids for c in used)

    def test_used_texts_appear_in_prompt(self) -> None:
        pb = PromptBuilder()
        text = "Dado específico sobre o agronegócio paranaense em 2024."
        prompt, used = pb.build("agronegócio?", [_chunk("c1", text)])
        for c in used:
            assert c.text[:_MAX_CHUNK_CHARS] in prompt


# ---------------------------------------------------------------------------
# Invariante de budget
# ---------------------------------------------------------------------------

class TestBudgetInvariant:
    def test_input_budget_equals_context_window_minus_max_tokens(self) -> None:
        # invariante central: espaço reservado para geração não pode ser usado por contexto
        pb = PromptBuilder()
        assert pb._input_budget == pb._context_window - pb._max_tokens
