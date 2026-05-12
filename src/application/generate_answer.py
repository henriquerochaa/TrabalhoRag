from __future__ import annotations

from src.application.prompt_builder import PromptBuilder
from src.application.search_chunks import SearchChunks
from src.domain.entities import Answer
from src.domain.exceptions import OllamaUnavailableError
from src.domain.ports import LLMPort

_OUT_OF_SCOPE_TEXT = "O assunto não está coberto pelos documentos disponíveis."
_UNAVAILABLE_TEXT = "Serviço de geração indisponível no momento. Tente novamente."


class GenerateAnswer:
    """Orquestra o pipeline completo: busca → prompt → geração.

    Recebe SearchChunks e PromptBuilder como dependências de aplicação e LLMPort
    como port de domínio, seguindo Clean Architecture: nenhuma importação de
    infraestrutura neste módulo.
    """

    def __init__(
        self,
        search: SearchChunks,
        prompt_builder: PromptBuilder,
        llm: LLMPort,
    ) -> None:
        self._search = search
        self._prompt_builder = prompt_builder
        self._llm = llm

    def execute(self, query: str) -> Answer:
        chunks, out_of_scope = self._search.execute(query)

        if out_of_scope:
            # Retorno antecipado sem chamar a LLM: evita gerar resposta inventada
            # quando nenhum chunk ultrapassou min_score_threshold. prompt_used=""
            # sinaliza explicitamente que a LLM não foi consultada.
            return Answer(
                text=_OUT_OF_SCOPE_TEXT,
                sources=[],
                prompt_used="",
                out_of_scope=True,
            )

        prompt, used_chunks = self._prompt_builder.build(query, chunks)
        # used_chunks pode ser subconjunto de chunks: PromptBuilder descarta os
        # de menor score se o overhead do template (instrução + cabeçalhos + query)
        # reduzir o budget disponível para o contexto.

        try:
            text = self._llm.generate(prompt)
        except OllamaUnavailableError:
            # sources e prompt_used preenchidos porque o pipeline chegou até a LLM:
            # o usuário sabe quais trechos seriam usados e pode retentar mais tarde
            return Answer(
                text=_UNAVAILABLE_TEXT,
                sources=used_chunks,
                prompt_used=prompt,
                out_of_scope=False,
            )

        return Answer(
            text=text,
            sources=used_chunks,
            prompt_used=prompt,
            out_of_scope=False,
        )
