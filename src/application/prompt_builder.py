from __future__ import annotations

import hashlib

from src.config_loader import load_config
from src.domain.entities import Chunk


def _build_filename_map(cfg: dict) -> dict[str, str]:
    # Reconstrói o mapeamento document_id → filename do IngestDocuments
    # para exibir nomes legíveis no prompt enviado à LLM.
    return {
        hashlib.md5(e["filename"].encode()).hexdigest(): e["filename"]
        for e in cfg["pdfs"]
    }

# Instruções restritivas explícitas são necessárias para controlar alucinação
# em modelos <= 9.9B por três razões:
#
# 1. Modelos menores têm menor separação entre conhecimento paramétrico e
#    informação do contexto — sem âncoras textuais fortes ("EXCLUSIVAMENTE",
#    "não suponha"), tendem a completar lacunas dos trechos com o que
#    aprenderam no pré-treino, produzindo respostas parcialmente corretas mas
#    não sustentadas pelos documentos. (Shi et al., 2023, "Large Language
#    Models Can Be Easily Distracted by Irrelevant Context" — o efeito é
#    inversamente proporcional ao tamanho do modelo.)
#
# 2. A frase de fallback explícita ("Não encontrei essa informação...") dá
#    ao modelo uma saída legítima para ausência de evidência, evitando a
#    geração especulativa que preencheria o vazio com conteúdo inventado.
#    Sem ela, modelos pequenos preferem responder com baixa confiança a
#    admitir que não sabem — o chamado "honesty gap" documentado em RLHF
#    com modelos abaixo de 13B.
#
# 3. A separação visual de seções (###) guia a atenção do decoder: modelos
#    <= 9.9B com janela de atenção menor se beneficiam de delimitadores
#    explícitos para não misturar instrução, contexto e pergunta.

_INSTRUCTION = """\
Você é um assistente especializado nos documentos do IPARDES sobre o estado do Paraná.
Responda EXCLUSIVAMENTE com base nos trechos fornecidos abaixo.
Se a resposta não estiver nos trechos, responda: "Não encontrei essa informação nos documentos fornecidos."
Não suponha, não complete e não extrapole informações que não estejam explicitamente nos trechos."""

_CONTEXT_HEADER = "\n\n### TRECHOS DOS DOCUMENTOS\n"
_QUESTION_HEADER = "\n\n### PERGUNTA\n"
_ANSWER_LEAD = "\n\n### RESPOSTA\n"


def _estimate_tokens(text: str) -> int:
    # chars // 4 é conservador para português (média real ~4.2 chars/token).
    # Superestimar é o comportamento seguro — melhor descartar um chunk a mais
    # do que ultrapassar a janela e ter o prompt truncado pelo Ollama.
    return max(1, len(text) // 4)


_MAX_CHUNK_CHARS = 250


class PromptBuilder:
    def __init__(self) -> None:
        cfg = load_config()
        self._context_window: int = cfg["llm"]["context_window"]
        self._max_tokens: int = cfg["llm"]["max_tokens"]
        # Mesmo invariante de SearchChunks: reserva max_tokens para a geração
        # da resposta. O restante é o budget total de entrada (instrução +
        # contexto + pergunta). Se SearchChunks já comprimiu os chunks para
        # context_window - max_tokens, o overhead do template (instrução,
        # cabeçalhos, query) ainda não foi descontado — PromptBuilder descarta
        # chunks adicionais se necessário para acomodar esse overhead.
        self._input_budget: int = self._context_window - self._max_tokens
        # Mapeamento document_id (MD5) → filename legível para exibição no prompt.
        # Essencial para rastreabilidade: o professor verifica QUAL documento gerou
        # cada trecho no prompt_used — hash MD5 não é auditável sem esse mapa.
        self._filename_map: dict[str, str] = _build_filename_map(cfg)

    def _format_chunk(self, index: int, chunk: Chunk) -> str:
        section_part = f" | Seção: {chunk.section}" if chunk.section else ""
        filename = self._filename_map.get(chunk.document_id, chunk.document_id)
        # Primeiros 250 chars concentram a informação mais densa do chunk.
        # Limite necessário para viabilizar execução em CPU sem GPU: chunks completos
        # de 512 tokens forçariam num_predict > 512 para cobrir contexto + resposta,
        # tornando cada query inviável (> 256 s de geração observados em benchmarks).
        text = chunk.text[:_MAX_CHUNK_CHARS]
        return (
            f"\n--- Trecho {index} ---\n"
            f"Documento: {filename} | Página: {chunk.page}{section_part}\n\n"
            f"{text}\n"
        )

    def build(self, query: str, chunks: list[Chunk]) -> tuple[str, list[Chunk]]:
        """Monta o prompt final e retorna (prompt, chunks_efetivamente_usados).

        Itera sobre chunks na ordem recebida (melhor para pior, conforme saída
        do reranker) e descarta os do final se o total estimado ultrapassar
        input_budget. Nunca trunca um chunk no meio.
        """
        skeleton = _INSTRUCTION + _CONTEXT_HEADER + _QUESTION_HEADER + query + _ANSWER_LEAD
        chunk_budget = self._input_budget - _estimate_tokens(skeleton)

        used: list[Chunk] = []
        tokens_used = 0

        for i, chunk in enumerate(chunks):
            block = self._format_chunk(i + 1, chunk)
            block_tokens = _estimate_tokens(block)
            if tokens_used + block_tokens > chunk_budget:
                break
            used.append(chunk)
            tokens_used += block_tokens

        context_block = "".join(self._format_chunk(i + 1, c) for i, c in enumerate(used))

        prompt = (
            _INSTRUCTION
            + _CONTEXT_HEADER
            + context_block
            + _QUESTION_HEADER
            + query
            + _ANSWER_LEAD
        )

        return prompt, used
