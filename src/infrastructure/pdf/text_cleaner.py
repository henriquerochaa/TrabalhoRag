from __future__ import annotations

import re
from collections import Counter


def remove_repeated_headers(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    # linhas que aparecem em mais de 30% das páginas são cabeçalhos/rodapés —
    # threshold empírico: baixo demais remove conteúdo legítimo repetido,
    # alto demais deixa ruído de paginação nos chunks
    if not pages:
        return pages

    threshold = max(2, int(len(pages) * 0.30))
    all_lines = [line.strip() for _, text in pages for line in text.splitlines()]
    counts = Counter(all_lines)
    repeated = {line for line, n in counts.items() if n >= threshold and line}

    cleaned = []
    for page_num, text in pages:
        filtered = "\n".join(
            line for line in text.splitlines() if line.strip() not in repeated
        )
        cleaned.append((page_num, filtered))

    return cleaned


def normalize_whitespace(text: str) -> str:
    # hífens de quebra de linha são artefatos do PDF e interrompem tokens —
    # reconectá-los antes de colapsar espaços preserva palavras compostas reais
    text = re.sub(r"-\n(\w)", r"\1", text)
    # múltiplos espaços/tabs colapsados em espaço único; mais de 2 quebras
    # consecutivas reduzidas a parágrafo duplo para manter separação semântica
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def filter_low_density_lines(text: str, min_chars: int = 20) -> str:
    # linhas curtas em PDFs governamentais são quase sempre numeração de página,
    # títulos de seção isolados ou artefatos de tabela — descartá-las reduz ruído
    # sem perder conteúdo semântico relevante para RAG
    lines = [line for line in text.splitlines() if len(line.strip()) >= min_chars]
    return "\n".join(lines)
