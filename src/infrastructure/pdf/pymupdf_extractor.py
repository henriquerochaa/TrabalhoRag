from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from src.domain.ports import PDFExtractorPort
from src.infrastructure.config_loader import load_config
from src.infrastructure.pdf.text_cleaner import (
    filter_low_density_lines,
    normalize_whitespace,
    remove_repeated_headers,
)

# PyMuPDF escolhido sobre pdfplumber e pdfminer por três razões:
# 1. Velocidade: parsing em C via MuPDF — ~10x mais rápido que pdfminer puro
# 2. Fidelidade: preserva layout e ordem de leitura melhor em PDFs com colunas
# 3. Robustez: lida com PDFs mal-formados sem lançar exceção na maioria dos casos


class PyMuPDFExtractor(PDFExtractorPort):
    def __init__(self) -> None:
        cfg = load_config()
        self._raw_dir = Path(cfg["paths"]["raw"])
        chunking_cfg = cfg["chunking"]
        self._header_threshold: float = chunking_cfg["header_frequency_threshold"]
        self._min_line_chars: int = chunking_cfg["min_line_chars"]

    def extract(self, path: str) -> list[tuple[int, str]]:
        full_path = self._raw_dir / path if not Path(path).is_absolute() else Path(path)

        raw_pages: list[tuple[int, str]] = []
        with fitz.open(str(full_path)) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    raw_pages.append((page.number + 1, text))

        # remove_repeated_headers precisa do corpus completo para calcular frequência —
        # por isso a limpeza ocorre aqui, após coletar todas as páginas, e não por página
        pages = remove_repeated_headers(raw_pages, threshold_ratio=self._header_threshold)
        result: list[tuple[int, str]] = []
        for num, text in pages:
            normalized = normalize_whitespace(text)
            if not normalized.strip():
                continue
            filtered = filter_low_density_lines(normalized, min_chars=self._min_line_chars)
            # página pode ficar vazia se só continha linhas curtas (ex: índice, numeração)
            if filtered.strip():
                result.append((num, filtered))
        return result
