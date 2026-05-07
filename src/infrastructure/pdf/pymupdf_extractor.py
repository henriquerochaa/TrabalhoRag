from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from src.domain.ports import PDFExtractorPort
from src.infrastructure.config_loader import load_config

# PyMuPDF escolhido sobre pdfplumber e pdfminer por três razões:
# 1. Velocidade: parsing em C via MuPDF — ~10x mais rápido que pdfminer puro
# 2. Fidelidade: preserva layout e ordem de leitura melhor em PDFs com colunas
# 3. Robustez: lida com PDFs mal-formados sem lançar exceção na maioria dos casos


class PyMuPDFExtractor(PDFExtractorPort):
    def __init__(self) -> None:
        self._raw_dir = Path(load_config()["paths"]["raw"])

    def extract(self, path: str) -> list[tuple[int, str]]:
        full_path = self._raw_dir / path if not Path(path).is_absolute() else Path(path)

        pages: list[tuple[int, str]] = []
        with fitz.open(str(full_path)) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages.append((page.number + 1, text))

        return pages
