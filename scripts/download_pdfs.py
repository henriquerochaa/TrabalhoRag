"""
Script de setup — executar UMA VEZ com internet antes do runtime.
URLs concentradas aqui para garantir que nenhum outro módulo faça chamadas externas.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

# URLs hardcoded aqui por design: única fonte autorizada de chamadas externas do projeto
_PDFS: list[tuple[str, str]] = [
    (
        "https://www.ipardes.pr.gov.br/sites/ipardes/arquivos_restritos/files/documento/2023-09/desenvolvimento_paranaense.pdf",
        "desenvolvimento_paranaense.pdf",
    ),
    (
        "https://www.ipardes.pr.gov.br/sites/ipardes/arquivos_restritos/files/documento/2026-02/Analise_Conjuntural_julho_agosto_2025.pdf",
        "analise_conjuntural_2025.pdf",
    ),
    (
        "https://www.ipardes.pr.gov.br/sites/ipardes/arquivos_restritos/files/documento/2025-12/Avaliacoes%20Politicas%20Publicas%20Brasil_revisao%20escopo.pdf",
        "avaliacoes_politicas_publicas.pdf",
    ),
]

_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_pdfs() -> None:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    for url, filename in _PDFS:
        dest = _RAW_DIR / filename

        if dest.exists():
            print(f"[skip] {filename} já existe")
            continue

        print(f"[download] {filename} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"OK ({size_kb} KB)")

    print("Concluído.")


if __name__ == "__main__":
    download_pdfs()
