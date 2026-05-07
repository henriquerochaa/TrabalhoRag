import hashlib
import re

import pytest

from src.domain.entities import Chunk, Document
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker
from src.infrastructure.config_loader import load_config
from src.infrastructure.pdf.pymupdf_extractor import PyMuPDFExtractor
from src.infrastructure.pdf.text_cleaner import (
    filter_low_density_lines,
    normalize_whitespace,
    remove_repeated_headers,
)


@pytest.fixture(scope="module")
def chunker() -> RecursiveChunker:
    return RecursiveChunker()


@pytest.fixture(scope="module")
def doc() -> Document:
    return Document(id="testdoc", filename="test.pdf")


@pytest.fixture(scope="module")
def chunk_size() -> int:
    return load_config()["chunking"]["chunk_size"]


@pytest.fixture(scope="module")
def overlap() -> int:
    return load_config()["chunking"]["overlap"]


def _long_paragraph(n_sentences: int = 30) -> str:
    sentence = "O desenvolvimento econômico do Paraná avançou de forma consistente nos últimos anos. "
    return sentence * n_sentences


def _pipe_table(n_rows: int = 4) -> str:
    rows = ["| Indicador        | 2021  | 2022  | 2023  |"]
    for i in range(n_rows):
        rows.append(f"| Item {i:03d}         | {100+i}   | {110+i}   | {120+i}   |")
    return "\n".join(rows)


def _space_table(n_rows: int = 4) -> str:
    rows = ["Indicador          2021    2022    2023"]
    for i in range(n_rows):
        rows.append(f"Item {i:03d}            {100+i}     {110+i}     {120+i}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# chunk_size
# ---------------------------------------------------------------------------

class TestChunkSize:
    def test_no_chunk_exceeds_chunk_size(self, chunker: RecursiveChunker, doc: Document, chunk_size: int) -> None:
        pages = [(1, _long_paragraph(40))]
        chunks = chunker.chunk(pages, doc)
        for c in chunks:
            assert len(c.text) <= chunk_size, f"chunk excede chunk_size: len={len(c.text)}"

    def test_short_paragraph_kept_as_single_chunk(self, chunker: RecursiveChunker, doc: Document) -> None:
        text = "Parágrafo curto que não precisa ser dividido."
        chunks = chunker.chunk([(1, text)], doc)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_generates_multiple_chunks(self, chunker: RecursiveChunker, doc: Document) -> None:
        pages = [(1, _long_paragraph(40))]
        chunks = chunker.chunk(pages, doc)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# overlap
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_consecutive_chunks_share_overlap_text(
        self, chunker: RecursiveChunker, doc: Document, overlap: int
    ) -> None:
        # forçar divisão por sentença — parágrafo único muito longo
        pages = [(1, _long_paragraph(40))]
        chunks = chunker.chunk(pages, doc)
        assert len(chunks) >= 2

        for i in range(len(chunks) - 1):
            suffix = chunks[i].text[-overlap:].strip()
            if suffix:
                assert suffix in chunks[i + 1].text, (
                    f"overlap ausente entre chunk {i} e {i+1}: "
                    f"sufixo {suffix!r} não encontrado em {chunks[i+1].text[:80]!r}"
                )

    def test_fixed_split_overlap_is_exact(
        self, chunker: RecursiveChunker, doc: Document, chunk_size: int, overlap: int
    ) -> None:
        # sentença maior que chunk_size → cai no _split_fixed
        single_sentence = "x" * (chunk_size * 3)
        pages = [(1, single_sentence)]
        chunks = chunker.chunk(pages, doc)
        assert len(chunks) >= 2

        for i in range(len(chunks) - 1):
            expected_overlap = chunks[i].text[-(overlap):]
            assert chunks[i + 1].text.startswith(expected_overlap), (
                f"overlap fixo incorreto entre chunk {i} e {i+1}"
            )


# ---------------------------------------------------------------------------
# tabelas
# ---------------------------------------------------------------------------

class TestTablePreservation:
    def test_small_pipe_table_is_single_chunk(self, chunker: RecursiveChunker, doc: Document) -> None:
        pages = [(1, _pipe_table(4))]
        chunks = chunker.chunk(pages, doc)
        table_chunks = [c for c in chunks if "|" in c.text]
        assert len(table_chunks) == 1

    def test_small_space_table_is_single_chunk(self, chunker: RecursiveChunker, doc: Document) -> None:
        pages = [(1, _space_table(4))]
        chunks = chunker.chunk(pages, doc)
        assert len(chunks) == 1

    def test_large_pipe_table_rows_never_split(
        self, chunker: RecursiveChunker, doc: Document
    ) -> None:
        pages = [(1, _pipe_table(60))]
        chunks = chunker.chunk(pages, doc)
        assert len(chunks) > 1
        for c in chunks:
            for line in c.text.splitlines():
                stripped = line.strip()
                if stripped.startswith("|"):
                    assert stripped.endswith("|"), f"linha de tabela incompleta: {stripped!r}"

    def test_all_table_rows_preserved_after_split(
        self, chunker: RecursiveChunker, doc: Document
    ) -> None:
        original = _pipe_table(60)
        original_rows = [l.strip() for l in original.splitlines() if l.strip()]
        pages = [(1, original)]
        chunks = chunker.chunk(pages, doc)
        recovered_rows = [
            line.strip()
            for c in chunks
            for line in c.text.splitlines()
            if line.strip()
        ]
        assert recovered_rows == original_rows

    def test_table_mixed_with_text_does_not_corrupt_text(
        self, chunker: RecursiveChunker, doc: Document
    ) -> None:
        normal = "Este parágrafo de texto corrido precisa ser mantido intacto após o chunking."
        pages = [(1, _pipe_table(4) + "\n\n" + normal)]
        chunks = chunker.chunk(pages, doc)
        text_chunks = [c for c in chunks if "parágrafo" in c.text]
        assert len(text_chunks) == 1
        assert text_chunks[0].text == normal


# ---------------------------------------------------------------------------
# metadados
# ---------------------------------------------------------------------------

class TestChunkMetadata:
    def _all_chunks(self, chunker: RecursiveChunker, doc: Document) -> list[Chunk]:
        pages = [
            (1, "SEÇÃO UM\n\nPrimeiro parágrafo da seção um com conteúdo relevante."),
            (2, _long_paragraph(20)),
            (3, _pipe_table(4)),
        ]
        return chunker.chunk(pages, doc)

    def test_document_id_always_filled(self, chunker: RecursiveChunker, doc: Document) -> None:
        for c in self._all_chunks(chunker, doc):
            assert c.document_id == doc.id

    def test_page_always_positive_int(self, chunker: RecursiveChunker, doc: Document) -> None:
        for c in self._all_chunks(chunker, doc):
            assert isinstance(c.page, int) and c.page > 0

    def test_position_unique_and_sequential(self, chunker: RecursiveChunker, doc: Document) -> None:
        chunks = self._all_chunks(chunker, doc)
        positions = [c.position for c in chunks]
        assert positions == list(range(len(positions)))

    def test_id_format(self, chunker: RecursiveChunker, doc: Document) -> None:
        for c in self._all_chunks(chunker, doc):
            parts = c.id.split("_")
            assert parts[0] == doc.id, f"id mal formado: {c.id}"

    def test_section_is_str_never_none(self, chunker: RecursiveChunker, doc: Document) -> None:
        for c in self._all_chunks(chunker, doc):
            assert isinstance(c.section, str), f"section não é str: {c.section!r}"
            assert c.section is not None

    def test_section_propagates_within_page(self, chunker: RecursiveChunker, doc: Document) -> None:
        pages = [(1, "TÍTULO DA SEÇÃO\n\n" + _long_paragraph(20))]
        chunks = chunker.chunk(pages, doc)
        for c in chunks:
            assert c.section == "TÍTULO DA SEÇÃO"

    def test_section_empty_string_when_not_detected(
        self, chunker: RecursiveChunker, doc: Document
    ) -> None:
        # parágrafo longo sem título detectável — section pode ser vazia, não None
        pages = [(1, _long_paragraph(3))]
        chunks = chunker.chunk(pages, doc)
        for c in chunks:
            assert c.section is not None
            assert isinstance(c.section, str)


# ---------------------------------------------------------------------------
# métricas de qualidade sobre PDFs reais
# ---------------------------------------------------------------------------

# Métricas definidas pela equipe com base em análise manual dos 3 PDFs do IPARDES:
#
# 1. section_coverage >= 80%
#    Avaliamos manualmente ~50 chunks de cada PDF e observamos que títulos
#    detectáveis (CAIXA ALTA, numeração, title case curto) cobrem a maioria das
#    seções. 80% é o piso aceitável considerando páginas de sumário, índice
#    e tabelas que legitimamente não têm seção acima delas.
#
# 2. artifact_free >= 95%
#    Artefatos de extração PDF (caracteres de substituição Unicode �,
#    bytes nulos, sequências de 4+ caracteres não-alfanuméricos consecutivos)
#    indicam falha no extrator ou encoding corrompido. 95% foi escolhido como
#    piso porque PDFs governamentais podem conter fórmulas matemáticas ou
#    caracteres especiais legítimos que acionam o padrão de artefato.


@pytest.fixture(scope="module")
def real_chunks(chunker: RecursiveChunker) -> list[Chunk]:
    """Gera chunks dos 3 PDFs reais aplicando o pipeline completo de limpeza."""
    cfg = load_config()
    extractor = PyMuPDFExtractor()
    all_chunks: list[Chunk] = []

    for pdf in cfg["pdfs"]:
        filename: str = pdf["filename"]
        doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
        document = Document(id=doc_id, filename=filename)

        pages = extractor.extract(filename)
        pages = remove_repeated_headers(pages)
        pages = [(n, normalize_whitespace(t)) for n, t in pages]
        pages = [(n, filter_low_density_lines(t)) for n, t in pages]
        pages = [(n, t) for n, t in pages if t.strip()]

        all_chunks.extend(chunker.chunk(pages, document))

    return all_chunks


def _has_artifact(text: str) -> bool:
    # � = caractere de substituição Unicode — indica encoding corrompido
    if "�" in text or "\x00" in text:
        return True
    # 4+ caracteres não-alfanuméricos consecutivos excluindo espaço, pontuação
    # comum e separadores de tabela (|, -, +) que são estrutura legítima
    if re.search(r"[^\w\s|,.:;!?()\-\'\"\n]{4,}", text):
        return True
    return False


class TestQualityMetrics:
    def test_section_coverage_above_threshold(self, real_chunks: list[Chunk]) -> None:
        # >= 80% dos chunks devem ter section não vazia
        threshold = 0.80
        with_section = sum(1 for c in real_chunks if c.section.strip())
        coverage = with_section / len(real_chunks)
        assert coverage >= threshold, (
            f"section_coverage={coverage:.1%} abaixo do mínimo {threshold:.0%} "
            f"({with_section}/{len(real_chunks)} chunks com seção)"
        )

    def test_artifact_free_above_threshold(self, real_chunks: list[Chunk]) -> None:
        # >= 95% dos chunks devem estar livres de artefatos de extração
        threshold = 0.95
        clean = sum(1 for c in real_chunks if not _has_artifact(c.text))
        ratio = clean / len(real_chunks)
        assert ratio >= threshold, (
            f"artifact_free={ratio:.1%} abaixo do mínimo {threshold:.0%} "
            f"({len(real_chunks) - clean} chunks com artefatos de {len(real_chunks)} total)"
        )

    def test_total_chunks_plausible(self, real_chunks: list[Chunk]) -> None:
        # 3 PDFs com centenas de páginas devem gerar ao menos 300 chunks
        assert len(real_chunks) >= 300, f"apenas {len(real_chunks)} chunks gerados — pipeline pode estar com falha"

    def test_no_empty_chunk_text(self, real_chunks: list[Chunk]) -> None:
        empty = [c for c in real_chunks if not c.text.strip()]
        assert not empty, f"{len(empty)} chunks com texto vazio nos PDFs reais"
