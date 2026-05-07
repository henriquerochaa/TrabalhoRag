import pytest

from src.infrastructure.config_loader import load_config
from src.infrastructure.pdf.pymupdf_extractor import PyMuPDFExtractor


@pytest.fixture(scope="module")
def extractor() -> PyMuPDFExtractor:
    return PyMuPDFExtractor()


@pytest.fixture(scope="module")
def pdf_filenames() -> list[str]:
    return [pdf["filename"] for pdf in load_config()["pdfs"]]


@pytest.fixture(scope="module")
def all_extractions(extractor: PyMuPDFExtractor, pdf_filenames: list[str]) -> dict[str, list[tuple[int, str]]]:
    return {filename: extractor.extract(filename) for filename in pdf_filenames}


class TestExtractReturnType:
    def test_returns_list(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            assert isinstance(pages, list), f"{filename}: esperado list"

    def test_each_element_is_tuple(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            for item in pages:
                assert isinstance(item, tuple), f"{filename}: elemento não é tuple"
                assert len(item) == 2, f"{filename}: tuple deve ter 2 elementos"

    def test_first_element_is_int(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            for page_num, _ in pages:
                assert isinstance(page_num, int), f"{filename}: page_num não é int"

    def test_second_element_is_str(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            for _, text in pages:
                assert isinstance(text, str), f"{filename}: text não é str"


class TestPageNumbers:
    def test_page_numbers_start_at_one(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            assert pages[0][0] == 1, f"{filename}: primeira página deve ser 1"

    def test_page_numbers_are_positive(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            for page_num, _ in pages:
                assert page_num > 0, f"{filename}: page_num deve ser positivo"

    def test_page_numbers_are_unique(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            nums = [p for p, _ in pages]
            assert len(nums) == len(set(nums)), f"{filename}: page_nums duplicados"

    def test_page_numbers_are_sequential(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            nums = [p for p, _ in pages]
            assert nums == sorted(nums), f"{filename}: page_nums fora de ordem"


class TestTextContent:
    def test_text_is_not_empty(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            for page_num, text in pages:
                assert text.strip(), f"{filename} p.{page_num}: texto vazio"

    def test_minimum_pages_extracted(self, all_extractions: dict) -> None:
        # PDFs do IPARDES têm ao menos 10 páginas cada
        for filename, pages in all_extractions.items():
            assert len(pages) >= 10, f"{filename}: poucas páginas extraídas ({len(pages)})"

    def test_text_contains_portuguese_chars(self, all_extractions: dict) -> None:
        # presença de acentos confirma que a codificação foi preservada
        full_text = " ".join(t for pages in all_extractions.values() for _, t in pages)
        has_accent = any(c in full_text for c in "ãõéêáçúí")
        assert has_accent, "nenhum caractere acentuado encontrado — possível problema de encoding"


class TestAllThreePdfs:
    def test_all_pdfs_were_extracted(self, all_extractions: dict, pdf_filenames: list[str]) -> None:
        assert set(all_extractions.keys()) == set(pdf_filenames)

    def test_each_pdf_has_content(self, all_extractions: dict) -> None:
        for filename, pages in all_extractions.items():
            total_chars = sum(len(t) for _, t in pages)
            assert total_chars > 1000, f"{filename}: conteúdo suspeito ({total_chars} chars)"
