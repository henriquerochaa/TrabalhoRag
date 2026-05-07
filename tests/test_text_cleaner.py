import pytest

from src.infrastructure.pdf.text_cleaner import (
    filter_low_density_lines,
    normalize_whitespace,
    remove_repeated_headers,
)


# ---------------------------------------------------------------------------
# remove_repeated_headers
# ---------------------------------------------------------------------------

class TestRemoveRepeatedHeaders:
    def _make_pages(self, n: int, header: str, body_prefix: str = "Conteúdo único da página") -> list[tuple[int, str]]:
        return [(i, f"{header}\n{body_prefix} {i}\nTexto relevante aqui.") for i in range(1, n + 1)]

    def test_removes_line_present_in_majority_of_pages(self) -> None:
        pages = self._make_pages(10, "Governo do Paraná — IPARDES")
        result = remove_repeated_headers(pages)
        for _, text in result:
            assert "Governo do Paraná — IPARDES" not in text

    def test_preserves_unique_content(self) -> None:
        pages = self._make_pages(10, "Cabeçalho Repetido")
        result = remove_repeated_headers(pages)
        for i, (_, text) in enumerate(result, start=1):
            assert f"Conteúdo único da página {i}" in text

    def test_does_not_remove_line_below_threshold(self) -> None:
        # linha que aparece em apenas 2 de 20 páginas não deve ser removida
        pages = [(i, f"Texto normal da página {i}.") for i in range(1, 21)]
        pages[0] = (1, "Linha rara\nTexto normal da página 1.")
        pages[1] = (2, "Linha rara\nTexto normal da página 2.")
        result = remove_repeated_headers(pages)
        assert "Linha rara" in result[0][1]

    def test_empty_input_returns_empty(self) -> None:
        assert remove_repeated_headers([]) == []

    def test_single_page_unchanged(self) -> None:
        pages = [(1, "Único conteúdo desta página sem repetição.")]
        result = remove_repeated_headers(pages)
        assert result == pages

    def test_no_repeated_headers_content_intact(self) -> None:
        pages = [(i, f"Parágrafo completamente distinto número {i}.") for i in range(1, 8)]
        result = remove_repeated_headers(pages)
        for original, cleaned in zip(pages, result):
            assert original[1] == cleaned[1]

    def test_preserves_page_numbers(self) -> None:
        pages = self._make_pages(5, "Cabeçalho")
        result = remove_repeated_headers(pages)
        assert [p for p, _ in result] == [1, 2, 3, 4, 5]

    def test_page_with_only_header_becomes_empty_string(self) -> None:
        pages = [(i, "Cabeçalho Repetido") for i in range(1, 11)]
        result = remove_repeated_headers(pages)
        for _, text in result:
            assert text.strip() == ""


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self) -> None:
        assert normalize_whitespace("texto   com   muitos   espaços") == "texto com muitos espaços"

    def test_collapses_tabs(self) -> None:
        assert normalize_whitespace("coluna1\t\tcoluna2") == "coluna1 coluna2"

    def test_reconnects_hyphenated_line_break(self) -> None:
        result = normalize_whitespace("desen-\nvolvimento")
        assert "desenvolvimento" in result

    def test_does_not_reconnect_valid_hyphen(self) -> None:
        # hífen no meio da linha não deve ser alterado
        result = normalize_whitespace("bem-estar da população")
        assert "bem-estar" in result

    def test_reduces_excess_newlines_to_double(self) -> None:
        result = normalize_whitespace("parágrafo 1\n\n\n\nparágrafo 2")
        assert "\n\n\n" not in result
        assert "parágrafo 1" in result
        assert "parágrafo 2" in result

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert normalize_whitespace("   texto   ") == "texto"

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_whitespace("") == ""

    def test_only_whitespace_returns_empty(self) -> None:
        assert normalize_whitespace("   \t\t\n\n   ") == ""

    def test_preserves_semantic_content(self) -> None:
        original = "O PIB do Paraná cresceu   3,2%  em  2023."
        result = normalize_whitespace(original)
        assert "PIB" in result
        assert "Paraná" in result
        assert "3,2%" in result


# ---------------------------------------------------------------------------
# filter_low_density_lines
# ---------------------------------------------------------------------------

class TestFilterLowDensityLines:
    def test_removes_short_lines(self) -> None:
        text = "42\nEste parágrafo tem conteúdo semântico suficiente para ser mantido."
        result = filter_low_density_lines(text)
        assert "42" not in result
        assert "semântico" in result

    def test_removes_page_number_artifacts(self) -> None:
        text = "Página 12\nO desenvolvimento paranaense avançou significativamente no período."
        result = filter_low_density_lines(text)
        assert "Página 12" not in result
        assert "desenvolvimento paranaense" in result

    def test_preserves_lines_at_exactly_min_chars(self) -> None:
        line = "a" * 20
        result = filter_low_density_lines(line, min_chars=20)
        assert line in result

    def test_removes_lines_below_min_chars(self) -> None:
        line = "a" * 19
        result = filter_low_density_lines(line, min_chars=20)
        assert line not in result

    def test_empty_string_returns_empty(self) -> None:
        assert filter_low_density_lines("") == ""

    def test_all_short_lines_returns_empty(self) -> None:
        text = "ok\nid\n42\ncap"
        result = filter_low_density_lines(text)
        assert result.strip() == ""

    def test_all_long_lines_returns_all(self) -> None:
        lines = [
            "O Paraná registrou crescimento econômico acima da média nacional.",
            "A taxa de desemprego recuou para 6,4% no segundo trimestre de 2024.",
        ]
        result = filter_low_density_lines("\n".join(lines))
        for line in lines:
            assert line in result

    def test_custom_min_chars_respected(self) -> None:
        text = "Texto curto.\nEste texto é um pouco mais longo que o anterior."
        result = filter_low_density_lines(text, min_chars=30)
        assert "Texto curto." not in result
        assert "Este texto" in result

    def test_whitespace_only_line_removed(self) -> None:
        text = "   \nEsta linha tem conteúdo relevante para o teste de densidade."
        result = filter_low_density_lines(text)
        lines = [l for l in result.splitlines() if l.strip() == ""]
        assert not lines
