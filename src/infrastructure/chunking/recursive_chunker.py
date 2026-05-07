from __future__ import annotations

import re

# spaCy carregado via pacote instalado em setup — spacy.load() não faz chamada
# de rede; o modelo pt_core_news_lg é um pacote Python local após `spacy download`
import spacy
from spacy.language import Language

from src.domain.entities import Chunk, Document
from src.domain.ports import ChunkerPort
from src.infrastructure.config_loader import load_config

# estratégia recursiva escolhida para respeitar fronteiras semânticas naturais:
# parágrafo > sentença > fixo — cada nível só é acionado quando o anterior
# gera um bloco maior que chunk_size, evitando cortes arbitrários no meio de ideias.
# chunk_size=512 tokens (~400 palavras) cabe na janela de contexto do gemma2:9b
# mantendo coerência semântica; overlap=64 preserva contexto entre chunks adjacentes
# para perguntas cujas respostas cruzam fronteiras de parágrafo.


class RecursiveChunker(ChunkerPort):
    def __init__(self) -> None:
        cfg = load_config()["chunking"]
        self._chunk_size: int = cfg["chunk_size"]
        self._overlap: int = cfg["overlap"]
        # disable=["ner"] reduz uso de memória — só precisamos do segmentador de sentenças
        self._nlp: Language = spacy.load("pt_core_news_lg", disable=["ner"])

    def chunk(self, pages: list[tuple[int, str]], document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        position = 0
        current_section = ""

        for page_num, text in pages:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            for para in paragraphs:
                detected = self._detect_section_title(para)
                if detected:
                    current_section = detected

                for fragment in self._split_paragraph(para):
                    chunks.append(Chunk(
                        id=f"{document.id}_{page_num}_{position}",
                        document_id=document.id,
                        text=fragment,
                        page=page_num,
                        section=current_section,
                        position=position,
                    ))
                    position += 1

        return chunks

    def _detect_section_title(self, paragraph: str) -> str:
        # heurística em três camadas para PDFs governamentais do IPARDES:
        # 1. CAIXA ALTA: títulos de capítulo/seção são frequentemente escritos em
        #    maiúsculas nesses documentos — sinal mais forte, peso máximo
        # 2. Linha curta + title case: subtítulos raramente passam de 80 chars
        #    e usam capitalização de título (ex: "Desenvolvimento Econômico")
        # 3. Numeração de seção (ex: "1.2 Contexto Regional"): padrão comum em
        #    documentos técnicos do IPARDES — detectado via regex
        # Não usamos metadados de negrito do PDF aqui pois o extrator entrega
        # texto plano; a detecção de spans bold exigiria acesso ao fitz.Document
        # original, quebrando a separação entre extração e chunking.
        lines = [l.strip() for l in paragraph.splitlines() if l.strip()]
        if not lines:
            return ""

        first = lines[0]

        if len(first) > 120:
            return ""

        # CAIXA ALTA — sinal mais forte
        if first.isupper() and 3 < len(first) < 120:
            return first

        # numeração de seção (1., 1.2, 1.2.3 seguidos de texto)
        if re.match(r"^\d+(\.\d+)*\.?\s+\S", first) and len(first) < 100:
            return first

        # título curto em title case sem pontuação final de sentença
        if (first.istitle() or first[0].isupper()) and len(first) < 80 and not first.endswith((".", ",", ";")):
            # parágrafo de uma única linha curta é quase sempre um título
            if len(lines) == 1:
                return first

        return ""

    def _is_table_block(self, text: str) -> bool:
        # critério duplo para cobrir os dois formatos predominantes nos PDFs do IPARDES:
        # 1. pipe-delimited (|col1|col2|): formato explícito, detectável com count("|")>=2
        # 2. space-aligned: PyMuPDF extrai tabelas sem pipes, mas preserva alinhamento
        #    de colunas via múltiplos espaços consecutivos — exigimos >=2 grupos de 2+
        #    espaços por linha para evitar falsos positivos em texto com recuo simples.
        # threshold de 60% de linhas tabulares aceita tabelas com linha de cabeçalho
        # ou separador que não segue o padrão (ex: "---+---+---").
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) < 3:
            return False

        def is_table_line(line: str) -> bool:
            if line.count("|") >= 2:
                return True
            return len(re.findall(r"  +", line)) >= 2

        table_line_count = sum(1 for l in lines if is_table_line(l))
        return table_line_count / len(lines) >= 0.6

    def _split_table(self, text: str) -> list[str]:
        # quebra apenas entre linhas completas (rows) — nunca no meio de uma linha —
        # para preservar a legibilidade e a semântica da célula para o LLM
        rows = [l for l in text.splitlines() if l.strip()]
        result: list[str] = []
        current_rows: list[str] = []
        current_len = 0

        for row in rows:
            row_len = len(row) + 1
            if current_len + row_len > self._chunk_size and current_rows:
                result.append("\n".join(current_rows))
                current_rows = []
                current_len = 0
            current_rows.append(row)
            current_len += row_len

        if current_rows:
            result.append("\n".join(current_rows))

        return result

    def _split_paragraph(self, text: str) -> list[str]:
        if self._is_table_block(text):
            if len(text) <= self._chunk_size:
                return [text]
            return self._split_table(text)
        if len(text) <= self._chunk_size:
            return [text]
        return self._split_by_sentences(text)

    def _split_by_sentences(self, text: str) -> list[str]:
        doc = self._nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        result: list[str] = []
        current = ""

        for sent in sentences:
            if len(sent) > self._chunk_size:
                if current:
                    result.append(current)
                    current = ""
                result.extend(self._split_fixed(sent))
                continue

            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # overlap: prefixo do chunk anterior para manter contexto na fronteira
                prefix = current[-self._overlap:] if current else ""
                current = (prefix + " " + sent).strip() if prefix else sent

        if current:
            result.append(current)

        return result

    def _split_fixed(self, text: str) -> list[str]:
        result: list[str] = []
        start = 0
        while start < len(text):
            result.append(text[start: start + self._chunk_size])
            start += self._chunk_size - self._overlap
        return result
