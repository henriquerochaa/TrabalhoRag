"""
Testes de avaliação end-to-end do pipeline RAG sobre os 3 PDFs do IPARDES.

Pré-requisito: python ingest.py deve ter sido executado antes deste módulo.
Os testes são pulados automaticamente se o índice FAISS não for encontrado.

Cobertura:
  Categoria A — por documento: verifica que a fonte retornada pertence ao PDF esperado
  Categoria B — fora do escopo: verifica anti-alucinação sem chamar a LLM
  Categoria C — multi-documento: verifica que a resposta cobre mais de um PDF
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.application.generate_answer import GenerateAnswer, _OUT_OF_SCOPE_TEXT
from src.application.prompt_builder import PromptBuilder
from src.application.search_chunks import SearchChunks
from src.config_loader import load_config
from src.domain.entities import Chunk
from src.domain.ports import LLMPort
from src.infrastructure.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.infrastructure.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.storage.faiss_vector_store import FAISSVectorStore
from src.infrastructure.storage.sqlite_metadata_store import SQLiteMetadataStore


# ---------------------------------------------------------------------------
# Constantes — document_ids derivados dos filenames (mesma lógica do IngestDocuments)
# ---------------------------------------------------------------------------

def _doc_id(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()


_DOC_DESENVOLVIMENTO = _doc_id("desenvolvimento_paranaense.pdf")
_DOC_CONJUNTURAL     = _doc_id("analise_conjuntural_2025.pdf")
_DOC_POLITICAS       = _doc_id("avaliacoes_politicas_publicas.pdf")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def search_uc() -> SearchChunks:
    """Carrega índice FAISS + SQLite reais; pula o módulo se não existirem."""
    cfg = load_config()
    processed = cfg["paths"]["processed"]

    if not (Path(processed) / "index.faiss").exists():
        pytest.skip(
            "índice FAISS não encontrado — execute `python ingest.py` antes de rodar este teste"
        )

    meta = SQLiteMetadataStore()
    meta.load(processed)
    vs = FAISSVectorStore(meta)
    vs.load(processed)
    # modelo carregado uma única vez para o módulo inteiro
    return SearchChunks(SentenceTransformerEmbedder(), vs, CrossEncoderReranker())


@pytest.fixture(scope="module")
def answer_uc(search_uc: SearchChunks) -> GenerateAnswer:
    """GenerateAnswer com LLM que falha se chamada — garante que Categoria B
    não aciona inferência."""
    class _NoCallLLM(LLMPort):
        def generate(self, prompt: str) -> str:
            raise AssertionError(
                "LLM não deve ser chamada para queries fora do escopo — "
                "equivale a alucinação sem contexto"
            )

    return GenerateAnswer(search_uc, PromptBuilder(), _NoCallLLM())


# ---------------------------------------------------------------------------
# Categoria A — por documento
# Cada pergunta usa terminologia exclusiva de um único PDF para garantir
# que as fontes retornadas pertençam ao documento esperado.
# ---------------------------------------------------------------------------

class TestCategoriaA:

    def test_credito_pessoas_fisicas_paraná(self, search_uc: SearchChunks) -> None:
        # "22,27% em termos reais" é uma estatística única na analise_conjuntural_2025
        # p.4 — sem correspondência nos outros dois PDFs, tornando o documento alvo
        # inequívoco para qualquer embedder com competência mínima em português
        query = "Qual foi a variação percentual da carteira de crédito a pessoas físicas no Paraná entre junho de 2024 e junho de 2025?"
        chunks, out_of_scope = search_uc.execute(query)

        assert not out_of_scope, "pergunta coberta pela conjuntural não pode ser out_of_scope"
        assert chunks, "deve retornar ao menos um chunk"
        doc_ids = {c.document_id for c in chunks}
        assert _DOC_CONJUNTURAL in doc_ids, (
            f"esperado chunk de analise_conjuntural_2025; recebido doc_ids={doc_ids}"
        )

    def test_taxa_desocupacao_paraná(self, search_uc: SearchChunks) -> None:
        # "segundo trimestre de 2023" como marco da desocupação abaixo de 5% é
        # dado específico da conjuntural p.5; não aparece nos outros PDFs
        query = "Desde qual trimestre o Paraná registra taxa de desocupação inferior a 5%?"
        chunks, out_of_scope = search_uc.execute(query)

        assert not out_of_scope
        assert chunks
        assert _DOC_CONJUNTURAL in {c.document_id for c in chunks}, (
            "dado de desocupação está na analise_conjuntural_2025"
        )

    def test_decomposicao_produtividade_trabalho(self, search_uc: SearchChunks) -> None:
        # A decomposição da variação de produtividade em três componentes
        # (realocação, ganho intrassetorial, termo cruzado) está nas páginas 18-21
        # do desenvolvimento_paranaense — metodologia ausente nos outros dois PDFs
        query = "Quais são os componentes da decomposição da variação de produtividade do trabalho analisados no estudo sobre o Paraná?"
        chunks, out_of_scope = search_uc.execute(query)

        assert not out_of_scope
        assert chunks
        assert _DOC_DESENVOLVIMENTO in {c.document_id for c in chunks}, (
            "decomposição de produtividade está em desenvolvimento_paranaense"
        )

    def test_protocolo_prisma_revisao_escopo(self, search_uc: SearchChunks) -> None:
        # "PRISMA ScR" (variante para scoping reviews) é mencionado exclusivamente
        # em avaliacoes_politicas_publicas p.7; o acrônimo não ocorre nos outros PDFs
        query = "Qual protocolo foi adotado para garantir a transparência e reprodutibilidade da revisão de escopo de políticas públicas brasileiras?"
        chunks, out_of_scope = search_uc.execute(query)

        assert not out_of_scope
        assert chunks
        assert _DOC_POLITICAS in {c.document_id for c in chunks}, (
            "PRISMA ScR está em avaliacoes_politicas_publicas"
        )


# ---------------------------------------------------------------------------
# Categoria B — fora do escopo
# Perguntas sobre temas completamente ausentes dos 3 PDFs.
# Assert central: out_of_scope=True sem chamar a LLM (anti-alucinação garantida
# pela ausência de chamada ao modelo — verificada pelo _NoCallLLM no answer_uc).
# ---------------------------------------------------------------------------

class TestCategoriaB:

    def test_geografia_internacional_fora_do_escopo(self, answer_uc: GenerateAnswer) -> None:
        # distância entre cidades estrangeiras é geograficamente incompatível
        # com o corpus IPARDES, que trata exclusivamente do Paraná e do Brasil
        answer = answer_uc.execute("Qual é a distância em quilômetros entre São Paulo e Buenos Aires?")

        assert answer.out_of_scope is True
        assert answer.text == _OUT_OF_SCOPE_TEXT
        assert answer.sources == []
        assert answer.prompt_used == ""

    def test_tecnologia_blockchain_fora_do_escopo(self, answer_uc: GenerateAnswer) -> None:
        # blockchain e Proof of Work não têm nenhuma ocorrência nos PDFs —
        # qualquer resposta seria alucinação pura do modelo
        answer = answer_uc.execute("Como funciona o protocolo de consenso Proof of Work utilizado no Bitcoin?")

        assert answer.out_of_scope is True
        assert answer.text == _OUT_OF_SCOPE_TEXT
        assert answer.sources == []
        assert answer.prompt_used == ""

    def test_esporte_fora_do_escopo(self, answer_uc: GenerateAnswer) -> None:
        # regras de xadrez não têm relação com desenvolvimento econômico,
        # políticas públicas ou análise conjuntural do Paraná
        answer = answer_uc.execute("Quais são as regras do campeonato mundial de xadrez segundo a FIDE?")

        assert answer.out_of_scope is True
        assert answer.text == _OUT_OF_SCOPE_TEXT
        assert answer.sources == []
        assert answer.prompt_used == ""

    def test_quimica_fora_do_escopo(self, answer_uc: GenerateAnswer) -> None:
        # tabela periódica e estrutura atômica são domínio de ciências exatas,
        # completamente fora do escopo socioeconômico dos documentos IPARDES
        answer = answer_uc.execute("Qual é a estrutura eletrônica do átomo de carbono segundo a tabela periódica?")

        assert answer.out_of_scope is True
        assert answer.text == _OUT_OF_SCOPE_TEXT
        assert answer.sources == []
        assert answer.prompt_used == ""


# ---------------------------------------------------------------------------
# Categoria C — multi-documento
# Perguntas cujos conceitos-chave estão distribuídos em PDFs distintos.
# Assert: sources com document_ids de ao menos 2 documentos diferentes;
# out_of_scope=False confirma que há contexto suficiente para não alucinar.
# ---------------------------------------------------------------------------

class TestCategoriaC:

    def test_mercado_trabalho_e_produtividade(self, search_uc: SearchChunks) -> None:
        # "mercado de trabalho" (desocupação, emprego) está na analise_conjuntural;
        # "produtividade do trabalho" (decomposição, setores) está em desenvolvimento_paranaense.
        # Query semântica abrange os dois campos, forçando busca multi-documento.
        query = "Qual é a situação do mercado de trabalho paranaense e como ele se relaciona com a produtividade do trabalho no estado?"
        chunks, out_of_scope = search_uc.execute(query)

        assert not out_of_scope, (
            "pergunta sobre emprego e produtividade tem cobertura em ao menos um PDF"
        )
        assert chunks

        doc_ids = {c.document_id for c in chunks}
        assert len(doc_ids) >= 2, (
            f"esperado chunks de ao menos 2 documentos; recebido apenas {doc_ids}. "
            "Verifique se a ingestão incluiu todos os 3 PDFs e se o reranker preserva diversidade."
        )

    def test_exportacoes_agricolas_e_economia(self, search_uc: SearchChunks) -> None:
        # Tabela de exportações (frango, suíno) está em analise_conjuntural p.22;
        # importância econômica do agronegócio paranaense está em desenvolvimento_paranaense.
        # Pergunta exige dados de ambos para resposta completa.
        query = "Quais são as principais exportações agrícolas do Paraná e qual é a importância do agronegócio para a economia estadual?"
        chunks, out_of_scope = search_uc.execute(query)

        assert not out_of_scope, (
            "agronegócio paranaense é coberto tanto na conjuntural quanto em desenvolvimento"
        )
        assert chunks

        doc_ids = {c.document_id for c in chunks}
        assert len(doc_ids) >= 2, (
            f"esperado chunks de ao menos 2 documentos; recebido {doc_ids}"
        )

    def test_avaliacao_politicas_e_desenvolvimento(self, search_uc: SearchChunks) -> None:
        # "avaliação de políticas públicas" está exclusivamente em avaliacoes_politicas_publicas;
        # "desenvolvimento econômico paranaense" está exclusivamente em desenvolvimento_paranaense.
        # Multi-documento por construção: não é possível responder usando apenas um PDF.
        query = "Quais metodologias são usadas para avaliar políticas públicas e como essas avaliações contribuem para o desenvolvimento econômico do Paraná?"
        chunks, out_of_scope = search_uc.execute(query)

        assert not out_of_scope, (
            "avaliação de políticas e desenvolvimento têm cobertura nos PDFs do IPARDES"
        )
        assert chunks

        doc_ids = {c.document_id for c in chunks}
        assert len(doc_ids) >= 2, (
            f"esperado chunks de ao menos 2 documentos; recebido {doc_ids}"
        )
