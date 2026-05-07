# CLAUDE.md — RAG Project

## Contexto do Projeto
API de chat baseada em RAG, executando 100% local após setup inicial.
Documentos-fonte: 3 PDFs oficiais do governo do Paraná (IPARDES).
LLM com no máximo 9.9 bilhões de parâmetros.

---

## Regras Críticas do Enunciado

> Violar qualquer uma dessas regras compromete a nota diretamente.

- **Execução 100% offline** — zero chamadas de rede durante runtime. Isso inclui:
  download de modelos, download de PDFs, qualquer chamada HTTP fora do Ollama local.
- **LLM máximo 9.9B parâmetros** — qualquer modelo deve respeitar esse limite.
- **Apenas os 3 PDFs do IPARDES** — proibido usar outras bases de dados.
- **Modelos pré-baixados no setup** — `sentence-transformers`, `cross-encoder` e `spaCy`
  fazem download automático se não encontrados. Isso é PROIBIDO em runtime.
  Use `TRANSFORMERS_OFFLINE=1` em todos os arquivos de infraestrutura que carregam modelos.
- **URLs dos PDFs** — pertencem APENAS a `scripts/download_pdfs.py`. Nunca em `config.yaml`.
- **Compartilhamento entre equipes** — arquivos, parâmetros e prompts são proibidos.
  Verbal é permitido. Violação = nota ZERO para todos os envolvidos.
- **Prova de autoria** — todos os membros devem saber explicar cada escolha do código.
- **Output duplo obrigatório** — por pergunta, retornar separado:
  1. `prompt_used` — prompt completo enviado à LLM
  2. `answer` — resposta final

---

## Fases de Execução

### Setup (com internet — executar uma vez)
```bash
pip install -r requirements.txt
ollama pull gemma2:9b
python scripts/download_models.py   # embedding + reranker → models/
python -m spacy download pt_core_news_lg
python scripts/download_pdfs.py     # PDFs → data/raw/
```

### Runtime (offline — sem internet)
```bash
docker-compose up
python ingest.py
```

---

## Critérios de Avaliação

| Critério | Peso |
|---|---|
| Organização e clareza (comentários com JUSTIFICATIVA das escolhas) | 20% |
| Testes unitários (cobertura + métricas de qualidade da equipe) | 20% |
| Qualidade do tratamento de dados (preprocessamento, chunking, filtragem) | 30% |
| Qualidade do RAG (escopo, anti-alucinação, multi-documento) | 30% |

---

## Arquitetura — Clean Architecture

Regra absoluta: camada interna NUNCA importa de camada externa.

```
src/
  domain/           # Entidades puras + ports — zero dependências externas
  application/      # Use cases — depende apenas de ports
  infrastructure/   # Implementações concretas
  interface/        # FastAPI + Streamlit
scripts/            # Setup only — download_pdfs.py, download_models.py
tests/
data/
  raw/              # PDFs — baixados no setup, não commitados
  processed/        # FAISS + SQLite — incluídos no ZIP de entrega
models/             # Pesos dos modelos — baixados no setup, montados como volume
```

---

## Regras de Código

- **Zero hardcode** — todos os parâmetros em `config.yaml`
- **URLs dos PDFs** — apenas em `scripts/download_pdfs.py`
- **`TRANSFORMERS_OFFLINE=1`** — obrigatório em todo arquivo que carrega modelo HuggingFace
- **Type hints obrigatórios** em todas as funções e métodos
- **Ports são ABCs** — use cases dependem de ports, nunca de implementações
- **Cada módulo novo** tem teste unitário correspondente
- **Comentários justificam decisões**, não descrevem o código

```python
# ERRADO
# carrega o modelo de embeddings

# CORRETO
# multilingual-e5-large escolhido por suporte nativo a português sem fine-tuning
# alternativa BGE-M3 tem recall superior mas exige mais VRAM
```

---

## Enforcement de Offline nos Modelos HuggingFace

Todo arquivo de infraestrutura que carrega modelo deve ter no topo:

```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = config["paths"]["models"]
os.environ["TRANSFORMERS_CACHE"] = config["paths"]["models"]
```

Isso garante que qualquer tentativa de download em runtime gera erro imediato
em vez de fazer chamada de rede silenciosa.

---

## config.yaml — sem URLs, sem paths absolutos

```yaml
pdfs:
  - filename: "desenvolvimento_paranaense.pdf"
  - filename: "analise_conjuntural_2025.pdf"
  - filename: "avaliacoes_politicas_publicas.pdf"

paths:
  raw: "data/raw/"
  processed: "data/processed/"
  models: "models/"

chunking:
  chunk_size: 512
  overlap: 64

retrieval:
  top_k_initial: 20
  top_k_final: 5
  min_score_threshold: 0.65

embedding:
  model_name: "intfloat/multilingual-e5-large"

reranker:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"

llm:
  model: "gemma2:9b"
  base_url: "http://localhost:11434"
  temperature: 0.1
  max_tokens: 1024
  context_window: 8192
```

---

## Entidades de Domínio

```python
@dataclass
class Document:
    id: str        # md5 do filename
    filename: str

@dataclass
class Chunk:
    id: str        # f"{document_id}_{page}_{position}"
    document_id: str
    text: str
    page: int
    section: str
    position: int

@dataclass
class SearchResult:
    chunk: Chunk
    score: float

@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    prompt_used: str      # obrigatório — exigido pelo professor
    out_of_scope: bool = False
```

---

## ZIP de Entrega

```
entrega.zip
  scripts/            # download_pdfs.py, download_models.py
  src/                # código-fonte completo
  tests/              # testes unitários
  data/raw/           # os 3 PDFs
  data/processed/     # FAISS index + SQLite prontos
  models/             # pesos dos modelos embedding + reranker
  config.yaml
  docker-compose.yml
  requirements.txt
  README.md           # separar claramente: setup (internet) vs runtime (offline)
```

---

## Como Usar no Claude Code

Um prompt por vez. Commit após cada tarefa concluída e testada.

```bash
claude "Implemente src/infrastructure/embeddings/sentence_transformer_embedder.py
implementando EmbedderPort. Carregar modelo do path em config['paths']['models'].
Definir TRANSFORMERS_OFFLINE=1 no topo do arquivo.
Seguir todas as regras do CLAUDE.md."
```
