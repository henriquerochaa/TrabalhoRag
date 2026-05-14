# TrabalhoRag — API de Chat RAG (IPARDES)

API de perguntas e respostas sobre os 3 PDFs do IPARDES, executando **100% offline** após o setup inicial.

---

## SETUP — requer internet (executar uma vez)

Execute os comandos abaixo **em ordem** antes de usar o projeto.
Após o setup, **nenhuma conexão de rede será feita em runtime**.

```bash
# 1. Dependências Python
pip install -r requirements.txt

# 2. Modelo LLM local (llama3.2:3b — ~2 GB)
#    Ollama deve estar instalado no host: https://ollama.com
ollama pull llama3.2:3b

# 3. Modelo de linguagem spaCy para português
python -m spacy download pt_core_news_lg

# 4. Modelos de embedding e reranker (salvos em models/)
python scripts/download_models.py

# 5. PDFs do IPARDES (salvos em data/raw/)
python scripts/download_pdfs.py

# 6. Indexar os PDFs no FAISS (gera data/processed/)
python ingest.py
```

> **Torch CPU-only** (opcional, download menor ~200 MB):
> ```bash
> pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
> ```

> **Atenção — hf-xet (AVX-512):** Se o comando `python scripts/download_models.py` terminar com
> `Fatal Error: HW capability... HW capability requested: 0x200000`, remova o hf-xet:
> ```bash
> pip uninstall hf-xet -y
> ```
> O hf-xet é um acelerador de download opcional que exige AVX-512 (ausente no i5-13500).
> Sem ele o download usa HTTP padrão e funciona normalmente.

---

## RUNTIME — offline, sem internet

Com o setup concluído, **nenhum dos comandos abaixo faz chamada de rede**.
Qualquer tentativa de acesso externo em runtime causa erro imediato
(`TRANSFORMERS_OFFLINE=1` está definido em todos os módulos que carregam modelo).

---

### Cenário A — Ollama já rodando nativamente no host (caso mais comum)

Use este cenário se você instalou o Ollama diretamente no seu sistema
(serviço nativo, porta 11434 já em uso).

```bash
# Confirme que o Ollama está ativo e com o modelo disponível:
curl http://localhost:11434/api/tags

# Sobe apenas API e Streamlit (sem container Ollama):
docker compose up --build
```

| Serviço | Porta | Descrição                        |
|---------|-------|----------------------------------|
| `api`   | 8000  | FastAPI — endpoint POST /chat    |
| `app`   | 8501  | Streamlit — interface de usuário |

A API acessa o Ollama nativo via `host.docker.internal:11434`
(resolvido para o gateway do host através de `extra_hosts` no Linux).

---

### Cenário B — Sem Ollama nativo (rodando tudo pelo Docker)

Use este cenário se o Ollama **não** está instalado no host.
O perfil `full` adiciona o container Ollama ao stack.

```bash
# Sobe Ollama + API + Streamlit:
docker compose --profile full up --build
```

> O serviço `ollama-init` baixa o modelo `llama3.2:3b` automaticamente assim que o Ollama estiver saudável.
> Acompanhe o progresso com: `docker compose --profile full logs -f ollama-init`

| Serviço  | Porta | Descrição                        |
|----------|-------|----------------------------------|
| `ollama` | 11434 | LLM llama3.2:3b (container)      |
| `api`    | 8000  | FastAPI — endpoint POST /chat    |
| `app`    | 8501  | Streamlit — interface de usuário |

---

### Acessar

- **Interface:** http://localhost:8501
- **API direta:** `POST http://localhost:8000/chat` com `{"question": "..."}`
- **Docs interativos:** http://localhost:8000/docs

---

## Setup no Linux Mint (sem PowerShell)

Os passos de setup são idênticos, substituindo comandos PowerShell por bash:

```bash
# 1. Python 3.12 + venv
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Remover hf-xet antes de instalar (AVX-512 crash no i5-13500)
pip install -r requirements.txt --no-cache-dir
pip uninstall hf-xet -y

# 3-6. Restante idêntico (comandos `python` funcionam igual)
ollama pull llama3.2:3b
python -m spacy download pt_core_news_lg
python scripts/download_models.py
python scripts/download_pdfs.py
python ingest.py
```

> **`host.docker.internal` no Linux**: em distribuições Linux sem Docker Desktop,
> esse hostname pode não resolver. O `docker-compose.yml` já inclui `extra_hosts`
> para mapear `host.docker.internal → host-gateway` automaticamente.

---

## API fora do Docker (desenvolvimento local)

Se rodar a API diretamente com `uvicorn` no host (sem Docker),
`host.docker.internal` não resolve no Linux — altere temporariamente em `config.yaml`:

```yaml
llm:
  base_url: "http://localhost:11434"
```

---

## Testes

```bash
# Testes unitários (não requerem índice nem Ollama)
pytest tests/ -v --ignore=tests/test_evaluation.py

# Testes de avaliação end-to-end (requerem ingest.py já executado)
pytest tests/test_evaluation.py -v

# Avaliação completa via HTTP (requer API rodando)
python scripts/run_evaluation.py
```

---

## Estrutura do Projeto

```
scripts/
  download_pdfs.py        # URLs dos PDFs (único lugar permitido)
  download_models.py      # Download de embedding + reranker para models/
  run_evaluation.py       # Avaliação end-to-end via HTTP (11 perguntas)
src/
  config_loader.py        # Loader YAML compartilhado entre todas as camadas
  domain/                 # Entidades e ports (interfaces) — zero dependências externas
  application/            # Use cases — orquestração via ports
  infrastructure/         # Implementações concretas (PDF, embeddings, FAISS, LLM)
  interface/              # FastAPI (api.py) + Streamlit (app.py)
tests/
  test_pymupdf_extractor.py
  test_text_cleaner.py
  test_chunking.py
  test_vector_store.py
  test_search_chunks.py
  test_generate_answer.py
  test_ollama_llm.py      # Retry/backoff + fallback de indisponibilidade
  test_evaluation.py      # 11 perguntas end-to-end (A: por doc, B: out-of-scope, C: multi-doc)
data/
  raw/                    # PDFs originais (baixados no setup, incluídos no ZIP de entrega)
  processed/              # FAISS index + SQLite (gerados pelo ingest.py, incluídos no ZIP)
models/                   # Pesos embedding + reranker (baixados no setup, incluídos no ZIP)
config.yaml               # Todos os parâmetros — zero hardcode no código
docker-compose.yml        # FastAPI + Streamlit (+ Ollama com --profile full)
Dockerfile                # Imagem única para api e app
ingest.py                 # Script raiz de ingestão
requirements.txt
README.md
```

---

## ZIP de Entrega

```
entrega.zip
  scripts/
  src/
  tests/
  data/raw/               # os 3 PDFs
  data/processed/         # FAISS index + SQLite prontos
  models/                 # pesos embedding + reranker
  config.yaml
  docker-compose.yml
  Dockerfile
  requirements.txt
  README.md
  ingest.py
```

> **Excluir do ZIP**: `models/xet/` (binário hf-xet com AVX-512 que trava o i5-13500),
> `.venv/`, `__pycache__/`, `*.pyc` e `models/ollama/` (modelo LLM, ~2 GB — o professor baixa via `ollama pull`).

> **Torch no ZIP**: não inclua o cache do pip. O professor instala `torch>=2.0.0` via
> `pip install -r requirements.txt`. Para CPU puro sem download de 2 GB:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`
