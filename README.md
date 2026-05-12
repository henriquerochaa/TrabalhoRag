# TrabalhoRag — API de Chat RAG (IPARDES)

API de perguntas e respostas sobre os 3 PDFs do IPARDES, executando **100% offline** após o setup inicial.

---

## SETUP — requer internet (executar uma vez)

Execute os comandos abaixo **em ordem** antes de usar o projeto.
Após o setup, **nenhuma conexão de rede será feita em runtime**.

```bash
# 1. Dependências Python
pip install -r requirements.txt

# 2. Modelo LLM local (gemma2:9b — ~5 GB)
#    Ollama deve estar instalado no host: https://ollama.com
ollama pull gemma2:9b

# 3. Modelos de embedding e reranker (salvos em models/)
python scripts/download_models.py

# 4. Modelo de linguagem spaCy para português
python -m spacy download pt_core_news_lg

# 5. PDFs do IPARDES (salvos em data/raw/)
python scripts/download_pdfs.py
```

> **Torch CPU-only** (opcional, download menor):
> ```bash
> pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
> ```

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

> **Atenção:** na primeira execução o container Ollama precisa baixar o modelo gemma2:9b (~5 GB).
> Aguarde o healthcheck do serviço `ollama` passar antes de fazer requisições.
> Acompanhe com: `docker compose logs -f ollama`

| Serviço  | Porta | Descrição                        |
|----------|-------|----------------------------------|
| `ollama` | 11434 | LLM gemma2:9b (container)        |
| `api`    | 8000  | FastAPI — endpoint POST /chat    |
| `app`    | 8501  | Streamlit — interface de usuário |

---

### Indexar os PDFs (primeira execução ou após limpar processed/)

```bash
python ingest.py
```

Idempotente — documentos já indexados são ignorados.

### Acessar

- **Interface:** http://localhost:8501
- **API direta:** `POST http://localhost:8000/chat` com `{"question": "..."}`
- **Docs interativos:** http://localhost:8000/docs

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
