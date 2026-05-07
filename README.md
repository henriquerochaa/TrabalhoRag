# TrabalhoRag — API de Chat RAG (IPARDES)

API de perguntas e respostas sobre os 3 PDFs do IPARDES, executando **100% offline** após o setup inicial.

---

## SETUP — requer internet (executar uma vez)

Execute os comandos abaixo **em ordem** antes de usar o projeto.
Após o setup, nenhuma conexão de rede será necessária.

```bash
# 1. Dependências Python
pip install -r requirements.txt

# 2. Modelo LLM local
ollama pull gemma2:9b

# 3. Modelos de embedding e reranker (salvos em models/)
python scripts/download_models.py

# 4. Modelo de linguagem spaCy para português
python -m spacy download pt_core_news_lg

# 5. PDFs do IPARDES (salvos em data/raw/)
python scripts/download_pdfs.py
```

---

## RUNTIME — offline, sem internet

Com o setup concluído, **nenhum comando abaixo faz chamada de rede**.
Qualquer tentativa de acesso externo em runtime causará erro imediato.

```bash
# Sobe o Ollama local (porta 11434)
docker-compose up

# Processa os PDFs e indexa no FAISS + SQLite
python ingest.py
```

---

## Estrutura do Projeto

```
src/
  domain/           # Entidades e ports (interfaces) — zero dependências externas
  application/      # Use cases — orquestração via ports
  infrastructure/   # Implementações concretas (PDF, embeddings, FAISS, LLM)
  interface/        # FastAPI + Streamlit
scripts/            # Setup: download_pdfs.py, download_models.py
tests/              # Testes unitários
data/
  raw/              # PDFs originais (baixados no setup)
  processed/        # FAISS index + SQLite (gerados pelo ingest.py)
models/             # Pesos dos modelos embedding + reranker (baixados no setup)
config.yaml         # Todos os parâmetros do sistema
docker-compose.yml  # Serviço Ollama
```
