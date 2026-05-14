# AFAZER — Tarefas Pendentes do Projeto RAG IPARDES

> **Status atual (13/05/2026)**: pipeline completo, 11/11 testes de avaliação passando,
> 125 testes unitários coletados. Ambiente de desenvolvimento: Windows 11 + RTX 3050 (GPU isolada).
> Ambiente alvo do professor: Linux Mint + i5-13500 + 16 GB RAM, sem GPU dedicada.

---

## 🔴 CRÍTICO — Antes da Entrega

### 1. Atingir 130+ testes unitários
- **Status**: 125 testes coletados (faltam 5+)
- **Meta do enunciado**: 130+ testes, cobertura >80%
- **O que adicionar**:
  - [ ] `tests/test_embedder.py` — testa `SentenceTransformerEmbedder` com mock do modelo (evita carregar 1 GB em teste unitário)
  - [ ] `tests/test_prompt_builder.py` — se existir um `PromptBuilder` separado, testar edge cases (chunk vazio, pergunta sem contexto)
  - [ ] `tests/test_reranker.py` — testa `CrossEncoderReranker` com mock do cross-encoder
  - [ ] `tests/test_mmr.py` — testa algoritmo MMR isolado (diversidade de chunks)
  - [ ] `tests/test_ollama_llm.py` — adicionar casos: timeout, resposta malformada, JSON inválido
- **Verificar cobertura**:
  ```bash
  pytest --cov=src tests/ --cov-report=term-missing
  # Meta: >80% de cobertura em src/
  ```

### 2. Validar pipeline completo no Linux Mint (máquina do professor)
- **Status**: testado apenas no Windows com GPU (GPU isolada via CUDA_VISIBLE_DEVICES)
- [ ] Rodar `pytest tests/ -v` na máquina Linux Mint 16GB
- [ ] Rodar `python scripts/run_evaluation.py` (11 perguntas via HTTP com API no ar)
- [ ] Confirmar tempo de resposta ~8-12s por pergunta em CPU puro
- [ ] Verificar que `faiss-cpu==1.9.0` instala sem AVX-512 crash no Linux
- [ ] Verificar que `hf-xet` não está instalado (ou removê-lo: `pip uninstall hf-xet -y`)

### 3. Testar interface Streamlit e FastAPI end-to-end
- **Status**: API testada via curl/pytest, Streamlit não validada visualmente
- [ ] Subir `uvicorn src.interface.api.main:app --reload --port 8000`
- [ ] Acessar `http://localhost:8000/docs` e testar POST /chat manualmente
- [ ] Subir `streamlit run src/interface/streamlit_app.py`
- [ ] Fazer pergunta em `http://localhost:8501` e verificar resposta + `prompt_used` + `sources`
- [ ] Verificar que output duplo (`answer` + `prompt_used`) aparece na UI

### 4. Testar Docker Compose
- **Status**: `docker-compose.yml` existe mas não foi testado nesta sessão
- [ ] `docker compose up --build` (Cenário A: Ollama nativo no host)
- [ ] `docker compose --profile full up --build` (Cenário B: tudo em container)
- [ ] Verificar healthcheck do container Ollama (Cenário B)
- [ ] Confirmar que `host.docker.internal:11434` resolve no Linux

---

## 🟡 IMPORTANTE — Qualidade e Nota

### 5. Verificar cobertura de testes >80%
```bash
pip install pytest-cov
pytest --cov=src tests/ --cov-report=term-missing --cov-fail-under=80
```
- Se abaixo de 80%: identificar módulos descobertos e adicionar testes

### 6. Rodar `scripts/run_evaluation.py` com API no ar
- **Status**: `test_evaluation.py` (pytest, sem HTTP) passou 11/11
- `run_evaluation.py` faz requisições HTTP reais para `localhost:8000/chat`
- [ ] Subir API + Ollama
- [ ] `python scripts/run_evaluation.py`
- [ ] Confirmar Summary: 11/11 passed

### 7. Revisar output duplo em todas as respostas
- **Requisito do enunciado**: cada resposta deve retornar `prompt_used` E `answer` separados
- [ ] Verificar que `Answer.prompt_used` está populado em todos os casos (in-scope e out-of-scope)
- [ ] Verificar que a API retorna ambos campos no JSON de resposta
- [ ] Verificar que a Streamlit exibe ambos (ex.: em expander "Prompt usado")

### 8. Verificar que `ingest.py` está na raiz (não em scripts/)
- **Status do README**: `python ingest.py` na raiz — confirmar que está correto
- [ ] Testar `python ingest.py` do zero em ambiente limpo (sem `data/processed/`)
- [ ] Confirmar que gera `data/processed/index.faiss`, `id_map.json`, `metadata.db`

---

## 🟢 ENTREGA — Empacotamento

### 9. Preparar ZIP de entrega
- [ ] Confirmar que `data/raw/` tem os 3 PDFs do IPARDES
- [ ] Confirmar que `data/processed/` tem `index.faiss`, `id_map.json`, `metadata.db`
- [ ] Confirmar que `models/` tem pesos do embedder e reranker (+ `processor_config.json` vazio)
- [ ] Confirmar que `requirements.txt` está em UTF-8 (não UTF-16 LE)
- [ ] Confirmar `faiss-cpu==1.9.0` no requirements.txt
- [ ] **Excluir do ZIP**: `.venv/`, `__pycache__/`, `.git/`, `.claude/`, arquivos `.pyc`
- [ ] Criar ZIP:
  ```powershell
  # Windows:
  Compress-Archive -Path scripts,src,tests,data,models,config.yaml,docker-compose.yml,Dockerfile,requirements.txt,CLAUDE.md,README.md,ingest.py -DestinationPath entrega.zip
  ```

### 10. Checklist final pré-entrega
- [ ] `pytest tests/ -v` → 130+ testes, todos passando
- [ ] `pytest --cov=src tests/` → cobertura >80%
- [ ] `python scripts/run_evaluation.py` → 11/11 passed
- [ ] `config.yaml`: `model: "llama3.2:3b"`, `timeout_seconds: 180`, `max_tokens: 150`
- [ ] Nenhuma URL de PDF fora de `scripts/download_pdfs.py`
- [ ] `TRANSFORMERS_OFFLINE=1` em todo arquivo de infrastructure que carrega modelo
- [ ] Todos os membros conseguem explicar cada decisão técnica

---

## 📝 OPCIONAL — Melhorias de Qualidade

### 11. Benchmark de chunk_size
- `scripts/benchmark_chunk_size.py` (se existir) — verificar que 512 é melhor que 256/1024
- Documentar resultado no CLAUDE.md

### 12. Adicionar setup Linux ao README
- Seção "Setup no Linux Mint" com comandos bash equivalentes aos PowerShell
- Relevante para: máquina do professor e ambientes Linux

### 13. Adicionar teste de performance
- Verificar que `FAISS busca < 200ms`, `Ollama < 180s`
- Pode ser teste de integração marcado com `@pytest.mark.slow`

---

## Problemas Resolvidos (Histórico)

| Problema | Solução | Commit |
|---|---|---|
| `faiss-cpu==1.7.4` removido do PyPI | Atualizar para `1.9.0` | 62e725a |
| `hf-xet` crash AVX-512 | `pip uninstall hf-xet -y` | — (documentado) |
| Ollama HTTP 500 (cudaMalloc OOM) | `CUDA_VISIBLE_DEVICES=""` + `device="cpu"` | cfe8487 |
| Ollama HTTP 500 (num_ctx 2048) | `"num_ctx": 8192` no payload | 07b3579 |
| `processor_config.json` offline | Criar `{}` vazio em models/ | — (manual) |
| `requirements.txt` UTF-16 LE | Reescrever em UTF-8 | 34288e7 |
| sentence-transformers 5.4.1 → 5.5.0 | Atualizar requirements.txt | 62e725a |
