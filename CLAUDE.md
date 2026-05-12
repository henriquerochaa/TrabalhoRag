# CLAUDE.md — RAG IPARDES: Baseline CPU Puro (i5-13500) + llama3.2:3b

---

## ⚠️ Regras Críticas do Enunciado

**Violar qualquer uma compromete a nota diretamente.**

### 1. Execução 100% Offline
- ✅ Zero chamadas de rede durante runtime (após setup)
- ✅ Download de modelos, PDFs, datasets: **apenas em setup** (scripts/)
- ✅ Runtime: apenas HTTP para `localhost:11434` (Ollama local)
- ✅ `TRANSFORMERS_OFFLINE=1` em todo arquivo que carrega modelo HuggingFace

### 2. LLM Máximo 9.9B Parâmetros
- ✅ `llama3.2:3b` — 3B parâmetros (desenvolvimento em CPU)
- ✅ Nunca usar `gemma2:9b` ou qualquer modelo > 9.9B
- ⚠️ Se precisar modelo melhor: `mistral-7b` (7B) é alternativa válida

### 3. Apenas os 3 PDFs do IPARDES
- ✅ Fonte única: `scripts/download_pdfs.py`
- ✅ Proibido usar outras bases de dados, Wikipedia, etc.
- ✅ PDFs salvos em `data/raw/` após setup

### 4. Modelos Pré-Baixados no Setup
- ✅ `sentence-transformers` (embedding)
- ✅ `cross-encoder` (reranking)
- ✅ `spaCy pt_core_news_lg`
- ✅ Tudo baixado em setup, **nunca em runtime**
- ✅ Diretório: `models/` (incluído no ZIP)

### 5. URLs dos PDFs
- ✅ **APENAS** em `scripts/download_pdfs.py`
- ❌ Nunca em `config.yaml`, comentários, ou outros arquivos
- ❌ Nunca commitar URLs em público (GitHub)

### 6. Compartilhamento Entre Equipes
- ❌ Código, parâmetros, prompts: proibido compartilhar (nota ZERO)
- ✅ Discussão verbal: permitido
- ✅ Cada membro deve saber explicar todas as escolhas

### 7. Prova de Autoria
- ✅ Todos os membros devem conseguir explicar cada decisão
- ✅ Código bem comentado com JUSTIFICATIVAS
- ❌ Aceitar código sem entender: nota ZERO

### 8. Output Duplo Obrigatório
- ✅ Por pergunta, retornar **separado**:
  1. `prompt_used` — prompt completo enviado à LLM
  2. `answer` — resposta final
- ✅ Permite auditoria e rastreabilidade completa

---

## Hardware de Referência (Professor/Avaliador)

```
CPU: i5-13500 (20 CPUs) — CPU puro, SEM GPU
RAM: 16GB
GPU: Intel UHD Graphics 770 (integrada, não suporta CUDA)
OS: Windows 11 Pro 64-bit
```

**Filosofia do Projeto**: Todo código é desenvolvido e testado **assumindo que não há GPU**.

- ✅ Funciona em CPU puro (~8-12s por resposta com `llama3.2:3b`)
- ✅ Seu ambiente pessoal com GPU acelera apenas no desenvolvimento local
- ⚠️ Nunca assuma CUDA, cuDNN, ou GPU na entrega

---

## Arquitetura — Clean Architecture

**Regra absoluta**: Camada interna NUNCA importa de camada externa.

```
src/
  domain/           # Entidades puras + ports (ABCs)
                    # Zero dependências externas
  application/      # Use cases (GenerateAnswer, SearchChunks, etc)
                    # Depende apenas de ports
  infrastructure/   # Implementações concretas (FAISS, Ollama, SQLite)
                    # TRANSFORMERS_OFFLINE=1 obrigatório aqui
  interface/        # FastAPI + Streamlit (entrada do usuário)

scripts/            # Setup only (download_pdfs.py, download_models.py)
                    # Nunca importado por outro código
tests/              # Testes unitários (pytest)
data/
  raw/              # PDFs — baixados no setup, não commitados
  processed/        # FAISS + SQLite — incluídos no ZIP
models/             # Pesos — embedding + reranker (setup)
```

### Ports (Interfaces Abstratas)

Todos em `src/domain/ports/`:

```python
# exemplo
from abc import ABC, abstractmethod

class EmbedderPort(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Retorna embeddings normalizados"""
        pass

class VectorStorePort(ABC):
    @abstractmethod
    def search(self, query_vector: list[float], k: int) -> list[SearchResult]:
        pass

class LLMPort(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
```

### Implementações (Infrastructure)

```python
# src/infrastructure/embedder/sentence_transformer_embedder.py
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # OBRIGATÓRIO no topo

from src.domain.ports import EmbedderPort

class SentenceTransformerEmbedder(EmbedderPort):
    def __init__(self, model_path: str):
        # carrega do models/ local, nunca HF Hub
        self.model = load_model(model_path)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        # implementação
        pass
```

---



### 1.1 Sistema Operacional
- **OS**: Windows 11 Pro 64-bit
- **Shell**: PowerShell (não cmd.exe)

### 1.2 Python e Virtual Environment

```powershell
# Verificar Python 3.12
py --list-paths

# Criar venv
py -3.12 -m venv .venv

# Ativar
.venv\Scripts\Activate.ps1

# Se der erro de script:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 1.3 Dependências

```powershell
# Instalar com --no-cache-dir (economiza RAM durante download)
pip install -r requirements.txt --no-cache-dir
```

**⚠️ CRÍTICO para CPU puro**: Garantir versão correta do faiss-cpu

```powershell
# Verificar versão instalada
pip show faiss-cpu

# Deve estar: Version: 1.7.4
# Se tiver 1.13.2 ou outra:
pip uninstall faiss-cpu -y
pip install faiss-cpu==1.7.4
```

**Por quê?** `faiss-cpu 1.13.2` exige Intel AVX-512 (i5-13500 não tem). A v1.7.4 funciona com SSE4/AVX.

---

## Critérios de Avaliação (Pesos)

| Critério | Peso | Baseline |
|---|---|---|
| **Organização e clareza** | 20% | Comentários com JUSTIFICATIVA, não descrição |
| **Testes unitários** | 20% | 130+ testes, cobertura >80% |
| **Qualidade do tratamento de dados** | 30% | Chunking hierárquico, limpeza, normalização |
| **Qualidade do RAG** | 30% | Out-of-scope detection, anti-alucinação, multi-documento |

---

## 2. Modelos e Dependências

### 2.1 Ollama (LLM Local)

```powershell
# Download e instalação: https://ollama.ai/download
# Instalar normalmente via wizard

# Em um terminal PowerShell (manter aberto):
ollama serve

# Em outro terminal, baixar modelo:
ollama pull llama3.2:3b

# Aguardar download completo (~2-3 GB)

# Validar:
curl http://localhost:11434/api/tags
# Deve retornar JSON com llama3.2:3b listado
```

### 2.2 spaCy (Processamento PT-BR)

```powershell
python -m spacy download pt_core_news_lg
```

### 2.3 Modelos de Embedding e Reranking

```powershell
# Download automático de:
# - intfloat/multilingual-e5-large (embeddings)
# - cross-encoder/ms-marco-MiniLM-L-6-v2 (reranking)
python scripts/download_models.py
```

### 2.4 PDFs do IPARDES

```powershell
python scripts/download_pdfs.py
# Baixa 3 PDFs em data/raw/
```

---

## 3. Configuração: config.yaml

### ⚠️ Regras Críticas
- ❌ **Nunca URLs dos PDFs** — apenas em `scripts/download_pdfs.py`
- ❌ **Nunca paths absolutos** — usar caminhos relativos (data/, models/)
- ✅ **Todos os parâmetros aqui** — zero hardcode em código
- ✅ **Type hints** em todas as funções

### Padrão (llama3.2:3b em CPU)

```yaml
llm:
  model: "llama3.2:3b"
  base_url: "http://localhost:11434"  # Ollama nativo (não Docker)
  temperature: 0                       # Determinístico
  max_tokens: 150                      # Resposta curta (pré-fill rápido)
  context_window: 8192
  timeout_seconds: 180                 # 3 min para CPU puro
  max_retries: 3

retrieval:
  top_k_initial: 20                    # Busca ampla em FAISS
  top_k_final: 3                       # Após MMR + reranking
  min_score_threshold: 0.65            # Anti-noise (out_of_scope)

chunking:
  chunk_size: 512                      # ~2000 tokens
  overlap: 64                          # Contexto entre chunks

prompt:
  max_chunk_chars: 250                 # Limita tokens no prompt
```

**Justificativas**:
- `timeout_seconds: 180` — CPU pura precisa de tempo. Com retry pode chegar a 30-40s, margem segura de 3min
- `max_tokens: 150` — Resposta curta = menos cálculo no decode
- `top_k_final: 3` — Menos chunks = prompt menor = decode mais rápido
- `base_url: http://localhost:11434` — Ollama nativo, não Docker

---

## Regras de Código — Enforcement

### 1. Zero Hardcode
- ✅ Todos os parâmetros em `config.yaml`
- ❌ Nunca constantes magic no código

### 2. Type Hints Obrigatórios
```python
# OBRIGATÓRIO em todos os métodos/funções
def search(self, query: str, top_k: int) -> list[SearchResult]:
    pass

def embed(self, texts: list[str]) -> np.ndarray:
    pass
```

### 3. TRANSFORMERS_OFFLINE=1
```python
# No topo de TODO arquivo que carrega modelo HuggingFace
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = config["paths"]["models"]
os.environ["TRANSFORMERS_CACHE"] = config["paths"]["models"]
```

### 4. Comentários com JUSTIFICATIVA (não descrição)
```python
# ❌ ERRADO
# carrega o modelo de embeddings

# ✅ CORRETO
# multilingual-e5-large: suporte nativo a português sem fine-tuning
# alternativa BGE-M3 tem recall superior mas exige mais VRAM
# escolhido: trade-off entre qualidade e memória para CPU de 16GB
```

### 5. Ports são ABCs (Abstract Base Classes)
```python
# src/domain/ports/embedder_port.py
from abc import ABC, abstractmethod

class EmbedderPort(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Implementações concretas em infrastructure/"""
        pass
```

### 6. Use Cases Dependem de Ports, Nunca de Implementações
```python
# ✅ CORRETO
class GenerateAnswer:
    def __init__(self, 
                 search: SearchPort,           # port (abstração)
                 llm: LLMPort,                 # port (abstração)
                 prompt_builder: PromptBuilder):
        self.search = search
        self.llm = llm
```

### 7. Cada Módulo Novo = Teste Unitário
- ✅ `src/application/use_cases/generate_answer.py` → `tests/test_generate_answer.py`
- ✅ `src/infrastructure/embedder/sentence_transformer_embedder.py` → `tests/test_embedder.py`
- ❌ Código sem teste: nota reduzida

### 8. Output Duplo Obrigatório
```python
@dataclass
class Answer:
    text: str                 # resposta final
    prompt_used: str          # OBRIGATÓRIO — prompt enviado à LLM
    sources: list[Chunk]      # documentos citados
    out_of_scope: bool = False

# retornar SEMPRE:
return Answer(
    text="resposta aqui",
    prompt_used="[prompt completo enviado]",
    sources=[chunk1, chunk2],
    out_of_scope=False
)
```

---



### 4.1 Por que Ollama Nativo (não Docker)?

- **Docker**: Precisaria de `host.docker.internal:11434` (Windows-specific, complexo)
- **Nativo**: Simples, `localhost:11434`, direto no host

### 4.2 Iniciar Ollama

```powershell
# Terminal 1 (manter aberto durante toda a sessão):
ollama serve

# Saída esperada:
# Listening on 127.0.0.1:11434
# Listening on [::]:11434
```

### 4.3 Verificar Conexão

```powershell
# Terminal 2:
curl http://localhost:11434/api/tags

# Retorna JSON:
# {"models": [{"name": "llama3.2:3b", ...}]}
```

### 4.4 Offline Garantido

- `TRANSFORMERS_OFFLINE=1` já está em `src/infrastructure/*.py`
- Nenhuma chamada HTTP para HuggingFace Hub, OpenAI, ou externos
- Única rede: `localhost:11434` (Ollama local)

---

## 5. Ingestão (Uma Vez)

```powershell
# Venv ativado, Ollama NÃO precisa estar rodando
python ingest.py

# Saída esperada:
# Processando desenvolvimento_paranaense.pdf...
# Carregando chunks da base de dados existente...
# Criando embeddings para 5000 chunks...
# Indexando no FAISS...
# ✓ Ingestão completa em 18min

# Tempo esperado: 15-30min em CPU puro (paciência)
```

**Resultado final**:
```
data/processed/
  ├─ index.faiss          (índice vetorial HNSW, ~6 MB)
  ├─ id_map.json          (mapping posição → chunk_id)
  └─ metadata.db          (SQLite com texto + metadata dos chunks)
```

---

## 6. Rodando a Aplicação

### 6.1 Terminal 1: Ollama (já rodando)

```powershell
ollama serve
# Listening on 127.0.0.1:11434
```

### 6.2 Terminal 2: FastAPI

```powershell
# Venv ativado
python -m uvicorn src.interface.api.main:app --reload --host 0.0.0.0 --port 8000

# Saída:
# Uvicorn running on http://127.0.0.1:8000
# API docs: http://127.0.0.1:8000/docs
```

### 6.3 Terminal 3: Streamlit (Interface)

```powershell
# Venv ativado
streamlit run src/interface/streamlit_app.py

# Abre automaticamente em http://localhost:8501
```

### 6.4 Fluxo de Teste

1. Abrir http://localhost:8501
2. Digitar pergunta: `"Qual é o PIB do Paraná?"`
3. Aguardar (~10-15s na primeira, ~5-8s nas seguintes)
4. Ver resposta + prompt_used + sources
5. Expandir colapsíveis para auditoria completa

---

## 7. Testes

### 7.1 Testes Unitários

```powershell
pytest tests/ -v --tb=short

# Esperado: 114+ testes passando
# Tempo: ~2-5min (lento em CPU, mas passam)

# Se algum falhar:
pytest tests/ -v --tb=short -k "nome_do_test"  # rodá-lo isolado
```

### 7.2 Avaliação Completa (11 Perguntas)

```powershell
# Ollama deve estar rodando (Terminal 1)
python scripts/run_evaluation.py

# Executa 11 perguntas de teste
# Valida out_of_scope, sources, qualidade
# Tempo esperado: ~2-3min

# Output:
# ✓ test_01_pib_parana.json
# ✓ test_02_variacao_credito.json
# ...
# Summary: 11/11 passed
```

---

## 8. Problemas Conhecidos + Soluções

### ❌ Ollama não conecta

**Erro**: `OllamaUnavailableError: Failed to connect to http://localhost:11434`

**Solução**:
```powershell
# Verificar que Terminal 1 tem "ollama serve" rodando
# Se não:
ollama serve

# Aguardar 2-3s para inicializar
# Se ainda falhar:
curl http://localhost:11434/api/tags  # testar conectividade
```

### ❌ FAISS crash (AVX-512)

**Erro**: `Fatal Error: HW capability found: 0xFFCBFBFF, but HW capability requested: 0x200000`

**Solução**:
```powershell
pip uninstall faiss-cpu -y
pip install faiss-cpu==1.7.4
```

### ❌ Timeout atingido

**Erro**: `LLMTimeoutError: llama3.2:3b não respondeu em 180s`

**Causa**: Máquina sobrecarregada (navegador aberto, muitos processos)

**Solução**:
- Fechar Chrome, Discord, Visual Studio, etc.
- Aumentar `timeout_seconds` em config.yaml para 240s
- Reduzir `max_tokens` de 150 → 100

### ❌ OOM (Out of Memory)

**Sintoma**: Sistema congela durante ingestão

**Causa**: 16GB RAM é justo para llama3.2:3b + FAISS + spaCy

**Solução**:
```powershell
# Durante ingestão:
# - Não rodar Ollama (Terminal 1 parado é OK)
# - Não rodar navegador (fecha sozinho se necessário)
# - Não rodar IDE pesada

# Se ainda travar:
# - Dividir PDFs em lotes (não recomendado)
# - Reduzir chunk_size de 512 → 256
```

### ❌ spaCy modelo não encontrado

**Erro**: `OSError: [E050] Can't find model 'pt_core_news_lg'`

**Solução**:
```powershell
python -m spacy download pt_core_news_lg
```

---

## 9. Performance Esperada

### Desenvolvimento (llama3.2:3b em CPU i5-13500)

| Operação | Tempo |
|---|---|
| FAISS busca (top-20 chunks) | ~50-100ms |
| MMR reranking (20 → 3) | ~200ms |
| Cross-encoder score (3 chunks) | ~500ms |
| Ollama prefill + decode | ~7-10s |
| **Total por pergunta** | **~8-12s** |

### Batch de 11 Perguntas

```
Pergunta 1: ~15s (warmup Ollama)
Perguntas 2-11: ~8-10s cada
Total: ~100-120s (~2min)
```

### Comparação com GPU (seu ambiente pessoal)

Se você tem GPU (seu ambiente de trabalho):
- FAISS: ~50ms (igual)
- Cross-encoder: ~200ms (mais rápido)
- Ollama: ~1-3s (muito mais rápido)
- **Total: ~2-5s** (mas código roda igual em CPU)

**Importante**: Seu código NÃO deve assumir GPU. A GPU é apenas acelerador local.

---

## 10. Entrega Final

### 10.1 Checklist Pré-Entrega

```powershell
# Garantir tudo funciona em CPU puro
pytest tests/ -v                      # 114+ testes passando
python scripts/run_evaluation.py      # 11 perguntas OK

# Validar config.yaml
# - model: "llama3.2:3b"
# - timeout_seconds: 180
# - max_tokens: 150

# Garantir requirements.txt tem:
# - faiss-cpu==1.7.4
# - Sem torch CUDA
# - Sem dependências GPU
```

### 10.2 Estrutura do ZIP de Entrega

```
rag-ipardes.zip
├─ src/                             (código-fonte)
│  ├─ domain/
│  ├─ application/
│  ├─ infrastructure/
│  └─ interface/
├─ tests/                            (130+ testes)
├─ scripts/
│  ├─ download_models.py
│  ├─ download_pdfs.py
│  ├─ run_evaluation.py
│  └─ ingest.py
├─ data/
│  ├─ raw/                           (3 PDFs originais)
│  └─ processed/                     (index.faiss + metadata.db)
├─ models/                           (embedder + reranker pré-downloaded)
├─ config.yaml                       (llama3.2:3b, timeout 180s, CPU-only)
├─ docker-compose.yml
├─ requirements.txt                  (faiss-cpu==1.7.4, CPU-only)
├─ CLAUDE.md                         (este arquivo)
└─ README.md                         (instruções PowerShell)
```

### 10.3 O Professor NÃO Precisa Rodar Ingestão

```powershell
# Tudo já pré-processado:
# 1. Descompactar ZIP
# 2. Instalar Python + venv + requirements
# 3. ollama pull llama3.2:3b
# 4. python -m spacy download pt_core_news_lg
# 5. python scripts/run_evaluation.py

# Pronto. data/processed/ já existe, FAISS já indexado.
```

---

## 11. Dicas Finais

### Para Seu Ambiente Pessoal (com GPU)

Se você quer acelerar desenvolvimento:
- Seu código pode rodar em GPU sem mudanças
- Ollama detecta GPU automaticamente (`nvidia-smi`)
- Resposta em 2-5s vs 8-12s em CPU
- **MAS**: Sempre valide em CPU antes de commit

### Para o Código

1. **Nunca assuma GPU**: Remova todo `device="cuda"`, `gpu=True`, etc.
2. **Teste em CPU**: Rode `pytest` e `run_evaluation.py` em CPU regularmente
3. **Timeouts generosos**: 180s é baseline CPU, OK para GPU também
4. **Documentação clara**: CLAUDE.md deixa claro o baseline

### Performance Aceitável?

- ✅ 8-12s por pergunta em CPU puro: aceitável
- ✅ 114+ testes em 2-5min: aceitável
- ✅ 11 perguntas em ~2min: aceitável
- ⚠️ Se o professor tiver CPU ainda mais fraca: timeout pode aumentar, mas `max_retries: 3` garante que não falha

---

## Checklist de Setup (CPU Puro)

### Ambiente
- [ ] Python 3.12 + venv criado
- [ ] faiss-cpu==1.7.4 instalado (verificar com `pip show`)
- [ ] Ollama instalado e funcionando (`ollama serve`)
- [ ] `ollama pull llama3.2:3b` completo

### Dependências
- [ ] `pip install -r requirements.txt` sem erros
- [ ] `python -m spacy download pt_core_news_lg` completo
- [ ] `python scripts/download_models.py` completo
- [ ] `python scripts/download_pdfs.py` completo (ou PDFs já em data/raw/)

### Ingestão
- [ ] `python ingest.py` completo (data/processed/ criado)
- [ ] Validar: `ls data/processed/` mostra `index.faiss`, `id_map.json`, `metadata.db`

### Testes
- [ ] `pytest tests/ -v` com 114+ tests passando
- [ ] `python scripts/run_evaluation.py` com 11 perguntas resolvidas

### Runtime
- [ ] Terminal 1: `ollama serve` rodando
- [ ] Terminal 2: `python -m uvicorn src.interface.api.main:app ...`
- [ ] Terminal 3: `streamlit run src/interface/streamlit_app.py`
- [ ] Primeira pergunta em http://localhost:8501 retorna em ~10-15s

---

## ZIP de Entrega

**Estrutura obrigatória**:

```
rag-ipardes.zip
├─ scripts/
│  ├─ download_pdfs.py         (URLs dos PDFs APENAS aqui)
│  ├─ download_models.py       (Ollama, spaCy, embedding, reranker)
│  ├─ ingest.py
│  └─ run_evaluation.py
├─ src/
│  ├─ domain/                  (entidades + ports)
│  ├─ application/             (use cases)
│  ├─ infrastructure/          (TRANSFORMERS_OFFLINE=1 obrigatório)
│  └─ interface/               (FastAPI + Streamlit)
├─ tests/                       (130+ testes unitários)
├─ data/
│  ├─ raw/                     (3 PDFs do IPARDES)
│  └─ processed/               (index.faiss + metadata.db)
├─ models/                      (pesos embedding + reranker)
├─ config.yaml                  (SEM URLs, SEM paths absolutos)
├─ docker-compose.yml
├─ requirements.txt             (faiss-cpu==1.7.4)
├─ CLAUDE.md                    (este arquivo)
└─ README.md                    (setup vs runtime bem separado)
```

### O que o Professor Não Precisa Fazer

```powershell
# Setup já feito:
# ✅ PDFs baixados (data/raw/)
# ✅ FAISS indexado (data/processed/)
# ✅ Modelos em cache (models/)

# Professor executa apenas:
pip install -r requirements.txt
ollama pull llama3.2:3b
python -m spacy download pt_core_news_lg
python scripts/run_evaluation.py

# Pronto. Sem internet adicional, sem ingestão.
```

---



Próximos passos: Executa os comandos na ordem do checklist acima.
