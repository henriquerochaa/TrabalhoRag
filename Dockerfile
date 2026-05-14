FROM python:3.12-slim

WORKDIR /app

# build-time: internet disponível para instalar dependências e baixar modelo spaCy
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download pt_core_news_lg

# código-fonte e configuração copiados para a imagem
COPY config.yaml .
COPY src/ src/

# PYTHONPATH garante que imports absolutos (src.domain.*, src.application.*)
# funcionem sem instalar o projeto como pacote
ENV PYTHONPATH=/app

# modelos HuggingFace e dados vêm de volumes montados em runtime — nunca baixados aqui
