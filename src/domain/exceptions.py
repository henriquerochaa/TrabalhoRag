from __future__ import annotations


class OllamaUnavailableError(Exception):
    """Lançada pelo OllamaLLM após esgotar todas as tentativas de conexão.

    Mantida no domínio para que GenerateAnswer (camada de aplicação) possa
    capturá-la sem importar nenhuma classe de infraestrutura — preserva a
    regra de que use cases dependem apenas de ports e entidades de domínio.
    """
