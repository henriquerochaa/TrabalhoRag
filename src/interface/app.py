from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st

from src.infrastructure.config_loader import load_config

_cfg = load_config()
# API_BASE_URL env var sobrescreve config.yaml — necessário em Docker, onde o
# serviço api é alcançado pelo nome do serviço (http://api:8000) e não por localhost
_API_URL = os.environ.get("API_BASE_URL", _cfg["api"]["base_url"]) + "/chat"


def _post_chat(question: str) -> dict:
    payload = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        _API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


st.set_page_config(page_title="RAG IPARDES", layout="centered")
st.title("RAG IPARDES")
st.caption("Consulta aos documentos oficiais do governo do Paraná.")

question = st.text_area(
    "Pergunta",
    placeholder="Digite sua pergunta sobre os documentos do IPARDES...",
    height=100,
)

if st.button("Enviar", disabled=not question.strip()):
    with st.spinner("Consultando..."):
        try:
            result = _post_chat(question.strip())
        except urllib.error.URLError as exc:
            st.error(f"Erro ao conectar à API em {_API_URL}: {exc}")
            st.stop()

    st.subheader("Resposta")
    if result.get("out_of_scope"):
        st.warning(result["answer"])
    else:
        st.write(result["answer"])

    with st.expander("Prompt enviado à LLM"):
        prompt = result.get("prompt_used", "")
        st.code(prompt if prompt else "(resposta fora do escopo — LLM não foi consultada)", language="text")

    with st.expander("Fontes utilizadas"):
        sources = result.get("sources", [])
        if sources:
            for src in sources:
                section_info = f", seção *{src['section']}*" if src.get("section") else ""
                st.markdown(f"- **{src['document_id']}** — página {src['page']}{section_info}")
        else:
            st.write("Nenhuma fonte.")
