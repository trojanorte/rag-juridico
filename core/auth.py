"""Password gate for LexRAG administrative pages."""

import hmac
import os

import streamlit as st


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or os.getenv(name, ""))
    except Exception:
        return os.getenv(name, "")


def require_admin_access() -> None:
    expected = _secret("LEXRAG_ADMIN_PASSWORD")
    if not expected:
        st.error("Acesso administrativo não configurado. Defina LEXRAG_ADMIN_PASSWORD.")
        st.stop()
    key = "lexrag_admin_authenticated"
    if st.session_state.get(key):
        return
    entered = st.text_input("Senha de administração", type="password", key=key + "_input")
    if entered and hmac.compare_digest(entered, expected):
        st.session_state[key] = True
        st.rerun()
    if entered:
        st.error("Credencial inválida.")
    st.stop()
