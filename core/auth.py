"""Small, fail-closed Streamlit access gate shared by the chat and admin pages."""

import hmac
import os

import streamlit as st


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or os.getenv(name, ""))
    except Exception:
        return os.getenv(name, "")


def require_access(admin: bool = False) -> None:
    admin_password = _secret("LEXRAG_ADMIN_PASSWORD") or _secret("ADMIN_PASSWORD")
    access_password = _secret("LEXRAG_ACCESS_PASSWORD") or admin_password
    expected = admin_password if admin else access_password
    if not expected:
        st.error("Acesso não configurado. Defina LEXRAG_ACCESS_PASSWORD e LEXRAG_ADMIN_PASSWORD.")
        st.stop()
    key = "lexrag_admin_authenticated" if admin else "lexrag_authenticated"
    if st.session_state.get(key):
        return
    entered = st.text_input("Senha de administração" if admin else "Senha de acesso", type="password", key=key + "_input")
    if entered and hmac.compare_digest(entered, expected):
        st.session_state[key] = True
        st.rerun()
    if entered:
        st.error("Credencial inválida.")
    st.stop()
