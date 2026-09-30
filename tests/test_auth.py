from types import SimpleNamespace

import pytest

from core import auth


class StopPage(Exception):
    pass


class RerunPage(Exception):
    pass


def fake_streamlit(entered=""):
    errors = []
    fields = []

    def text_input(label, **kwargs):
        fields.append((label, kwargs))
        return entered

    return SimpleNamespace(
        session_state={}, error=errors.append, text_input=text_input,
        stop=lambda: (_ for _ in ()).throw(StopPage()),
        rerun=lambda: (_ for _ in ()).throw(RerunPage()),
        errors=errors, fields=fields,
    )


def test_admin_requires_only_dedicated_password(monkeypatch):
    st = fake_streamlit()
    requested = []
    monkeypatch.setattr(auth, "st", st)
    monkeypatch.setattr(auth, "_secret", lambda name: requested.append(name) or "")

    with pytest.raises(StopPage):
        auth.require_admin_access()

    assert requested == ["LEXRAG_ADMIN_PASSWORD"]
    assert st.errors == ["Acesso administrativo não configurado. Defina LEXRAG_ADMIN_PASSWORD."]
    assert not st.fields


def test_admin_rejects_wrong_password(monkeypatch):
    st = fake_streamlit("wrong")
    monkeypatch.setattr(auth, "st", st)
    monkeypatch.setattr(auth, "_secret", lambda name: "correct")

    with pytest.raises(StopPage):
        auth.require_admin_access()

    assert st.errors == ["Credencial inválida."]
    assert st.session_state == {}


def test_admin_accepts_password(monkeypatch):
    st = fake_streamlit("correct")
    monkeypatch.setattr(auth, "st", st)
    monkeypatch.setattr(auth, "_secret", lambda name: "correct")

    with pytest.raises(RerunPage):
        auth.require_admin_access()

    assert st.session_state["lexrag_admin_authenticated"] is True
    assert st.fields[0][0] == "Senha de administração"
