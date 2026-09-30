import numpy as np
from core.config import NO_RELEVANT_CONTEXT, UNVERIFIED_CITATION
from vectorstore.faiss_store import FAISSStore
from observability.telemetry import telemetry
import rag_generator as rag

def item(doc, chunk, text):
    return {"document_id": doc, "document_hash": doc, "chunk_id": chunk,
            "document_title": doc, "filename": doc + ".docx", "clause_number": "1",
            "clause_title": "CLÁUSULA PRIMEIRA", "content": text}

def test_filtered_search_never_returns_other_document():
    store = FAISSStore(2)
    store.add(np.array([[1, 0], [0.99, 0.1]], dtype="float32"),
              [item("A", "A:1", "A"), item("B", "B:1", "B")])
    hits = store.search(np.array([[1, 0]], dtype="float32"), document_id="B")
    assert [hit["document_id"] for hit in hits] == ["B"]


def test_global_search_keeps_document_identity_and_citations():
    store = FAISSStore(2)
    store.add(np.array([[1, 0], [0.99, 0.1]], dtype="float32"),
              [item("A", "A:1", "Piso A."), item("B", "B:1", "Piso B.")])
    hits = store.search(np.array([[1, 0]], dtype="float32"), top_k=2)
    assert {hit["document_id"] for hit in hits} == {"A", "B"}

    class Embedder:
        def embed_query(self, question):
            return np.array([[1, 0]], dtype="float32")

    context, sources = rag.retrieve_context(Embedder(), store, "piso", None, top_k=2)
    assert "Convenção: A" in context and "Convenção: B" in context
    assert {source["document_id"] for source in sources} == {"A", "B"}
    assert rag.validate_citations("Piso A [1]. Piso B [2].", sources, None)[1] == sources

def test_threshold_and_no_evidence(monkeypatch):
    class Embedder:
        def embed_query(self, question):
            return np.array([[1, 0]], dtype="float32")
    class Store:
        def search(self, *_args, **_kwargs):
            return [{**item("A", "A:1", "Texto jurídico."), "score": -0.1}]
    telemetry.reset()
    context, sources = rag.retrieve_context(Embedder(), Store(), "jornada", "A")
    assert context == "" and sources == []
    assert telemetry.metrics["no_relevant_context"] is True
    assert rag.validate_citations("Resposta [1]", [], "A") == (UNVERIFIED_CITATION, [])

def test_pipeline_global_skips_llm_without_context(monkeypatch):
    class Store:
        def list_documents(self):
            return [{"document_id": "A"}]
    monkeypatch.setattr(rag, "load_components", lambda: (object(), Store()))
    monkeypatch.setattr(rag, "retrieve_context", lambda *_args: ("", []))
    monkeypatch.setattr(rag, "generate_answer", lambda *_args: (_ for _ in ()).throw(AssertionError("LLM called")))
    assert "nas convenções" in rag.answer_question("Qual é a jornada?")[0]
    assert rag.answer_question("Qual é a jornada?", document_id="A")[0] == NO_RELEVANT_CONTEXT

def test_pipeline_responds_and_rewrites_only_contextual(monkeypatch):
    class Store:
        def list_documents(self):
            return [{"document_id": "A"}]
    source = {"id": 1, "document_id": "A"}
    monkeypatch.setattr(rag, "load_components", lambda: (object(), Store()))
    monkeypatch.setattr(rag, "retrieve_context", lambda *_args: ("texto", [source]))
    monkeypatch.setattr(rag, "generate_answer", lambda *_args: "Resposta [1]")
    monkeypatch.setattr(rag, "rewrite_question_with_llm", lambda *_args: "Qual é o percentual do reajuste?")
    answer, sources = rag.answer_question("Qual é o adicional noturno?", "Pergunta anterior 1: jornada", "A")
    assert answer == "Resposta [1]" and sources == [source]
    assert telemetry.metrics["query_rewrite_used"] is False
    rag.answer_question("E qual é o percentual?", "Pergunta anterior 1: reajuste", "A")
    assert telemetry.metrics["query_rewrite_used"] is True


def test_greetings_scope_and_global_followup(monkeypatch):
    assert rag.answer_question("Boa tarde")[0].startswith("Boa tarde")
    assert rag.answer_question("Good afternoon")[0].startswith("Good afternoon")
    assert rag.answer_question("Hola")[0].startswith("¡Hola")
    assert rag.answer_question("Hola, ¿cómo estás?")[0].startswith("¡Hola")
    assert rag.is_in_scope("me fale sobre hora extra")
    assert rag.is_in_scope("qual o piso salarial?")
    assert rag.needs_rewrite("e domingo?")
    assert rag.needs_rewrite("qual percentual?")
    assert rag.needs_rewrite("e para quem trabalha à noite?")

    class Store:
        def list_documents(self):
            return [{"document_id": "A"}, {"document_id": "B"}]

    calls = []
    source = {"id": 1, "document_id": "B"}
    selected_source = {"id": 1, "document_id": "A"}
    monkeypatch.setattr(rag, "load_components", lambda: (object(), Store()))
    monkeypatch.setattr(rag, "retrieve_context", lambda _e, _s, query, document_id: (
        calls.append((query, document_id)) or
        ("texto", [selected_source if document_id == "A" else source])))
    monkeypatch.setattr(rag, "generate_answer", lambda *_args: "Resposta [1]")
    monkeypatch.setattr(rag, "update_conversation_state", lambda *_args: None)
    monkeypatch.setattr(rag, "rewrite_question_with_llm", lambda *_args: "Como funciona no domingo?")

    assert rag.answer_question("me fale sobre hora extra")[1] == [source]
    assert rag.answer_question("qual o piso salarial?")[1] == [source]
    assert rag.answer_question("qual o piso salarial?", document_id="A")[1] == [selected_source]
    assert calls[-1] == ("qual o piso salarial?", "A")
    assert rag.answer_question("e domingo?", "Pergunta anterior 1: hora extra")[1] == [source]
    assert calls[-1] == ("Como funciona no domingo?", None)
    assert telemetry.metrics["query_rewrite_used"] is True


def test_chat_context_keeps_history_but_isolates_selected_agreement(monkeypatch):
    import app

    state = {"selected_document": "A", "chat_history": [
        {"question": "Piso global?", "answer": "Resposta B", "document_id": None},
        {"question": "Piso A?", "answer": "Resposta A", "document_id": "A"},
        {"question": "Piso B?", "answer": "Resposta B", "document_id": "B"},
    ]}
    monkeypatch.setattr(app.st, "session_state", state)
    assert "Piso A?" in app.conversation_context()
    assert "Piso B?" not in app.conversation_context()
    assert len(state["chat_history"]) == 3


def test_generation_keeps_user_language_and_separates_global_sources(monkeypatch):
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"output_text": "Answer [1]"})()

    monkeypatch.setattr(rag, "get_openai_client", lambda: type("Client", (), {"responses": Responses()})())
    assert rag.generate_answer("What is the overtime rate?", None) == "Answer [1]"
    assert "mesmo idioma do usuário" in captured["instructions"]
    assert "convenções recuperadas" in captured["instructions"]
    assert "nunca funda cláusulas" in captured["instructions"]
