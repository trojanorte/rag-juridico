import numpy as np
from core.config import NO_RELEVANT_CONTEXT, UNVERIFIED_CITATION, SELECT_DOCUMENT
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

def test_pipeline_requires_selection_and_skips_llm_without_context(monkeypatch):
    assert rag.answer_question("Qual é a jornada?")[0] == SELECT_DOCUMENT
    class Store:
        def list_documents(self):
            return [{"document_id": "A"}]
    monkeypatch.setattr(rag, "load_components", lambda: (object(), Store()))
    monkeypatch.setattr(rag, "retrieve_context", lambda *_args: ("", []))
    monkeypatch.setattr(rag, "generate_answer", lambda *_args: (_ for _ in ()).throw(AssertionError("LLM called")))
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
