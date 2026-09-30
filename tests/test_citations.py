from core.config import UNVERIFIED_CITATION, NO_RELEVANT_CONTEXT
from rag_generator import validate_citations
from app import source_preview_data
from evaluation.evaluate_grounding import evaluate_answer

SOURCE = {"id": 1, "document_id": "A", "document_title": "Convenção A", "arquivo": "a.docx",
          "clause_number": "14", "titulo": "Horas Extras", "content": "Adicional previsto.",
          "score": 0.8, "chunk_id": "A:clause_14:part_1"}

def test_valid_and_invalid_citations():
    assert validate_citations("Regra [1]", [SOURCE], "A")[1] == [SOURCE]
    assert validate_citations("Regra [9]", [SOURCE], "A") == (UNVERIFIED_CITATION, [])
    assert validate_citations("Regra [1]", [SOURCE], "B") == (UNVERIFIED_CITATION, [])
    assert validate_citations("Regra sem citação", [SOURCE], "A") == (UNVERIFIED_CITATION, [])

def test_preview_contains_verifiable_excerpt():
    preview = source_preview_data(SOURCE)
    assert preview["content"] == "Adicional previsto."
    assert preview["chunk_id"] == "A:clause_14:part_1"

def test_grounding_checks_negative_and_cross_document():
    assert evaluate_answer(NO_RELEVANT_CONTEXT, [], "A", False)["no_evidence_refused"]
    assert not evaluate_answer("Regra [1]", [SOURCE], "B", True)["correct_document"]
