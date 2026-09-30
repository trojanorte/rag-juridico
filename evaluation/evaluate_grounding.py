"""Deterministic structural checks; does not certify legal correctness."""
from core.config import NO_RELEVANT_CONTEXT, UNVERIFIED_CITATION
from rag_generator import validate_citations

def evaluate_answer(answer: str, sources: list[dict], document_id: str, answerable: bool) -> dict:
    checked, cited = validate_citations(answer, sources, document_id)
    same_document = all(source.get("document_id") == document_id for source in sources)
    return {
        "correct_document": same_document,
        "citations_valid": bool(cited) if answerable else not cited,
        "no_cross_document_citation": same_document and (not answerable or checked != UNVERIFIED_CITATION),
        "no_evidence_refused": answer == NO_RELEVANT_CONTEXT if not answerable else None,
        "invalid_citation_rejected": checked == UNVERIFIED_CITATION if answerable and not cited else None,
    }
