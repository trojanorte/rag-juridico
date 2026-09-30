from ingest.parser import chunk_document, split_by_clausula
from build_index import deduplicate_chunks

def test_normal_clause_and_short_clause_are_preserved():
    chunks = chunk_document("CLÁUSULA PRIMEIRA - VIGÊNCIA\nPrazo de um ano.\nCLÁUSULA SEGUNDA - PISO\nR$ 100.", "a.docx")
    assert len(chunks) == 2
    assert chunks[0]["clause_number"] == "PRIMEIRA"
    assert chunks[1]["content"].endswith("R$ 100.")
    assert chunks[1]["page"] is None

def test_long_clause_splits_without_losing_title_or_text():
    text = "CLÁUSULA PRIMEIRA - REGRAS\n" + "Regra de jornada. " * 100
    chunks = chunk_document(text, "a.docx", max_tokens=30)
    assert len(chunks) > 1
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert all(c["clause_title"].startswith("CLÁUSULA PRIMEIRA") for c in chunks)
    assert all(c["document_id"] == chunks[0]["document_id"] for c in chunks)

def test_fallback_and_preamble():
    assert split_by_clausula("Texto inicial.")[0]["titulo"] == "Texto sem cláusula identificada"
    assert split_by_clausula("Preâmbulo\nCLÁUSULA PRIMEIRA - TESTE\nRegra.")[0]["titulo"] == "Preâmbulo"

def test_deduplication():
    chunks = chunk_document("CLÁUSULA PRIMEIRA - TESTE\nRegra.", "a.docx")
    assert len(deduplicate_chunks(chunks + chunks)) == len(chunks)
