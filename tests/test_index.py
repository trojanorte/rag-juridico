import numpy as np
from vectorstore.faiss_store import FAISSStore

def test_index_manifest_and_jsonl_roundtrip(tmp_path):
    store = FAISSStore(2)
    store.add(np.array([[1, 0]], dtype="float32"), [{"document_id": "A", "document_hash": "abc",
               "chunk_id": "A:1", "document_title": "A", "filename": "a.docx", "content": "texto"}])
    manifest = store.save(tmp_path)
    loaded = FAISSStore().load(tmp_path)
    assert manifest["chunk_count"] == 1
    assert loaded.search([[1, 0]], document_id="A")[0]["chunk_id"] == "A:1"
    assert (tmp_path / "CURRENT").read_text() == manifest["index_version"]
