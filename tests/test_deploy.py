import numpy as np
import pytest

from vectorstore.deploy import ensure_index_available
from vectorstore.faiss_store import FAISSStore
from vectorstore.package_index import package_index


class FakeResponse:
    status_code = 200

    def __init__(self, content):
        self.content = content

    def iter_content(self, chunk_size):
        yield self.content

    def close(self):
        pass


def test_cloud_startup_loads_private_bundle(tmp_path, monkeypatch):
    source = tmp_path / "source"
    store = FAISSStore(2)
    store.add(np.array([[1, 0]], dtype="float32"), [{
        "document_id": "A", "document_hash": "abc", "chunk_id": "A:1",
        "document_title": "Convenção fictícia", "filename": "example.docx", "content": "Trecho fictício",
    }])
    store.save(source)
    bundle, digest = package_index(source)
    destination = tmp_path / "cloud"
    monkeypatch.setattr("vectorstore.deploy._setting", lambda key: {
        "LEXRAG_INDEX_BUNDLE_URL": "https://private.example/index.zip",
        "LEXRAG_INDEX_BUNDLE_SHA256": digest,
        "LEXRAG_INDEX_BUNDLE_TOKEN": "test-token",
    }[key])

    def get(url, **kwargs):
        assert url.startswith("https://")
        assert kwargs["headers"] == {"Authorization": "Bearer test-token"}
        assert kwargs["allow_redirects"] is False
        return FakeResponse(bundle.read_bytes())

    monkeypatch.setattr("vectorstore.deploy.requests.get", get)
    version = ensure_index_available(destination)
    loaded = FAISSStore().load(destination)
    assert loaded.manifest["index_version"] == version
    assert loaded.list_documents()[0]["document_id"] == "A"
    assert ensure_index_available(destination) == version


def test_cloud_startup_rejects_tampered_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr("vectorstore.deploy._setting", lambda key: {
        "LEXRAG_INDEX_BUNDLE_URL": "https://private.example/index.zip",
        "LEXRAG_INDEX_BUNDLE_SHA256": "0" * 64,
        "LEXRAG_INDEX_BUNDLE_TOKEN": "",
    }[key])
    monkeypatch.setattr("vectorstore.deploy.requests.get", lambda *args, **kwargs: FakeResponse(b"tampered"))
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_index_available(tmp_path / "cloud")
    assert not (tmp_path / "cloud" / "CURRENT").exists()
