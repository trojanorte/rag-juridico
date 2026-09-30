"""FAISS vectors with versioned JSONL metadata and atomic CURRENT pointer."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
import faiss
import numpy as np
from core.config import EMBEDDING_MODEL

ROOT = Path("vectorstore")

class FAISSStore:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []
        self.manifest: dict = {}

    @staticmethod
    def _vectors(embeddings):
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("Embeddings inválidos")
        vectors = vectors.copy()
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, embeddings, metadata_items):
        vectors = self._vectors(embeddings)
        if vectors.shape[1] != self.dimension or len(metadata_items) != len(vectors):
            raise ValueError("Dimensão ou contagem de metadados incompatível")
        if any(not item.get("document_id") or not item.get("chunk_id") for item in metadata_items):
            raise ValueError("Metadados sem IDs estáveis")
        self.index.add(vectors)
        self.metadata.extend(dict(item) for item in metadata_items)

    @staticmethod
    def _validate_files(folder: Path, manifest: dict):
        index = faiss.read_index(str(folder / "faiss.index"))
        rows = [json.loads(line) for line in (folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
        if index.d != manifest["embedding_dimension"] or index.ntotal != manifest["chunk_count"] or len(rows) != index.ntotal:
            raise ValueError("Índice e metadados inconsistentes")
        if any(row.get("faiss_position") != pos for pos, row in enumerate(rows)):
            raise ValueError("Posições FAISS inconsistentes")
        if len({row["document_id"] for row in rows}) != manifest["document_count"]:
            raise ValueError("Contagem de documentos inconsistente")
        documents = sorted({row["document_id"]: row["document_hash"] for row in rows}.items())
        if hashlib.sha256(json.dumps(documents).encode()).hexdigest() != manifest["corpus_hash"]:
            raise ValueError("Hash do corpus inconsistente")
        return index, rows

    def save(self, root: Path = ROOT, parser_version: str = "2") -> dict:
        root = Path(root)
        versions = root / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        version = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="staging-", dir=versions) as temporary:
            staged = Path(temporary)
            faiss.write_index(self.index, str(staged / "faiss.index"))
            with (staged / "metadata.jsonl").open("w", encoding="utf-8") as handle:
                for position, item in enumerate(self.metadata):
                    handle.write(json.dumps({"faiss_position": position, **item}, ensure_ascii=False) + "\n")
            documents = sorted({item["document_id"]: item["document_hash"] for item in self.metadata}.items())
            manifest = {"index_version": version, "created_at": datetime.now(timezone.utc).isoformat(),
                        "embedding_model": EMBEDDING_MODEL, "embedding_dimension": self.dimension,
                        "document_count": len(documents), "chunk_count": len(self.metadata),
                        "corpus_hash": hashlib.sha256(json.dumps(documents).encode()).hexdigest(),
                        "parser_version": parser_version}
            (staged / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self._validate_files(staged, manifest)
            os.replace(staged, versions / version)
        pointer = root / "CURRENT.tmp"
        pointer.write_text(version, encoding="ascii")
        os.replace(pointer, root / "CURRENT")
        self.manifest = manifest
        return manifest

    def load(self, root: Path = ROOT):
        root = Path(root)
        version = (root / "CURRENT").read_text(encoding="ascii").strip()
        if not version.isalnum():
            raise ValueError("Versão inválida")
        folder = root / "versions" / version
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["index_version"] != version or manifest["embedding_model"] != EMBEDDING_MODEL:
            raise ValueError("Manifesto incompatível")
        self.index, self.metadata = self._validate_files(folder, manifest)
        self.dimension, self.manifest = self.index.d, manifest
        return self

    def list_documents(self) -> list[dict]:
        documents = {}
        keys = ("document_id", "document_title", "filename", "document_version", "categoria",
                "sindicato_laboral", "sindicato_patronal", "abrangencia_territorial", "vigencia_inicio", "vigencia_fim")
        for item in self.metadata:
            documents.setdefault(item["document_id"], {key: item.get(key) for key in keys})
        return sorted(documents.values(), key=lambda row: (row["document_title"] or "", row["filename"]))

    def search(self, query_embedding, top_k=5, document_id: str | None = None) -> list[dict]:
        if not document_id:
            raise ValueError("document_id é obrigatório")
        if self.index.ntotal == 0:
            return []
        vector = self._vectors(query_embedding)
        if vector.shape[1] != self.dimension:
            raise ValueError("Dimensão de consulta incompatível")
        scores, positions = self.index.search(vector, self.index.ntotal)
        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            item = self.metadata[int(position)]
            if item["document_id"] == document_id:
                results.append({**item, "score": float(score), "index_id": int(position), "rank": len(results) + 1})
            if len(results) >= top_k:
                break
        return results
