import os
import pickle

import faiss
import numpy as np


class FAISSStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []

    def _to_float32(self, embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = embeddings.copy()
        faiss.normalize_L2(embeddings)
        return embeddings

    def add(self, embeddings, texts):
        embeddings = self._to_float32(embeddings)
        embeddings = self._normalize(embeddings)

        self.index.add(embeddings)
        self.texts.extend(texts)

    def save(
        self,
        index_path="vectorstore/faiss.index",
        metadata_path="vectorstore/metadata.pkl",
    ):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        faiss.write_index(self.index, index_path)

        with open(metadata_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(
        self,
        index_path="vectorstore/faiss.index",
        metadata_path="vectorstore/metadata.pkl",
    ):
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                "Índice FAISS não encontrado. Rode build_index.py primeiro."
            )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                "Metadados não encontrados. Rode build_index.py primeiro."
            )

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.texts = pickle.load(f)

    def search(self, query_embedding, top_k=5):
        query_embedding = self._to_float32(query_embedding)
        query_embedding = self._normalize(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            if idx >= len(self.texts):
                continue

            item = self.texts[idx]

            if isinstance(item, dict):
                result = item.copy()
                result["score"] = float(score)
            else:
                result = {
                    "content": str(item),
                    "score": float(score),
                }

            results.append(result)

        return results

    def get_dimension(self):
        return self.index.d