import os
import pickle

import faiss
import numpy as np


class FAISSStore:
    def __init__(self, dimension: int):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("A dimensão do índice FAISS deve ser um inteiro positivo.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def _to_float32(self, embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.size == 0:
            raise ValueError("Embeddings vazios.")

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings devem ser 2D. Shape recebido: {embeddings.shape}")

        return embeddings

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = embeddings.copy()
        faiss.normalize_L2(embeddings)
        return embeddings

    def add(self, embeddings, metadata_items):
        embeddings = self._to_float32(embeddings)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimensão incompatível: embeddings têm dimensão {embeddings.shape[1]}, "
                f"mas o índice espera {self.dimension}."
            )

        if len(metadata_items) != embeddings.shape[0]:
            raise ValueError(
                f"Quantidade de embeddings ({embeddings.shape[0]}) diferente da quantidade "
                f"de metadados ({len(metadata_items)})."
            )

        embeddings = self._normalize(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(metadata_items)

    def save(
        self,
        index_path="vectorstore/faiss.index",
        metadata_path="vectorstore/metadata.pkl",
    ):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        faiss.write_index(self.index, index_path)

        payload = {
            "dimension": self.dimension,
            "metadata": self.metadata,
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(payload, f)

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
            payload = pickle.load(f)

        # compatibilidade com versão antiga
        if isinstance(payload, dict):
            loaded_dimension = payload.get("dimension", self.index.d)
            self.metadata = payload.get("metadata", [])
        else:
            loaded_dimension = self.index.d
            self.metadata = payload

        self.dimension = self.index.d

        if loaded_dimension != self.index.d:
            raise ValueError(
                f"Inconsistência entre metadados e índice: metadados indicam dimensão "
                f"{loaded_dimension}, mas índice carregado possui dimensão {self.index.d}."
            )

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Inconsistência entre índice e metadados: índice possui {self.index.ntotal} vetores, "
                f"mas há {len(self.metadata)} metadados."
            )

    def search(self, query_embedding, top_k=5):
        if self.index.ntotal == 0:
            return []

        query_embedding = self._to_float32(query_embedding)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Dimensão incompatível na consulta: embedding tem dimensão {query_embedding.shape[1]}, "
                f"mas o índice espera {self.dimension}."
            )

        query_embedding = self._normalize(query_embedding)

        safe_top_k = max(1, min(int(top_k), self.index.ntotal))
        scores, indices = self.index.search(query_embedding, safe_top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue

            if idx >= len(self.metadata):
                continue

            item = self.metadata[idx]

            if isinstance(item, dict):
                result = item.copy()
            else:
                result = {"content": str(item)}

            result.setdefault("content", "")
            result.setdefault("filename", "arquivo_desconhecido")
            result.setdefault("titulo", "trecho_sem_titulo")

            result["score"] = float(score)
            result["rank"] = rank
            result["index_id"] = int(idx)

            results.append(result)

        return results

    def get_dimension(self):
        return self.index.d

    def get_total_vectors(self):
        return self.index.ntotal