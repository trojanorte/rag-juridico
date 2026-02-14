import faiss
import numpy as np
import pickle
import os


class FAISSStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(embeddings)
        self.texts.extend(texts)

    def save(self, index_path="vectorstore/faiss.index", metadata_path="vectorstore/metadata.pkl"):
        faiss.write_index(self.index, index_path)

        with open(metadata_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, index_path="vectorstore/faiss.index", metadata_path="vectorstore/metadata.pkl"):
        if not os.path.exists(index_path):
            raise FileNotFoundError("Índice FAISS não encontrado. Rode build_index.py primeiro.")

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.texts = pickle.load(f)

    def search(self, query_embedding, top_k=5):
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.texts[idx])

        return results

    def get_dimension(self):
        return self.index.d
