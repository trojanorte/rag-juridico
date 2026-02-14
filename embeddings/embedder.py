from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("Carregando modelo de embeddings...")
        self.model = SentenceTransformer(model_name)
        print("Modelo carregado.")

    def embed_texts(self, texts):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return np.array(embeddings)
