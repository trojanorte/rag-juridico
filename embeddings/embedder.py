import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name="intfloat/multilingual-e5-base",
        batch_size=32,
        device=None,
    ):
        print(f"Carregando modelo de embeddings: {model_name}")

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

        print("Modelo carregado.")

    def embed_texts(self, texts):
        """
        Gera embeddings para documentos (chunks do RAG).
        Para o modelo E5 é recomendado usar o prefixo 'passage:'.
        """

        if not texts:
            raise ValueError("Lista de textos vazia para geração de embeddings.")

        texts = [f"passage: {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings com formato inválido: {embeddings.shape}")

        return embeddings

    def embed_query(self, text: str):
        """
        Gera embedding para consulta do usuário.
        O modelo E5 recomenda o prefixo 'query:'.
        """

        if not text or not text.strip():
            raise ValueError("Consulta vazia para geração de embedding.")

        query = f"query: {text.strip()}"

        embeddings = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(embeddings, dtype="float32")

        return embeddings