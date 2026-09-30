import logging
import numpy as np

from ingest.parser import load_and_chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DOCUMENTS_DIR = "convencoes coletivas"
MIN_CHUNK_LEN = 80


def is_valid_chunk(chunk: dict) -> bool:
    content = (chunk.get("content") or "").strip()
    if not content:
        return False
    if len(content) < MIN_CHUNK_LEN:
        return False
    return True


def normalize_chunk(chunk: dict) -> dict:
    return {
        "content": (chunk.get("content") or "").strip(),
        "filename": chunk.get("filename", "arquivo_desconhecido"),
        "titulo": chunk.get("titulo", "trecho_sem_titulo"),
    }


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    seen = set()
    unique_chunks = []

    for chunk in chunks:
        key = (
            chunk.get("filename", ""),
            chunk.get("titulo", ""),
            chunk.get("content", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_chunks.append(chunk)

    return unique_chunks


def main():
    logging.info("Carregando e fragmentando documentos...")
    raw_chunks = load_and_chunk_documents(DOCUMENTS_DIR)

    if not raw_chunks:
        raise RuntimeError("Nenhum chunk foi gerado pelo parser.")

    logging.info("Total bruto de chunks: %s", len(raw_chunks))

    normalized_chunks = [normalize_chunk(chunk) for chunk in raw_chunks]
    valid_chunks = [chunk for chunk in normalized_chunks if is_valid_chunk(chunk)]
    unique_chunks = deduplicate_chunks(valid_chunks)

    if not unique_chunks:
        raise RuntimeError("Nenhum chunk válido restou após limpeza e deduplicação.")

    texts = [chunk["content"] for chunk in unique_chunks]

    avg_len = sum(len(text) for text in texts) / len(texts)
    logging.info("Chunks válidos: %s", len(valid_chunks))
    logging.info("Chunks únicos: %s", len(unique_chunks))
    logging.info("Tamanho médio dos chunks: %.1f caracteres", avg_len)

    logging.info("Exemplo de chunk indexado:")
    logging.info("Arquivo: %s", unique_chunks[0]["filename"])
    logging.info("Título: %s", unique_chunks[0]["titulo"])
    logging.info("Trecho: %s", unique_chunks[0]["content"][:500])

    logging.info("Gerando embeddings...")
    embedder = Embedder()
    embeddings = embedder.embed_texts(texts)

    if embeddings is None or len(embeddings) == 0:
        raise RuntimeError("O modelo de embeddings não retornou vetores.")

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype="float32")

    if embeddings.ndim != 2:
        raise RuntimeError(f"Embeddings com formato inválido: {embeddings.shape}")

    dimension = embeddings.shape[1]
    logging.info("Dimensão dos embeddings: %s", dimension)

    store = FAISSStore(dimension)
    store.add(embeddings, unique_chunks)
    store.save()

    logging.info("Indexação concluída com sucesso.")
    logging.info("Total de chunks indexados: %s", len(unique_chunks))


if __name__ == "__main__":
    main()