from ingest.parser import load_and_chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
import numpy as np

print("Carregando documentos...")
chunks = load_and_chunk_documents("convencoes coletivas")

texts = [chunk["content"] for chunk in chunks]

embedder = Embedder()
embeddings = embedder.embed_texts(texts)

dimension = embeddings.shape[1]
store = FAISSStore(dimension)

store.add(embeddings, chunks)
store.save()

print("Indexação concluída.")
print(f"Total de documentos indexados: {len(chunks)}")
