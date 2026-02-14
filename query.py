from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
import numpy as np

# Carregar embedder
embedder = Embedder()

# Carregar índice
dimension = 384  # dimensão do MiniLM
store = FAISSStore(dimension)
store.load()

def search(query, top_k=5):
    query_embedding = embedder.embed_texts([query])
    results = store.search(query_embedding, top_k=top_k)
    return results

if __name__ == "__main__":
    while True:
        pergunta = input("\nDigite sua pergunta (ou 'sair'): ")
        if pergunta.lower() == "sair":
            break

        resultados = search(pergunta)

        print("\nResultados encontrados:\n")
        for i, r in enumerate(resultados):
            print(f"Resultado {i+1}")
            print("Arquivo:", r["filename"])
            print("Cláusula:", r["titulo"])
            print("Trecho:\n", r["content"][:500])
            print("-" * 80)
