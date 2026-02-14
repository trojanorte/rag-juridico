from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
import logging


logging.basicConfig(level=logging.INFO)


def load_components():
    """
    Inicializa o modelo de embeddings e carrega o índice FAISS.
    """
    logging.info("Inicializando modelo de embeddings...")
    embedder = Embedder()

    logging.info("Carregando índice vetorial...")
    store = FAISSStore(384)
    store.load()

    return embedder, store


def search(embedder, store, query, top_k=5):
    """
    Executa busca semântica no índice vetorial.
    """
    query_embedding = embedder.embed_texts([query])
    return store.search(query_embedding, top_k=top_k)


def print_results(results):
    """
    Exibe os resultados encontrados de forma organizada.
    """
    if not results:
        print("\nNenhum resultado encontrado.\n")
        return

    print("\nResultados encontrados:\n")

    for i, item in enumerate(results, start=1):
        print(f"Resultado {i}")
        print("Arquivo:", item["filename"])
        print("Cláusula:", item["titulo"])
        print("Trecho:\n", item["content"][:500])
        print("-" * 80)


def main():
    embedder, store = load_components()

    while True:
        pergunta = input("\nDigite sua pergunta (ou 'sair'): ").strip()

        if pergunta.lower() == "sair":
            print("\nEncerrando busca.")
            break

        resultados = search(embedder, store, pergunta)
        print_results(resultados)


if __name__ == "__main__":
    main()
