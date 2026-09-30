"""Local retrieval inspection. Requires an explicit document ID."""
import argparse
from core.guardrails import check_input
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--document-id")
    args = parser.parse_args()
    store = FAISSStore().load()
    if not args.document_id:
        for document in store.list_documents():
            print(document["document_id"], document["filename"], document["document_title"])
        return
    result = check_input(args.question or "")
    if not result.ok:
        parser.error(result.reason)
    if args.document_id not in {doc["document_id"] for doc in store.list_documents()}:
        parser.error("document_id não encontrado")
    vector = Embedder().embed_query(args.question)
    for hit in store.search(vector, document_id=args.document_id):
        print(hit["score"], hit["chunk_id"], hit["clause_title"], hit["content"][:500])

if __name__ == "__main__":
    main()
