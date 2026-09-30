"""Offline index builder. Run from the repository root after DOCX conversion."""
import argparse
import logging
from pathlib import Path

from core.config import EMBEDDING_MODEL
from embeddings.embedder import Embedder
from ingest.parser import PARSER_VERSION, load_and_chunk_documents
from vectorstore.faiss_store import FAISSStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for chunk in chunks:
        if chunk["chunk_id"] in seen:
            log.warning("Duplicate chunk ignored: %s", chunk["chunk_id"])
            continue
        seen.add(chunk["chunk_id"])
        unique.append(chunk)
    return unique

def fit_model_window(chunks: list[dict], tokenizer) -> list[dict]:
    fitted = []
    for chunk in chunks:
        if len(tokenizer.encode("passage: " + chunk["content"], add_special_tokens=True)) <= 512:
            fitted.append(chunk)
            continue
        words = chunk["content"].split()
        group, part = [], 1
        for word in words:
            if group and len(tokenizer.encode("passage: " + " ".join(group + [word]), add_special_tokens=True)) > 512:
                fitted.append({**chunk, "chunk_id": f"{chunk['chunk_id']}:sub_{part}", "content": " ".join(group)})
                group, part = [], part + 1
            group.append(word)
        if group:
            fitted.append({**chunk, "chunk_id": f"{chunk['chunk_id']}:sub_{part}", "content": " ".join(group)})
        log.warning("Split oversize E5 chunk: %s into %s", chunk["chunk_id"], part)
    return fitted

def main(documents_dir: str = "convencoes coletivas", output_dir: str = "vectorstore") -> dict:
    chunks = deduplicate_chunks(load_and_chunk_documents(documents_dir))
    if not chunks:
        raise RuntimeError("Nenhum chunk gerado")
    embedder = Embedder(EMBEDDING_MODEL)
    tokenizer = embedder.model.tokenizer
    chunks = fit_model_window(chunks, tokenizer)
    embeddings = embedder.embed_texts([chunk["content"] for chunk in chunks])
    store = FAISSStore(embeddings.shape[1])
    store.add(embeddings, chunks)
    manifest = store.save(Path(output_dir), PARSER_VERSION)
    log.info("Indexed %s documents and %s chunks (version %s)", manifest["document_count"], manifest["chunk_count"], manifest["index_version"])
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-dir", default="convencoes coletivas")
    parser.add_argument("--output-dir", default="vectorstore")
    args = parser.parse_args()
    main(args.documents_dir, args.output_dir)
