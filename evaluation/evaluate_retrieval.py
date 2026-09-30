"""Offline retrieval benchmark. No LLM/API calls; hybrid remains experimental."""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
from core.config import MIN_RELEVANCE_SCORE

DATASET = Path("evaluation/gold_dataset.json")

def load_dataset():
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    assert 15 <= len(rows) <= 25
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["document_id"] and isinstance(row["expected_chunk_ids"], list)
        assert bool(row["expected_chunk_ids"]) == row["answerable"]
    return rows

def words(value):
    return re.findall(r"\w+", value.lower())

def bm25(query, items):
    query_terms = set(words(query))
    docs = [words(item["content"] + " " + item["clause_title"]) for item in items]
    average = sum(map(len, docs)) / max(1, len(docs))
    df = {term: sum(term in doc for doc in docs) for term in query_terms}
    scores = []
    for doc in docs:
        score = 0.0
        for term in query_terms:
            tf = doc.count(term)
            if tf:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(doc) / max(1, average)))
        scores.append(score)
    scale = max(scores, default=0) or 1
    return [score / scale for score in scores]

def metrics(rows, rankings):
    positives = [(row, rankings[row["id"]]) for row in rows if row["answerable"]]
    count = len(positives)
    positions = []
    for row, ranked in positives:
        expected = set(row["expected_chunk_ids"])
        positions.append(next((i for i, item in enumerate(ranked, 1) if item["chunk_id"] in expected), None))
    return {"cases": count, "hit@1": sum(p is not None and p <= 1 for p in positions) / count,
            "hit@3": sum(p is not None and p <= 3 for p in positions) / count,
            "hit@5": sum(p is not None and p <= 5 for p in positions) / count,
            "recall@5": sum(len(set(row["expected_chunk_ids"]) & {x["chunk_id"] for x in ranked[:5]}) / len(row["expected_chunk_ids"]) for row, ranked in positives) / count,
            "MRR": sum(1 / p for p in positions if p) / count}

def benchmark():
    rows = load_dataset()
    store = FAISSStore().load()
    ids = {item["chunk_id"] for item in store.metadata}
    for row in rows:
        if row["answerable"] and not set(row["expected_chunk_ids"]) <= ids:
            raise ValueError(f"Gold IDs missing from index: {row['id']}")
    embedder = Embedder()
    results = {"A_global": {}, "B_filtered": {}, "C_filtered_bm25": {}}
    durations = {name: [] for name in results}
    negative_rejections = 0
    for row in rows:
        vector = embedder.embed_query(row["question"])
        # A models the old global retrieval for comparison only.
        start = time.perf_counter()
        scores, positions = store.index.search(store._vectors(vector), 5)
        results["A_global"][row["id"]] = [{**store.metadata[int(pos)], "score": float(score)} for score, pos in zip(scores[0], positions[0]) if pos >= 0]
        durations["A_global"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        filtered = store.search(vector, top_k=store.index.ntotal, document_id=row["document_id"])
        results["B_filtered"][row["id"]] = filtered[:5]
        durations["B_filtered"].append((time.perf_counter() - start) * 1000)
        if not row["answerable"] and not any(item["score"] >= MIN_RELEVANCE_SCORE for item in filtered[:5]):
            negative_rejections += 1
        start = time.perf_counter()
        lexical = bm25(row["question"], filtered)
        combined = sorted(zip(filtered, lexical), key=lambda pair: 0.7 * pair[0]["score"] + 0.3 * pair[1], reverse=True)
        results["C_filtered_bm25"][row["id"]] = [item for item, _ in combined[:5]]
        durations["C_filtered_bm25"].append((time.perf_counter() - start) * 1000)
    report = {name: {**metrics(rows, ranking), "retrieval_stage_ms_mean": sum(durations[name]) / len(rows)} for name, ranking in results.items()}
    report["threshold_negative_rejections"] = {"count": negative_rejections, "total": sum(not row["answerable"] for row in rows), "threshold": MIN_RELEVANCE_SCORE}
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-dataset", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    print(f"Gold cases: {len(load_dataset())}")
    if args.benchmark:
        print(json.dumps(benchmark(), indent=2))
