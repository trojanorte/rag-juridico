import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_generator import answer_question

EVALUATION_FILE = "evaluation/evaluation_set.json"
RESULTS_FILE = "evaluation/evaluation_results.json"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return text.lower().strip()


def has_valid_sources(sources) -> bool:
    return bool(sources and len(sources) > 0)


def answer_mentions_expected_topic(answer: str, expected_topic: str) -> bool:
    answer_norm = normalize_text(answer)
    topic_norm = normalize_text(expected_topic)

    if not answer_norm or not topic_norm:
        return False

    return topic_norm in answer_norm


def looks_like_weak_answer(answer: str) -> bool:
    answer_norm = normalize_text(answer)

    weak_patterns = [
        "não encontrei",
        "nao encontrei",
        "não foi possível identificar",
        "nao foi possível identificar",
        "não há informação suficiente",
        "nao ha informacao suficiente",
        "não consta",
        "nao consta",
        "não localizei",
        "nao localizei",
        "não identificado",
        "nao identificado",
    ]

    if len(answer_norm) < 30:
        return True

    return any(pattern in answer_norm for pattern in weak_patterns)


def evaluate_single_question(question: str, expected_topic: str) -> dict:
    try:
        answer, sources = answer_question(question)

        source_score = 1 if has_valid_sources(sources) else 0
        topic_score = 1 if answer_mentions_expected_topic(answer, expected_topic) else 0
        weakness_penalty = 1 if looks_like_weak_answer(answer) else 0

        final_score = source_score + topic_score - weakness_penalty
        final_score = max(0, min(final_score, 2))

        return {
            "question": question,
            "expected_topic": expected_topic,
            "answer": answer,
            "sources": sources,
            "source_score": source_score,
            "topic_score": topic_score,
            "weakness_penalty": weakness_penalty,
            "final_score": final_score,
            "status": (
                "good" if final_score == 2
                else "partial" if final_score == 1
                else "bad"
            ),
            "error": None,
        }

    except Exception as exc:
        return {
            "question": question,
            "expected_topic": expected_topic,
            "answer": "",
            "sources": [],
            "source_score": 0,
            "topic_score": 0,
            "weakness_penalty": 0,
            "final_score": 0,
            "status": "error",
            "error": str(exc),
        }


def main() -> None:
    if not os.path.exists(EVALUATION_FILE):
        raise FileNotFoundError(f"Arquivo não encontrado: {EVALUATION_FILE}")

    with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
        evaluation_set = json.load(f)

    results = []
    total_score = 0
    max_score = len(evaluation_set) * 2

    print("=" * 80)
    print("INICIANDO AVALIAÇÃO DO RAG")
    print("=" * 80)

    for idx, item in enumerate(evaluation_set, start=1):
        question = item["question"]
        expected_topic = item["expected_topic"]

        result = evaluate_single_question(question, expected_topic)
        results.append(result)
        total_score += result["final_score"]

        print(f"\n[{idx}/{len(evaluation_set)}]")
        print("PERGUNTA:", result["question"])
        print("TÓPICO ESPERADO:", result["expected_topic"])
        print("STATUS:", result["status"])
        print("SCORE:", f"{result['final_score']}/2")

        if result["error"]:
            print("ERRO:", result["error"])
        else:
            print("FONTES:", len(result["sources"]))
            print("RESPOSTA:", result["answer"][:300], "..." if len(result["answer"]) > 300 else "")

        print("-" * 80)

    overall_percentage = round((total_score / max_score) * 100, 2) if max_score > 0 else 0

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_questions": len(evaluation_set),
        "total_score": total_score,
        "max_score": max_score,
        "overall_percentage": overall_percentage,
        "results": results,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("AVALIAÇÃO FINALIZADA")
    print(f"Score total: {total_score}/{max_score}")
    print(f"Percentual geral: {overall_percentage}%")
    print(f"Resultados salvos em: {RESULTS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()