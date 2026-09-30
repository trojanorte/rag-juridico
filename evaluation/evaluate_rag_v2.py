import json
import os
import re
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_generator import answer_question


EVALUATION_FILE = "evaluation/evaluation_set.json"
RESULTS_FILE = "evaluation/evaluation_results_v2.json"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return text.lower().strip()


def has_valid_sources(sources) -> bool:
    return bool(sources and len(sources) > 0)


def detect_contamination(answer: str) -> bool:
    answer_norm = normalize_text(answer)

    contamination_markers = [
        "\npergunta:",
        "\nresposta:",
        "nova pergunta",
        "pergunta adicional",
    ]

    repeated_patterns = [
        "resposta objetiva:",
        "fundamentação:",
        "fontes:",
    ]

    marker_hit = any(marker in answer_norm for marker in contamination_markers)
    repeated_hit = sum(answer_norm.count(pattern) for pattern in repeated_patterns) > 1

    return marker_hit or repeated_hit


def looks_like_no_evidence(answer: str) -> bool:
    answer_norm = normalize_text(answer)

    patterns = [
        "não encontrei informação suficiente",
        "nao encontrei informacao suficiente",
        "não foi encontrada",
        "nao foi encontrada",
        "não localizei",
        "nao localizei",
        "não consta",
        "nao consta",
        "não há informação suficiente",
        "nao ha informacao suficiente",
    ]

    return any(pattern in answer_norm for pattern in patterns)


def extract_keywords(expected_topic: str) -> list[str]:
    topic = normalize_text(expected_topic)

    keyword_map = {
        "vigência do acordo": ["vigência", "vigencia", "prazo", "validade"],
        "reajuste salarial": ["reajuste", "salário", "salario", "percentual", "correção"],
        "seguro de vida": ["seguro", "vida"],
        "auxílio alimentação": ["alimentação", "alimentacao", "refeição", "refeicao", "ticket"],
        "vale transporte": ["vale-transporte", "vale transporte", "transporte"],
        "piso salarial": ["piso", "salário", "salario"],
        "contribuição assistencial": ["contribuição assistencial", "contribuicao assistencial"],
        "contribuição sindical": ["contribuição sindical", "contribuicao sindical", "sindical"],
        "jornada de trabalho": ["jornada", "horário", "horario", "carga horária", "carga horaria"],
        "banco de horas": ["banco de horas", "compensação", "compensacao"],
        "adicional noturno": ["adicional noturno", "noturno"],
        "horas extras": ["horas extras", "hora extra", "extra"],
        "estabilidade gestante": ["gestante", "gravidez", "licença maternidade", "licenca maternidade"],
        "estabilidade pré aposentadoria": ["pré-aposentadoria", "pre aposentadoria", "aposentadoria"],
        "auxílio creche": ["creche", "pré-escola", "pre escola"],
        "uniformes": ["uniforme", "uniformes"],
        "epi": ["epi", "equipamento de proteção", "equipamento de protecao"],
        "licença casamento": ["casamento", "gala"],
        "licença falecimento": ["falecimento", "óbito", "obito", "luto"],
        "férias": ["férias", "ferias"],
        "plr": ["plr", "participação nos lucros", "participacao nos lucros", "resultados"],
        "plano de saúde": ["plano de saúde", "plano de saude", "assistência médica", "assistencia medica"],
        "qualificação profissional": ["qualificação", "qualificacao", "treinamento", "capacitação", "capacitacao"],
        "aviso prévio": ["aviso prévio", "aviso previo", "demissão", "demissao"],
        "representação sindical": ["representação sindical", "representacao sindical", "sindicato", "dirigente sindical"],
    }

    return keyword_map.get(topic, [topic])


def answer_matches_topic(answer: str, expected_topic: str) -> bool:
    answer_norm = normalize_text(answer)
    keywords = extract_keywords(expected_topic)
    return any(keyword in answer_norm for keyword in keywords)


def sources_match_topic(sources, expected_topic: str) -> bool:
    if not sources:
        return False

    keywords = extract_keywords(expected_topic)

    for source in sources:
        try:
            file_name, excerpt = source
            text = normalize_text(f"{file_name} {excerpt}")
        except Exception:
            text = normalize_text(str(source))

        if any(keyword in text for keyword in keywords):
            return True

    return False


def detect_wrong_topic(answer: str, expected_topic: str) -> bool:
    answer_norm = normalize_text(answer)
    topic = normalize_text(expected_topic)

    confusion_map = {
        "contribuição assistencial": ["assistência médica", "assistencia medica", "plano de saúde", "plano de saude"],
        "representação sindical": ["domingo", "celular", "nr17"],
        "licença falecimento": ["adoção", "adocao"],
        "plr": ["vale transporte", "creche", "uniforme"],
    }

    confusing_terms = confusion_map.get(topic, [])
    return any(term in answer_norm for term in confusing_terms)


def classify_answer(answer: str, sources, expected_topic: str, error: str = None) -> tuple[str, str]:
    if error:
        return "error", "Erro técnico durante a execução."

    if not answer or len(normalize_text(answer)) < 10:
        return "wrong", "Resposta vazia ou muito curta."

    no_evidence = looks_like_no_evidence(answer)
    contaminated = detect_contamination(answer)
    answer_topic_match = answer_matches_topic(answer, expected_topic)
    source_topic_match = sources_match_topic(sources, expected_topic)
    wrong_topic = detect_wrong_topic(answer, expected_topic)

    if no_evidence:
        if source_topic_match:
            return "partial", "O sistema recuperou algo do tema, mas declarou evidência insuficiente."
        return "no_evidence", "O sistema não encontrou evidência suficiente."

    if wrong_topic:
        return "wrong", "A resposta parece tratar de um tema diferente do esperado."

    if answer_topic_match and source_topic_match:
        if contaminated:
            return "correct_but_contaminated", "Conteúdo correto, mas saída contaminada por continuação indevida."
        return "correct", "Conteúdo e fontes compatíveis com o tópico esperado."

    if answer_topic_match or source_topic_match:
        if contaminated:
            return "partial", "Há sinais de acerto, mas com contaminação ou evidência incompleta."
        return "partial", "Há indícios de acerto, mas não suficientes para classificar como totalmente correto."

    return "wrong", "Nem a resposta nem as fontes sustentam bem o tópico esperado."


def main() -> None:
    if not os.path.exists(EVALUATION_FILE):
        raise FileNotFoundError(f"Arquivo não encontrado: {EVALUATION_FILE}")

    with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
        evaluation_set = json.load(f)

    results = []
    summary_counts = {
        "correct": 0,
        "correct_but_contaminated": 0,
        "partial": 0,
        "wrong": 0,
        "no_evidence": 0,
        "error": 0,
    }

    print("=" * 90)
    print("AVALIAÇÃO V2 DO RAG")
    print("=" * 90)

    for idx, item in enumerate(evaluation_set, start=1):
        question = item["question"]
        expected_topic = item["expected_topic"]

        try:
            answer, sources = answer_question(question)
            classification, rationale = classify_answer(answer, sources, expected_topic)

            result = {
                "question": question,
                "expected_topic": expected_topic,
                "answer": answer,
                "sources": sources,
                "classification": classification,
                "rationale": rationale,
                "error": None,
            }

        except Exception as exc:
            classification, rationale = classify_answer("", [], expected_topic, error=str(exc))
            result = {
                "question": question,
                "expected_topic": expected_topic,
                "answer": "",
                "sources": [],
                "classification": classification,
                "rationale": rationale,
                "error": str(exc),
            }

        summary_counts[classification] += 1
        results.append(result)

        print(f"\n[{idx}/{len(evaluation_set)}]")
        print("PERGUNTA:", result["question"])
        print("TÓPICO ESPERADO:", result["expected_topic"])
        print("CLASSIFICAÇÃO:", result["classification"])
        print("JUSTIFICATIVA:", result["rationale"])

        if result["error"]:
            print("ERRO:", result["error"])
        else:
            print("FONTES:", len(result["sources"]))
            print("RESPOSTA:", result["answer"][:350], "..." if len(result["answer"]) > 350 else "")

        print("-" * 90)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_questions": len(evaluation_set),
        "summary_counts": summary_counts,
        "results": results,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("RESUMO FINAL")
    print("=" * 90)
    for key, value in summary_counts.items():
        print(f"{key}: {value}")

    print(f"\nResultados salvos em: {RESULTS_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()