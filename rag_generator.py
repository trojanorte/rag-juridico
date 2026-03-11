import logging
import requests
from functools import lru_cache

from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
from observability.decorators import measure
from observability.telemetry import telemetry


logging.basicConfig(level=logging.INFO)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"

TOP_K = 4
MAX_CHARS = 2200
MAX_CHUNK_CHARS = 600
MIN_SCORE = 0.20
MIN_ACCEPTABLE_TOP_SCORE = 0.15


@lru_cache(maxsize=1)
def load_components():
    logging.info("Inicializando modelo de embeddings...")
    embedder = Embedder()

    logging.info("Carregando índice vetorial...")
    store = FAISSStore(384)
    store.load()

    return embedder, store


def normalize_score(score):
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def is_in_scope(question: str) -> bool:
    keywords = [
        "acordo", "convenção", "convencao", "cláusula", "clausula",
        "salário", "salario", "vigência", "vigencia", "jornada",
        "sindicato", "sindical", "benefício", "beneficio", "empresa",
        "empregado", "trabalho", "categoria", "piso", "vale",
        "adicional", "estabilidade", "férias", "ferias", "horas extras",
        "banco de horas", "creche", "uniforme", "epi", "plr",
        "aviso prévio", "aviso previo", "seguro", "licença", "licenca",
        "reajuste", "desconto", "descontos", "valor", "valores",
        "contribuição", "contribuicao", "cláusulas", "clausulas"
    ]
    q = question.lower()
    return any(k in q for k in keywords)

def clean_answer(answer: str) -> str:
    answer = (answer or "").strip()

    stop_markers = [
        "\nPergunta:",
        "\n\nPergunta:",
        "\nResposta:",
        "\n\nResposta:",
        "\nUsuário:",
        "\nUsuario:",
    ]

    cleaned = answer
    for marker in stop_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0].strip()

    return cleaned


def needs_rewrite(question: str) -> bool:
    q = question.lower().strip()

    continuation_starts = [
        "e ",
        "e o ",
        "e a ",
        "e os ",
        "e as ",
        "sobre isso",
        "sobre esse",
        "sobre essa",
        "quanto a",
        "e quanto",
    ]

    if len(q) <= 25:
        return True

    return any(q.startswith(prefix) for prefix in continuation_starts)


def rewrite_question(question: str, conversation_context: str = "") -> str:
    if not conversation_context or not needs_rewrite(question):
        return question

    history_lines = [line.strip() for line in conversation_context.splitlines() if line.strip()]
    last_user_question = None

    for line in reversed(history_lines):
        if line.lower().startswith("pergunta anterior"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                last_user_question = parts[1].strip()
                break

    q = question.strip()

    if not last_user_question:
        return question

    lowered = q.lower()

    if lowered.startswith("e o reajuste") or lowered == "e o reajuste?":
        return f"Qual é o reajuste salarial previsto na mesma convenção coletiva da pergunta anterior: '{last_user_question}'?"

    if lowered.startswith("e o seguro") or lowered == "e o seguro?":
        return f"Existe cláusula de seguro na mesma convenção coletiva da pergunta anterior: '{last_user_question}'?"

    if lowered.startswith("e a vigência") or lowered == "e a vigência?":
        return f"Qual é a vigência da mesma convenção coletiva da pergunta anterior: '{last_user_question}'?"

    if lowered.startswith("e o piso") or lowered == "e o piso?":
        return f"Qual é o piso salarial previsto na mesma convenção coletiva da pergunta anterior: '{last_user_question}'?"

    return f"Reescrevendo a pergunta com contexto da conversa anterior: {q} Referência anterior: {last_user_question}"


@measure("retrieval_time")
def retrieve_context(embedder, store, query, top_k=TOP_K, max_chars=MAX_CHARS):
    query_embedding = embedder.embed_texts([query])
    results = store.search(query_embedding, top_k=top_k)

    telemetry.logs["retrieved_chunks"] = results

    filtered_results = []
    for item in results:
        score = normalize_score(item.get("score", 0))
        if score >= MIN_SCORE:
            filtered_results.append(item)

    if not filtered_results:
        filtered_results = results[:2]

    context_parts = []
    sources = []
    used_chars = 0

    telemetry.metrics["chunks_retrieved"] = len(results)
    telemetry.metrics["chunks_used"] = len(filtered_results)

    top_score = max(
        (normalize_score(item.get("score", 0)) for item in results),
        default=0.0,
    )
    avg_score = (
        sum(normalize_score(item.get("score", 0)) for item in results) / len(results)
        if results else 0.0
    )

    telemetry.metrics["top_score"] = round(top_score, 4)
    telemetry.metrics["avg_score"] = round(avg_score, 4)

    for idx, item in enumerate(filtered_results, start=1):
        trecho = item.get("content", "")[:MAX_CHUNK_CHARS].strip()
        filename = item.get("filename", "arquivo_desconhecido")
        titulo = item.get("titulo", "trecho_sem_titulo")

        block = (
            f"[Fonte {idx}]\n"
            f"Arquivo: {filename}\n"
            f"Título: {titulo}\n"
            f"Trecho: {trecho}\n"
        )

        if used_chars + len(block) > max_chars:
            break

        context_parts.append(block)
        sources.append((filename, titulo))
        used_chars += len(block)

    context = "\n\n".join(context_parts)

    telemetry.logs["context"] = context
    telemetry.logs["sources"] = sources
    telemetry.metrics["context_chars"] = len(context)

    return context, sources


def build_prompt(context, question, conversation_context=""):
    prompt_base = f"""
Você é um assistente jurídico especializado em convenções coletivas de trabalho.

Regras obrigatórias:
- Responda SOMENTE com base no contexto fornecido.
- Não use conhecimento externo.
- Não invente cláusulas, datas, valores ou obrigações.
- Se não houver evidência suficiente, diga:
"Não encontrei informação suficiente no contexto recuperado."
- Responda apenas à pergunta atual.
- Não continue a conversa sozinho.
- Sempre cite as fontes no formato [Fonte X] quando houver evidência.

Formato obrigatório:
Resposta objetiva: ...
Fundamentação: ...
Fontes: [Fonte 1], [Fonte 2]

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

    if conversation_context:
        prompt_base = f"""
Histórico recente da conversa:
{conversation_context}

{prompt_base}
"""

    return prompt_base


def postprocess_answer(answer, sources):
    answer = clean_answer(answer)

    weak_answers = {"sim", "não", "nao", "sim.", "não.", "nao."}

    if not answer:
        return "Não encontrei informação suficiente no contexto recuperado."

    if answer.lower() in weak_answers:
        if sources:
            return (
                f"{answer.capitalize()}, mas a resposta gerada ficou incompleta. "
                "Consulte também as fontes recuperadas para validação."
            )
        return "Não encontrei informação suficiente no contexto recuperado."

    if len(answer) < 40:
        if sources:
            return (
                f"{answer} "
                "A resposta foi curta demais; consulte também as fontes recuperadas para validação."
            )
        return "Não encontrei informação suficiente no contexto recuperado."

    return answer


@measure("generation_time")
def generate_answer(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 220,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()

        telemetry.logs["raw_answer"] = answer
        return answer

    except requests.exceptions.RequestException as e:
        telemetry.metrics["error"] = str(e)
        logging.exception("Erro ao comunicar com o Ollama")

        if "response" in locals():
            logging.error("Resposta bruta do Ollama: %s", response.text)

        raise


def answer_question(question, conversation_context=""):
    telemetry.logs["question"] = question

    rewritten_question = rewrite_question(question, conversation_context)
    telemetry.logs["rewritten_question"] = rewritten_question

    if not is_in_scope(rewritten_question):
        answer = "Esta aplicação foi projetada para responder apenas perguntas sobre convenções coletivas de trabalho."
        telemetry.logs["answer"] = answer
        telemetry.logs["context"] = ""
        telemetry.logs["sources"] = []
        telemetry.metrics["chunks_retrieved"] = 0
        telemetry.metrics["chunks_used"] = 0
        telemetry.metrics["top_score"] = 0
        telemetry.metrics["avg_score"] = 0
        return answer, []

    embedder, store = load_components()
    context, sources = retrieve_context(embedder, store, rewritten_question)

    top_score = telemetry.metrics.get("top_score", 0)

    if not context.strip() or top_score < MIN_ACCEPTABLE_TOP_SCORE:
        answer = "Não encontrei informação suficiente no contexto recuperado."
        telemetry.logs["prompt"] = ""
        telemetry.logs["answer"] = answer
        return answer, sources

    prompt = build_prompt(context, rewritten_question, conversation_context)

    telemetry.logs["prompt"] = prompt
    telemetry.metrics["prompt_chars"] = len(prompt)

    raw_answer = generate_answer(prompt)
    answer = postprocess_answer(raw_answer, sources)

    telemetry.logs["answer"] = answer
    return answer, sources


def main():
    embedder, store = load_components()

    while True:
        question = input("\nDigite sua pergunta (ou 'sair'): ").strip()

        if question.lower() == "sair":
            print("\nEncerrando.")
            break

        telemetry.reset()
        telemetry.logs["question"] = question

        rewritten_question = rewrite_question(question)
        telemetry.logs["rewritten_question"] = rewritten_question

        if not is_in_scope(rewritten_question):
            answer = "Esta aplicação foi projetada para responder apenas perguntas sobre convenções coletivas de trabalho."
            print("\n" + answer)
            continue

        context, sources = retrieve_context(embedder, store, rewritten_question)
        top_score = telemetry.metrics.get("top_score", 0)

        if not context.strip() or top_score < MIN_ACCEPTABLE_TOP_SCORE:
            answer = "Não encontrei informação suficiente no contexto recuperado."
            telemetry.logs["answer"] = answer
            print("\n" + answer)
            continue

        prompt = build_prompt(context, rewritten_question)

        telemetry.logs["prompt"] = prompt
        telemetry.metrics["prompt_chars"] = len(prompt)

        print("\nGerando resposta...\n")
        raw_answer = generate_answer(prompt)
        answer = postprocess_answer(raw_answer, sources)

        telemetry.logs["answer"] = answer

        print(answer)
        print("\nFontes utilizadas:")
        for file, clause in sources:
            print(f"- {file} | {clause}")


if __name__ == "__main__":
    main()