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
MAX_CHARS = 2500
MAX_CHUNK_CHARS = 700
MIN_SCORE = 0.20


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
        score = normalize_score(item.get("score", 0))

        block = (
            f"[Fonte {idx}]\n"
            f"Arquivo: {filename}\n"
            f"Título: {titulo}\n"
            f"Score: {score:.4f}\n"
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


def build_prompt(context, question):
    return f"""
Você é um assistente jurídico especializado em convenções coletivas de trabalho.

Regras obrigatórias:
- Responda SOMENTE com base no contexto fornecido.
- Não use conhecimento externo.
- Não faça inferências além do que estiver escrito.
- Se o contexto não contiver evidência suficiente para responder, diga exatamente:
  "Não encontrei informação suficiente no contexto recuperado."
- Não invente cláusulas, valores, datas ou obrigações.
- Sempre cite as fontes no formato [Fonte X] quando houver evidência.

Formato obrigatório da resposta:
1. Resposta objetiva à pergunta
2. Explicação curta baseada no contexto
3. Fontes utilizadas no formato [Fonte X]

Contexto:
{context}

Pergunta:
{question}

Resposta:
""".strip()


def postprocess_answer(answer, sources):
    answer = (answer or "").strip()

    weak_answers = {"sim", "não", "nao", "sim.", "não.", "nao."}

    if not answer:
        return "Não encontrei informação suficiente no contexto recuperado."

    if answer.lower() in weak_answers:
        if sources:
            return (
                f"{answer.capitalize()}, mas a resposta gerada ficou incompleta. "
                "Revise os trechos recuperados nas fontes apresentadas."
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
            "num_predict": 300,
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
        raise


def answer_question(question):
    telemetry.logs["question"] = question

    embedder, store = load_components()
    context, sources = retrieve_context(embedder, store, question)
    prompt = build_prompt(context, question)

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

        context, sources = retrieve_context(embedder, store, question)
        prompt = build_prompt(context, question)

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