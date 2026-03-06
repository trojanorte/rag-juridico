import requests
import logging

from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
from observability.decorators import measure
from observability.telemetry import telemetry


logging.basicConfig(level=logging.INFO)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"


def load_components():
    logging.info("Inicializando modelo de embeddings...")
    embedder = Embedder()

    logging.info("Carregando índice vetorial...")
    store = FAISSStore(384)
    store.load()

    return embedder, store


@measure("retrieval_time")
def retrieve_context(embedder, store, query, top_k=5, max_chars=3000):
    query_embedding = embedder.embed_texts([query])
    results = store.search(query_embedding, top_k=top_k)

    context = ""
    sources = []

    telemetry.metrics["chunks"] = len(results)

    for item in results:
        trecho = item["content"][:800]
        context += f"\nFonte: {item['filename']} - {item['titulo']}\n{trecho}\n"
        sources.append((item["filename"], item["titulo"]))

        if len(context) >= max_chars:
            break

    telemetry.logs["context"] = context
    telemetry.logs["sources"] = sources

    return context, sources


def build_prompt(context, question):
    return f"""
Você é um assistente jurídico especializado em convenções coletivas de trabalho.

Responda apenas com base no contexto fornecido.
Se a informação não estiver no contexto, diga que não foi encontrada.

Contexto:
{context}

Pergunta:
{question}

Resposta:
""".strip()


@measure("generation_time")
def generate_answer(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()

        telemetry.logs["answer"] = answer

        return answer

    except requests.exceptions.RequestException as e:
        telemetry.metrics["error"] = str(e)
        print("\nErro ao comunicar com o Ollama:")
        print(str(e))
        if 'response' in locals():
            print("Resposta do servidor:", response.text)
        raise


def answer_question(question):
    telemetry.logs["question"] = question

    embedder, store = load_components()
    context, sources = retrieve_context(embedder, store, question)
    prompt = build_prompt(context, question)

    telemetry.logs["prompt"] = prompt

    answer = generate_answer(prompt)
    return answer, sources


def main():
    embedder, store = load_components()

    while True:
        question = input("\nDigite sua pergunta (ou 'sair'): ").strip()

        if question.lower() == "sair":
            print("\nEncerrando.")
            break

        context, sources = retrieve_context(embedder, store, question)
        prompt = build_prompt(context, question)

        print("\nGerando resposta...\n")
        answer = generate_answer(prompt)

        print(answer)
        print("\nFontes utilizadas:")
        for file, clause in sources:
            print(f"- {file} | {clause}")


if __name__ == "__main__":
    main()