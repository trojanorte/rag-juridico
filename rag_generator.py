import logging
import os
import re
from functools import lru_cache

import streamlit as st
from openai import OpenAI

from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
from observability.decorators import measure
from observability.telemetry import telemetry


logging.basicConfig(level=logging.INFO)

MODEL_NAME = "gpt-4.1-mini"

TOP_K = 5
MAX_CHARS = 3000
MAX_CHUNK_CHARS = 900
MIN_SCORE = 0.15
MIN_ACCEPTABLE_TOP_SCORE = 0.10
MAX_OUTPUT_TOKENS = 420


@lru_cache(maxsize=1)
def get_openai_client():
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não encontrada. Defina nos Secrets do Streamlit ou como variável de ambiente."
        )

    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def load_components():
    logging.info("Inicializando modelo de embeddings...")
    embedder = Embedder()

    logging.info("Carregando índice vetorial...")
    store = FAISSStore(768)
    store.load()

    return embedder, store


def normalize_score(score):
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def is_greeting(text: str) -> bool:
    if not text:
        return False

    greetings = [
        "oi", "olá", "ola", "bom dia", "boa tarde", "boa noite",
        "e aí", "ei", "hello", "hi"
    ]

    lowered = text.lower().strip()
    return any(lowered == g or lowered.startswith(g + " ") for g in greetings)


def detect_greeting_type(text: str) -> str | None:
    lowered = (text or "").lower().strip()

    if lowered.startswith("bom dia"):
        return "bom dia"
    if lowered.startswith("boa tarde"):
        return "boa tarde"
    if lowered.startswith("boa noite"):
        return "boa noite"
    if lowered in {"oi", "olá", "ola", "e aí", "ei", "hello", "hi"}:
        return "olá"

    return None


def build_greeting_message(text: str) -> str:
    greeting_type = detect_greeting_type(text)

    if greeting_type == "bom dia":
        saudacao = "Bom dia"
    elif greeting_type == "boa tarde":
        saudacao = "Boa tarde"
    elif greeting_type == "boa noite":
        saudacao = "Boa noite"
    else:
        saudacao = "Olá"

    return (
        f"{saudacao}! Pode me perguntar sobre cláusulas, benefícios, piso, "
        f"vigência, reajuste e outras regras de convenções coletivas."
    )


def is_small_talk(text: str) -> bool:
    if not text:
        return False

    small_talk_patterns = [
        "meu dia foi",
        "tudo bem",
        "como vai",
        "como você está",
        "como vc está",
        "como esta",
        "estou bem",
        "que legal",
        "legal",
        "kkk",
        "haha",
    ]

    lowered = text.lower().strip()
    return any(p in lowered for p in small_talk_patterns)

def is_conversation_question(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower().strip()

    patterns = [
        "qual foi a primeira pergunta",
        "qual foi minha primeira pergunta",
        "qual foi a primira pergunta",
        "qual foi a primiera pergunta",
        "qual foi a pergunta anterior",
        "qual foi minha pergunta anterior",
        "qual foi a última pergunta",
        "qual foi a ultima pergunta",
        "o que eu perguntei antes",
        "o que eu perguntei primeiro",
        "eu perguntei primeiro sobre o que",
        "eu perguntei primiero sobre o que",
        "eu perguntei primiro sobre o que",
        "lembra da pergunta anterior",
        "o que eu falei antes",
    ]

    return any(p in lowered for p in patterns)

def answer_about_conversation(question: str, conversation_context: str) -> str:
    if not conversation_context:
        return "Ainda não há histórico suficiente da conversa para eu responder isso."

    lines = [line.strip() for line in conversation_context.splitlines() if line.strip()]

    perguntas = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("pergunta anterior"):
            partes = line.split(":", 1)
            if len(partes) == 2 and partes[1].strip():
                perguntas.append(partes[1].strip())

    if not perguntas:
        return "Não consegui identificar perguntas anteriores no histórico da conversa."

    lowered_question = (question or "").lower()

    if "primeira" in lowered_question or "primiera" in lowered_question:
        return f'A primeira pergunta que você fez foi: "{perguntas[0]}"'

    if "anterior" in lowered_question or "antes" in lowered_question or "última" in lowered_question or "ultima" in lowered_question:
        return f'A última pergunta antes desta foi: "{perguntas[-1]}"'

    ultimas = ", ".join([f'"{p}"' for p in perguntas[-3:]])
    return f"Identifiquei estas perguntas anteriores: {ultimas}"

def is_in_scope(question: str) -> bool:
    keywords = [
        "acordo", "convenção", "convencao", "cláusula", "clausula",
        "salário", "salario", "vigência", "vigencia", "jornada",
        "sindicato", "sindical", "benefício", "beneficio", "empresa",
        "empregado", "trabalho", "categoria", "categorias", "piso", "vale",
        "adicional", "estabilidade", "férias", "ferias", "horas extras",
        "banco de horas", "creche", "uniforme", "epi", "plr",
        "aviso prévio", "aviso previo", "seguro", "licença", "licenca",
        "reajuste", "desconto", "descontos", "valor", "valores",
        "contribuição", "contribuicao", "cláusulas", "clausulas",
        "hospital", "hospitais", "varejista", "varejo", "plano de saúde",
        "plano de saude", "assistência médica", "assistencia medica",
        "segmento", "setor", "convenções se aplicam", "convencoes se aplicam",
        "aplica", "aplicável", "aplicavel", "obrigação", "obrigacao",
        "deve", "deverá", "devera", "facultativo", "facultativa",
        "obrigatório", "obrigatorio", "auxílio", "auxilio", "cct", "act",
        "vale alimentação", "vale-alimentação", "vale transporte", "vale-transporte"
    ]
    q = (question or "").lower()
    return any(k in q for k in keywords)


def extract_legal_question(text: str) -> str:
    if not text:
        return ""

    triggers = [
        "existe",
        "há",
        "ha",
        "qual",
        "quais",
        "o que",
        "obriga",
        "obrigatório",
        "obrigatorio",
        "facultativo",
        "vigência",
        "vigencia",
        "reajuste",
        "piso",
        "cláusula",
        "clausula",
        "assistência médica",
        "assistencia medica",
        "seguro",
        "plano de saúde",
        "plano de saude",
        "convenção",
        "convencao",
        "categoria",
        "categorias",
        "benefício",
        "beneficio",
        "jornada",
        "vale",
    ]

    lowered = text.lower()
    positions = [lowered.find(t) for t in triggers if lowered.find(t) != -1]

    if not positions:
        return text.strip()

    start = min(positions)
    extracted = text[start:].strip(" ,.-")
    return extracted.strip()


def preprocess_user_input(question: str):
    q = (question or "").strip()

    if not q:
        return {
            "type": "empty",
            "message": "Digite uma pergunta sobre convenções coletivas de trabalho.",
            "question": "",
        }

    if is_greeting(q) and not is_in_scope(q):
        return {
            "type": "greeting",
            "message": build_greeting_message(q),
            "question": "",
        }

    extracted_question = extract_legal_question(q)

    if extracted_question != q and is_in_scope(extracted_question):
        return {
            "type": "mixed",
            "message": None,
            "question": extracted_question,
        }

    if is_small_talk(q) and not is_in_scope(q):
        return {
            "type": "small_talk",
            "message": (
                "Posso te ajudar com perguntas sobre convenções coletivas de trabalho. "
                "Mande a cláusula, benefício ou obrigação que você quer verificar."
            ),
            "question": "",
        }

    return {
        "type": "normal",
        "message": None,
        "question": q,
    }


def clean_answer(answer: str) -> str:
    answer = (answer or "").strip()

    stop_markers = [
        "\nPergunta:",
        "\n\nPergunta:",
        "\nResposta:",
        "\n\nResposta:",
        "\nUsuário:",
        "\nUsuario:",
        "\nUser:",
        "\nAssistant:",
    ]

    cleaned = answer
    for marker in stop_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0].strip()

    return cleaned


def needs_rewrite(question: str) -> bool:
    q = (question or "").lower().strip()

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
        "e no caso",
        "e nesse caso",
        "e nessa",
        "e nesse",
        "isso",
        "essa",
        "esse",
        "ele",
        "ela",
        "mesma convenção",
        "mesmo acordo",
        "o mesmo",
        "a mesma",
        "tem ",
        "tem o ",
        "tem a ",
    ]

    return any(q.startswith(prefix) for prefix in continuation_starts)


def extract_last_user_question(conversation_context: str):
    if not conversation_context:
        return None

    history_lines = [line.strip() for line in conversation_context.splitlines() if line.strip()]

    for line in reversed(history_lines):
        lowered = line.lower()
        if lowered.startswith("pergunta anterior"):
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()

        if lowered.startswith("usuário:") or lowered.startswith("usuario:"):
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()

    return None


def rewrite_question(question: str, conversation_context: str = "") -> str:
    q = (question or "").strip()

    if not conversation_context:
        return q

    last_user_question = extract_last_user_question(conversation_context)
    if not last_user_question:
        return q

    lowered = q.lower()

    explicit_patterns = {
        ("e o reajuste", "reajuste"): "Qual é o reajuste salarial previsto na mesma convenção coletiva da pergunta anterior?",
        ("e o seguro", "seguro"): "Existe cláusula sobre seguro na mesma convenção coletiva da pergunta anterior?",
        ("e a vigência", "vigência", "vigencia"): "Qual é a vigência da mesma convenção coletiva da pergunta anterior?",
        ("e o piso", "piso"): "Qual é o piso salarial previsto na mesma convenção coletiva da pergunta anterior?",
        ("e o plano de saúde", "plano de saúde", "plano de saude"): "Há previsão de plano de saúde na mesma convenção coletiva da pergunta anterior?",
        ("e a assistência médica", "assistência médica", "assistencia medica"): "Há previsão de assistência médica na mesma convenção coletiva da pergunta anterior?",
        ("e o adicional noturno", "adicional noturno"): "Existe cláusula sobre adicional noturno na mesma convenção coletiva da pergunta anterior?",
        ("e a insalubridade", "insalubridade"): "Existe cláusula sobre adicional de insalubridade na mesma convenção coletiva da pergunta anterior?",
        ("e a periculosidade", "periculosidade"): "Existe cláusula sobre adicional de periculosidade na mesma convenção coletiva da pergunta anterior?",
        ("e as horas extras", "horas extras"): "Como a mesma convenção coletiva da pergunta anterior trata as horas extras?",
        ("e o vale alimentação", "vale alimentação", "vale-alimentação"): "Há previsão de vale-alimentação na mesma convenção coletiva da pergunta anterior?",
        ("e o vale transporte", "vale transporte", "vale-transporte"): "Há previsão de vale-transporte na mesma convenção coletiva da pergunta anterior?",
        ("e o auxílio alimentação", "auxílio alimentação", "auxilio alimentação"): "Há previsão de auxílio-alimentação na mesma convenção coletiva da pergunta anterior?",
        ("e a jornada", "jornada"): "Como a mesma convenção coletiva da pergunta anterior trata a jornada de trabalho?",
        ("tem vale transporte", "vale transporte", "vale-transporte"): "Há previsão de vale-transporte na mesma convenção coletiva da pergunta anterior?",
        ("tem vale alimentação",): "Há previsão de vale-alimentação na mesma convenção coletiva da pergunta anterior?",
    }

    for triggers, rewritten in explicit_patterns.items():
        if any(lowered == t or lowered.startswith(t + " ") or lowered == f"{t}?" for t in triggers):
            return rewritten

    if needs_rewrite(q):
        return f"{q} considerando o contexto da pergunta anterior: {last_user_question}"

    if len(q.split()) <= 6:
        return f"{q} considerando o contexto da pergunta anterior: {last_user_question}"

    return q


@measure("rewrite_time")
def rewrite_question_with_llm(question: str, conversation_context: str = "") -> str:
    q = (question or "").strip()

    if not conversation_context:
        return q

    client = get_openai_client()

    prompt = f"""
Você receberá o histórico recente de uma conversa e a pergunta atual do usuário.
Reescreva a pergunta atual para que ela fique independente, completa e adequada para busca semântica em convenções coletivas de trabalho.
Se a pergunta atual for claramente independente, mantenha o sentido original.
Não responda a pergunta.
Não invente fatos.
Apenas devolva a pergunta reescrita.

Histórico:
{conversation_context}

Pergunta atual:
{q}

Pergunta reescrita:
""".strip()

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            temperature=0,
            max_output_tokens=120,
        )

        rewritten = getattr(response, "output_text", "") or ""
        rewritten = rewritten.strip()

        return rewritten or rewrite_question(q, conversation_context)

    except Exception:
        logging.exception("Falha ao reescrever pergunta com LLM; usando fallback local.")
        return rewrite_question(q, conversation_context)


def trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{trimmed}..."


@measure("retrieval_time")
def retrieve_context(embedder, store, query, top_k=TOP_K, max_chars=MAX_CHARS):
    query_embedding = embedder.embed_query(query)
    results = store.search(query_embedding, top_k=top_k) or []

    telemetry.logs["retrieved_chunks"] = results

    filtered_results = []
    for item in results:
        score = normalize_score(item.get("score", 0))
        if score >= MIN_SCORE:
            filtered_results.append(item)

    if not filtered_results:
        filtered_results = results[:3]

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
        trecho = trim_text(item.get("content", ""), MAX_CHUNK_CHARS)
        filename = item.get("filename", "arquivo_desconhecido")
        titulo = item.get("titulo", "trecho_sem_titulo")
        score = normalize_score(item.get("score", 0))

        if not trecho:
            continue

        block = (
            f"[Fonte {idx}]\n"
            f"Documento: {filename}\n"
            f"Cláusula/Título: {titulo}\n"
            f"Relevância: {score:.3f}\n"
            f"Texto:\n{trecho}\n"
        )

        if used_chars + len(block) > max_chars:
            break

        context_parts.append(block)
        sources.append({
            "id": idx,
            "label": f"Fonte {idx}",
            "arquivo": filename,
            "titulo": titulo,
            "score": round(score, 4),
        })
        used_chars += len(block)

    context = "\n\n".join(context_parts)

    telemetry.logs["context"] = context
    telemetry.logs["sources"] = sources
    telemetry.metrics["context_chars"] = len(context)

    return context, sources


def build_prompt(context, question):
    return f"""
Você é um assistente jurídico especializado em convenções coletivas de trabalho.

Sua tarefa é responder APENAS com base no contexto recuperado.

Regras obrigatórias:
- Não use conhecimento externo.
- Não invente cláusulas, datas, valores, percentuais, categorias ou obrigações.
- Não reformule a pergunta do usuário.
- Não mencione histórico da conversa.
- Não escreva expressões como "considerando a continuidade da conversa" ou "no contexto da pergunta anterior".
- Responda diretamente à pergunta do usuário com base apenas nos trechos recuperados.
- Quando o contexto trouxer evidência parcial, explique exatamente o que foi encontrado e o que não foi encontrado.
- Não responda apenas "não encontrei" se houver informação parcialmente útil.
- Se houver indicação de facultatividade, manutenção de benefício já existente ou ausência de obrigação expressa, destaque isso claramente.
- Use linguagem clara, objetiva e jurídica.
- Não continue a conversa por conta própria.
- Cite somente as fontes efetivamente utilizadas, no formato [Fonte X].
- Não mencione fontes que não estejam no contexto.
- Não afirme que algo é obrigatório sem evidência textual no contexto.

Formato desejado:
Resposta curta: responda objetivamente em 1 a 3 frases.
Explicação: explique com base nos trechos recuperados.
Fontes consultadas: liste apenas as fontes usadas.

Se a evidência for insuficiente, use formulação como:
"Os trechos recuperados não permitem afirmar com segurança..."
ou
"Não foi identificada, nos trechos recuperados, cláusula expressa que..."

Contexto recuperado:
{context}

Pergunta do usuário:
{question}

Resposta:
""".strip()


def extract_used_source_labels(answer: str):
    if not answer:
        return []
    return sorted(set(re.findall(r"\[Fonte\s+\d+\]", answer)))


def append_sources_if_missing(answer: str, sources: list[dict]) -> str:
    if not sources:
        return answer

    used_labels = extract_used_source_labels(answer)

    if used_labels:
        source_line = "Fontes consultadas: " + ", ".join(used_labels)
        if "Fontes consultadas:" not in answer:
            return f"{answer}\n\n{source_line}"
        return answer

    fallback_labels = [f"[{s['label']}]" for s in sources[:3]]
    source_line = "Fontes consultadas: " + ", ".join(fallback_labels)

    if "Fontes consultadas:" not in answer:
        return f"{answer}\n\n{source_line}"

    return answer


def postprocess_answer(answer, sources):
    answer = clean_answer(answer)

    if not answer:
        return "Não encontrei informação suficiente no contexto recuperado."

    weak_answers = {"sim", "não", "nao", "sim.", "não.", "nao."}
    if answer.lower() in weak_answers:
        return "A resposta gerada ficou incompleta com base no contexto recuperado."

    if len(answer) < 60:
        answer = (
            f"{answer}\n\n"
            "Observação: a resposta foi curta e pode não refletir toda a nuance dos trechos recuperados."
        )

    overly_generic_patterns = [
        "não encontrei informação suficiente no contexto recuperado",
        "não foi possível identificar",
    ]

    lower_answer = answer.lower()
    if any(p in lower_answer for p in overly_generic_patterns):
        if sources:
            answer += (
                "\n\nObservação: verifique também os trechos recuperados, "
                "pois pode haver evidência parcial ou indireta nas fontes."
            )

    answer = append_sources_if_missing(answer, sources)
    return answer


@measure("generation_time")
def generate_answer(prompt):
    client = get_openai_client()

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            temperature=0.1,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        answer = getattr(response, "output_text", "") or ""
        answer = answer.strip()

        telemetry.logs["raw_answer"] = answer
        return answer

    except Exception as e:
        telemetry.metrics["error"] = str(e)
        logging.exception("Erro ao comunicar com a OpenAI")
        raise


def format_sources_for_display(sources):
    if not sources:
        return []

    formatted = []
    for src in sources:
        formatted.append(f"{src['arquivo']} | {src['titulo']} | score={src['score']}")
    return formatted


def out_of_scope_answer() -> str:
    return (
        "Esta base é especializada em convenções coletivas de trabalho. "
        "A pergunta enviada não parece relacionada a esse escopo. "
        "Posso ajudar com cláusulas, piso salarial, vigência, benefícios, reajuste, jornada e obrigações previstas em convenções."
    )


def reset_empty_metrics():
    telemetry.logs["context"] = ""
    telemetry.logs["sources"] = []
    telemetry.metrics["chunks_retrieved"] = 0
    telemetry.metrics["chunks_used"] = 0
    telemetry.metrics["top_score"] = 0
    telemetry.metrics["avg_score"] = 0


def answer_question(question, conversation_context=""):
    telemetry.reset()
    telemetry.logs["question"] = question

    preprocessed = preprocess_user_input(question)

    if is_conversation_question(question):
        answer = answer_about_conversation(question, conversation_context)
        telemetry.logs["answer"] = answer
        reset_empty_metrics()
        return answer, []

    if preprocessed["type"] in {"empty", "greeting", "small_talk"}:
        answer = preprocessed["message"]
        telemetry.logs["answer"] = answer
        reset_empty_metrics()
        return answer, []

    effective_question = preprocessed["question"]

    if not is_in_scope(effective_question) and not needs_rewrite(effective_question):
        answer = out_of_scope_answer()
        telemetry.logs["answer"] = answer
        reset_empty_metrics()
        return answer, []

    rewritten_question = rewrite_question_with_llm(
        question=effective_question,
        conversation_context=conversation_context,
    )
    telemetry.logs["rewritten_question"] = rewritten_question

    if not is_in_scope(rewritten_question):
        answer = out_of_scope_answer()
        telemetry.logs["answer"] = answer
        reset_empty_metrics()
        return answer, []

    embedder, store = load_components()
    context, sources = retrieve_context(embedder, store, rewritten_question)

    top_score = telemetry.metrics.get("top_score", 0.0)

    if not context.strip():
        answer = "Não encontrei trechos relevantes suficientes para responder com segurança."
        telemetry.logs["prompt"] = ""
        telemetry.logs["answer"] = answer
        return answer, sources

    if top_score < MIN_ACCEPTABLE_TOP_SCORE and len(sources) == 0:
        answer = "Não encontrei trechos relevantes suficientes para responder com segurança."
        telemetry.logs["prompt"] = ""
        telemetry.logs["answer"] = answer
        return answer, sources

    prompt = build_prompt(context, rewritten_question)

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

        preprocessed = preprocess_user_input(question)

        if is_conversation_question(question):
            answer = "No modo terminal, perguntas sobre histórico só funcionam se você passar conversation_context."
            telemetry.logs["answer"] = answer
            print("\n" + answer)
            continue

        if preprocessed["type"] in {"empty", "greeting", "small_talk"}:
            print("\n" + preprocessed["message"])
            continue

        effective_question = preprocessed["question"]

        if not is_in_scope(effective_question) and not needs_rewrite(effective_question):
            answer = out_of_scope_answer()
            telemetry.logs["answer"] = answer
            print("\n" + answer)
            continue

        rewritten_question = rewrite_question_with_llm(
            question=effective_question,
            conversation_context="",
        )
        telemetry.logs["rewritten_question"] = rewritten_question

        if not is_in_scope(rewritten_question):
            answer = out_of_scope_answer()
            telemetry.logs["answer"] = answer
            print("\n" + answer)
            continue

        context, sources = retrieve_context(embedder, store, rewritten_question)
        top_score = telemetry.metrics.get("top_score", 0.0)

        if not context.strip():
            answer = "Não encontrei trechos relevantes suficientes para responder com segurança."
            telemetry.logs["answer"] = answer
            print("\n" + answer)
            continue

        if top_score < MIN_ACCEPTABLE_TOP_SCORE and len(sources) == 0:
            answer = "Não encontrei trechos relevantes suficientes para responder com segurança."
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
        print("\nFontes recuperadas:")
        for line in format_sources_for_display(sources):
            print(f"- {line}")


if __name__ == "__main__":
    main()