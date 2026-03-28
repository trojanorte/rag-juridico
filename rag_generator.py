import logging
import os
import re
import unicodedata
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


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_gibberish(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False

    if len(t) >= 8 and " " not in t:
        vowels = sum(1 for c in t if c in "aeiou")
        if vowels <= 1:
            return True

    return False


def is_greeting(text: str) -> bool:
    lowered = normalize_text(text)

    greetings = [
        "oi", "ola", "bom dia", "boa tarde", "boa noite",
        "e ai", "ei", "hello", "hi"
    ]

    return any(lowered == g or lowered.startswith(g + " ") for g in greetings)


def detect_greeting_type(text: str) -> str | None:
    lowered = normalize_text(text)

    if lowered.startswith("bom dia"):
        return "bom dia"
    if lowered.startswith("boa tarde"):
        return "boa tarde"
    if lowered.startswith("boa noite"):
        return "boa noite"
    if lowered in {"oi", "ola", "e ai", "ei", "hello", "hi"}:
        return "ola"

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
    lowered = normalize_text(text)

    small_talk_patterns = [
        "meu dia foi",
        "tudo bem",
        "como vai",
        "como voce esta",
        "como vc esta",
        "estou bem",
        "que legal",
        "legal",
        "kkk",
        "haha",
    ]

    return any(p in lowered for p in small_talk_patterns)


def is_topic_question(text: str) -> bool:
    lowered = normalize_text(text)

    patterns = [
        "estamos falando sobre o que",
        "sobre o que estamos falando",
        "qual o assunto",
        "qual e o assunto",
        "qual tema da conversa",
        "qual o tema",
        "qual e o tema",
        "sobre o que e a conversa",
    ]
    return any(p in lowered for p in patterns)


def is_conversation_question(text: str) -> bool:
    lowered = normalize_text(text)

    patterns = [
        "qual foi a primeira pergunta",
        "qual foi minha primeira pergunta",
        "qual a primeira pergunta",
        "qual a primira pergunta",
        "qual a primiera pergunta",
        "qual foi a primira pergunta",
        "qual foi a primiera pergunta",
        "qual foi a primeira pergunta que eu fiz",
        "qual foi a pergunta anterior",
        "qual minha pergunta anterior",
        "qual foi minha pergunta anterior",
        "qual foi a ultima pergunta",
        "qual foi a última pergunta",
        "qual a ultima pergunta",
        "qual a última pergunta",
        "o que eu perguntei antes",
        "o que eu perguntei primeiro",
        "eu perguntei primeiro sobre o que",
        "eu perguntei primiero sobre o que",
        "eu perguntei primiro sobre o que",
        "lembra da pergunta anterior",
        "o que eu falei antes",
    ]

    return any(p in lowered for p in patterns)


def get_conversation_state():
    if "conversation_state" not in st.session_state:
        st.session_state["conversation_state"] = {
            "first_legal_question": None,
            "last_legal_question": None,
            "current_topic": None,
            "recent_legal_questions": [],
        }
    return st.session_state["conversation_state"]


def infer_topic(question_processed: str, answer: str = "") -> str:
    q = normalize_text(question_processed)
    a = normalize_text(answer)

    text = f"{q} {a}"

    if any(k in text for k in ["vale alimentacao", "vale refeicao", "cesta", "auxilio alimentacao"]):
        return "benefícios da convenção, com foco em vale alimentação e auxílio alimentação"

    if "vale transporte" in text:
        return "benefícios da convenção, com foco em vale transporte"

    if "reajuste" in text or "data base" in text:
        return "condições econômicas da convenção, com foco em reajuste salarial"

    if "jornada" in text or "horas extras" in text or "banco de horas" in text:
        return "jornada de trabalho e regras de tempo"

    if "insalubridade" in text or "periculosidade" in text or "adicional noturno" in text:
        return "adicionais trabalhistas previstos na convenção"

    if "seguro" in text or "plano de saude" in text or "assistencia medica" in text:
        return "benefícios e proteção do empregado"

    return "cláusulas e benefícios da convenção coletiva"


def update_conversation_state(question_processed: str, answer: str = "") -> None:
    if not question_processed:
        return

    state = get_conversation_state()

    if state["first_legal_question"] is None:
        state["first_legal_question"] = question_processed

    state["last_legal_question"] = question_processed
    state["recent_legal_questions"].append(question_processed)
    state["recent_legal_questions"] = state["recent_legal_questions"][-5:]
    state["current_topic"] = infer_topic(question_processed, answer)


def answer_about_conversation(question: str, conversation_context: str) -> str:
    state = get_conversation_state()
    lowered_question = normalize_text(question)

    if "primeira" in lowered_question or "primira" in lowered_question or "primiera" in lowered_question or "primeiro" in lowered_question or "primiero" in lowered_question:
        first_q = state.get("first_legal_question")
        if first_q:
            return f'A primeira pergunta jurídica que você fez foi: "{first_q}"'
        return "Ainda não identifiquei uma pergunta jurídica anterior."

    if "ultima" in lowered_question or "última" in lowered_question or "anterior" in lowered_question or "antes" in lowered_question:
        last_q = state.get("last_legal_question")
        if last_q:
            return f'A última pergunta jurídica antes desta foi: "{last_q}"'
        return "Ainda não identifiquei uma pergunta jurídica anterior."

    recent = state.get("recent_legal_questions", [])
    if recent:
        ultimas = ", ".join([f'"{p}"' for p in recent[-3:]])
        return f"Identifiquei estas perguntas jurídicas recentes: {ultimas}"

    return "Ainda não há histórico suficiente da conversa para eu responder isso."


def answer_about_topic() -> str:
    state = get_conversation_state()
    topic = state.get("current_topic")

    if topic:
        return f"Estamos falando sobre {topic}."

    last_q = state.get("last_legal_question")
    if last_q:
        return f'Até aqui, sua última pergunta jurídica foi: "{last_q}"'

    return "Ainda não consegui consolidar um assunto principal da conversa."


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
        "inss", "vale alimentação", "vale alimentacao", "vale-transporte", "vale transporte",
    ]
    q = normalize_text(question)
    return any(k in q for k in keywords)


def extract_legal_question(text: str) -> str:
    if not text:
        return ""

    normalized = normalize_text(text)

    triggers = [
        "existe", "ha", "qual", "quais", "o que", "obriga", "obrigatorio",
        "facultativo", "vigencia", "reajuste", "piso", "clausula",
        "assistencia medica", "seguro", "plano de saude", "convencao",
        "categoria", "categorias", "beneficio", "jornada", "vale", "inss"
    ]

    positions = [normalized.find(t) for t in triggers if normalized.find(t) != -1]

    if not positions:
        return text.strip()

    start = min(positions)

    original_lower = text.lower()
    if start < len(original_lower):
        extracted = text[start:].strip(" ,.-")
        return extracted.strip()

    return text.strip()


def extract_legal_fragment(text: str) -> str:
    if not text:
        return ""

    original = text.strip()
    normalized = normalize_text(original)

    fragments = re.split(r"[,.;/]| e ", normalized)

    legal_candidates = []
    legal_keywords = [
        "vale", "reajuste", "inss", "jornada", "piso", "seguro",
        "clausula", "beneficio", "vigencia", "transporte", "alimentacao",
        "horas extras", "insalubridade", "periculosidade", "plano de saude",
    ]

    for frag in fragments:
        frag = frag.strip()
        if any(k in frag for k in legal_keywords):
            legal_candidates.append(frag)

    if legal_candidates:
        return legal_candidates[-1]

    return extract_legal_question(original)


def preprocess_user_input(question: str):
    q = (question or "").strip()

    if not q:
        return {
            "type": "empty",
            "message": "Digite uma pergunta sobre convenções coletivas de trabalho.",
            "question": "",
        }

    if is_gibberish(q):
        return {
            "type": "noise",
            "message": (
                "Não consegui entender sua mensagem. "
                "Pode reformular a pergunta sobre a convenção coletiva?"
            ),
            "question": "",
        }

    if is_greeting(q) and not is_in_scope(q):
        return {
            "type": "greeting",
            "message": build_greeting_message(q),
            "question": "",
        }

    extracted_fragment = extract_legal_fragment(q)
    extracted_question = extract_legal_question(q)

    best_candidate = extracted_fragment if is_in_scope(extracted_fragment) else extracted_question

    if best_candidate != q and is_in_scope(best_candidate):
        return {
            "type": "mixed",
            "message": None,
            "question": best_candidate,
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
    q = normalize_text(question)

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
        "mesma convencao",
        "mesmo acordo",
        "o mesmo",
        "a mesma",
        "tem ",
        "tem o ",
        "tem a ",
    ]

    short_followup = len(q.split()) <= 4 and any(
        k in q for k in [
            "reajuste", "inss", "vale", "seguro", "jornada",
            "piso", "vigencia", "transporte", "alimentacao"
        ]
    )

    return any(q.startswith(prefix) for prefix in continuation_starts) or short_followup


def extract_last_user_question(conversation_context: str):
    if not conversation_context:
        return None

    history_lines = [line.strip() for line in conversation_context.splitlines() if line.strip()]

    for line in reversed(history_lines):
        lowered = normalize_text(line)
        if lowered.startswith("pergunta anterior"):
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

    lowered = normalize_text(q)

    explicit_patterns = {
        ("reajuste", "e o reajuste"): "Qual é o reajuste salarial previsto na mesma convenção coletiva da pergunta anterior?",
        ("seguro", "e o seguro"): "Existe cláusula sobre seguro na mesma convenção coletiva da pergunta anterior?",
        ("vigencia", "e a vigencia"): "Qual é a vigência da mesma convenção coletiva da pergunta anterior?",
        ("piso", "e o piso"): "Qual é o piso salarial previsto na mesma convenção coletiva da pergunta anterior?",
        ("plano de saude", "e o plano de saude"): "Há previsão de plano de saúde na mesma convenção coletiva da pergunta anterior?",
        ("assistencia medica", "e a assistencia medica"): "Há previsão de assistência médica na mesma convenção coletiva da pergunta anterior?",
        ("adicional noturno", "e o adicional noturno"): "Existe cláusula sobre adicional noturno na mesma convenção coletiva da pergunta anterior?",
        ("insalubridade", "e a insalubridade"): "Existe cláusula sobre adicional de insalubridade na mesma convenção coletiva da pergunta anterior?",
        ("periculosidade", "e a periculosidade"): "Existe cláusula sobre adicional de periculosidade na mesma convenção coletiva da pergunta anterior?",
        ("horas extras", "e as horas extras"): "Como a mesma convenção coletiva da pergunta anterior trata as horas extras?",
        ("vale alimentacao", "vale alimentação", "vale alimentacao", "e o vale alimentacao"): "Há previsão de vale-alimentação na mesma convenção coletiva da pergunta anterior?",
        ("vale transporte", "vale-transporte", "e o vale transporte"): "Há previsão de vale-transporte na mesma convenção coletiva da pergunta anterior?",
        ("auxilio alimentacao",): "Há previsão de auxílio-alimentação na mesma convenção coletiva da pergunta anterior?",
        ("jornada", "e a jornada"): "Como a mesma convenção coletiva da pergunta anterior trata a jornada de trabalho?",
        ("inss", "e o inss"): "Há alguma previsão relacionada a INSS ou descontos previdenciários na mesma convenção coletiva da pergunta anterior?",
    }

    for triggers, rewritten in explicit_patterns.items():
        if any(lowered == t or lowered.startswith(t + " ") or lowered == f"{t}?" for t in triggers):
            return rewritten

    if needs_rewrite(q):
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
Se a pergunta atual já estiver clara sozinha, preserve o sentido original.
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

    if is_topic_question(question):
        answer = answer_about_topic()
        telemetry.logs["answer"] = answer
        reset_empty_metrics()
        return answer, []

    if is_conversation_question(question):
        answer = answer_about_conversation(question, conversation_context)
        telemetry.logs["answer"] = answer
        reset_empty_metrics()
        return answer, []

    if preprocessed["type"] in {"empty", "greeting", "small_talk", "noise"}:
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

    update_conversation_state(effective_question, answer)

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

        if is_topic_question(question):
            print("\n" + answer_about_topic())
            continue

        if is_conversation_question(question):
            print("\nNo modo terminal, perguntas sobre histórico dependem do estado da sessão Streamlit.")
            continue

        preprocessed = preprocess_user_input(question)

        if preprocessed["type"] in {"empty", "greeting", "small_talk", "noise"}:
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

        update_conversation_state(effective_question, answer)

        telemetry.logs["answer"] = answer

        print(answer)
        print("\nFontes recuperadas:")
        for line in format_sources_for_display(sources):
            print(f"- {line}")


if __name__ == "__main__":
    main()