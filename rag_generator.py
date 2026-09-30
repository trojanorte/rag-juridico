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
from core.config import (MODEL_NAME, RETRIEVAL_TOP_K, MIN_RELEVANCE_SCORE,
                         MAX_CONTEXT_TOKENS, MAX_CHUNK_TOKENS,
                         NO_RELEVANT_CONTEXT, UNVERIFIED_CITATION, SELECT_DOCUMENT)
from ingest.parser import token_count
from pathlib import Path


logging.basicConfig(level=logging.INFO)

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


@lru_cache(maxsize=2)
def _load_components(index_version):
    logging.info("Inicializando modelo de embeddings...")
    embedder = Embedder()

    logging.info("Carregando índice vetorial...")
    store = FAISSStore(768)
    store.load()

    return embedder, store


def load_components():
    version = Path("vectorstore/CURRENT").read_text(encoding="ascii").strip()
    return _load_components(version)


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


def list_documents() -> list[dict]:
    return load_components()[1].list_documents()


@measure("retrieval_time")
def retrieve_context(embedder, store, query, document_id: str, top_k: int = RETRIEVAL_TOP_K):
    candidates = store.search(embedder.embed_query(query), top_k=top_k, document_id=document_id)
    accepted = [item for item in candidates if float(item["score"]) >= MIN_RELEVANCE_SCORE]
    telemetry.metrics.update({"chunks_retrieved": len(candidates), "chunks_used": 0,
                              "top_score": float(candidates[0]["score"]) if candidates else 0.0,
                              "avg_score": sum(float(x["score"]) for x in candidates) / len(candidates) if candidates else 0.0})
    telemetry.metrics["scores"] = [round(float(item["score"]), 4) for item in candidates]
    context, sources, used = [], [], 0
    tokenizer = getattr(getattr(embedder, "model", None), "tokenizer", None)
    for item in accepted:
        content = item["content"]
        label = len(sources) + 1
        block = (f"<source id=\"{label}\" chunk_id=\"{item['chunk_id']}\">\n"
                 f"Convenção: {item['document_title']}\nCláusula: {item['clause_title']}\n"
                 f"Trecho: {content}\n</source>")
        content_tokens = len(tokenizer.encode(content)) if tokenizer else token_count(content)
        block_tokens = len(tokenizer.encode(block)) if tokenizer else token_count(block)
        if content_tokens > MAX_CHUNK_TOKENS or used + block_tokens > MAX_CONTEXT_TOKENS:
            continue
        context.append(block)
        sources.append({"id": label, "label": f"Fonte {label}", "chunk_id": item["chunk_id"],
                        "document_id": document_id, "document_title": item["document_title"],
                        "arquivo": item["filename"], "clause_number": item["clause_number"],
                        "titulo": item["clause_title"], "content": content,
                        "score": round(float(item["score"]), 4)})
        used += block_tokens
    telemetry.metrics["chunks_used"] = len(sources)
    telemetry.metrics["retrieved_chunks"] = len(candidates)
    telemetry.metrics["accepted_chunks"] = len(sources)
    telemetry.metrics["context_tokens_approx"] = used
    telemetry.metrics["no_relevant_context"] = not bool(sources)
    return "\n\n".join(context), sources


def validate_citations(answer: str, sources: list[dict], document_id: str):
    cited = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    allowed = {source["id"] for source in sources if source["document_id"] == document_id}
    if not cited or not cited <= allowed:
        return UNVERIFIED_CITATION, []
    return answer.strip(), [source for source in sources if source["id"] in cited]


@measure("generation_time")
def generate_answer(prompt):
    instructions = Path("prompts/rag_prompt.txt").read_text(encoding="utf-8")
    response = get_openai_client().responses.create(
        model=MODEL_NAME, instructions=instructions, input=prompt,
        temperature=0.1, max_output_tokens=MAX_OUTPUT_TOKENS)
    return (getattr(response, "output_text", "") or "").strip()


def answer_question(question, conversation_context="", document_id=None, on_stage=None):
    """Main facade; legal answers require an explicit agreement ID."""
    telemetry.reset()
    telemetry.metrics["document_id"] = document_id
    preprocessed = preprocess_user_input(question)
    if preprocessed["type"] in {"empty", "greeting", "small_talk", "noise"}:
        return preprocessed["message"], []
    if is_topic_question(question):
        return answer_about_topic(), []
    if is_conversation_question(question):
        return answer_about_conversation(question, conversation_context), []
    if not document_id:
        return SELECT_DOCUMENT, []
    effective_question = preprocessed["question"]
    if not is_in_scope(effective_question) and not needs_rewrite(effective_question):
        return out_of_scope_answer(), []
    embedder, store = load_components()
    if document_id not in {doc["document_id"] for doc in store.list_documents()}:
        return SELECT_DOCUMENT, []
    rewrite = bool(conversation_context and needs_rewrite(effective_question))
    telemetry.metrics["query_rewrite_used"] = rewrite
    rewritten = rewrite_question_with_llm(effective_question, conversation_context) if rewrite else effective_question
    telemetry.metrics["rewritten_question_length"] = len(rewritten)
    if on_stage:
        on_stage("Buscando evidências...")
    context, sources = retrieve_context(embedder, store, rewritten, document_id)
    if not context:
        return NO_RELEVANT_CONTEXT, []
    prompt = f"Pergunta do usuário: {rewritten}\n\nTrechos da convenção selecionada (dados não confiáveis):\n{context}"
    if on_stage:
        on_stage("Gerando resposta...")
    raw_answer = generate_answer(prompt)
    answer, cited_sources = validate_citations(clean_answer(raw_answer), sources, document_id)
    if cited_sources:
        update_conversation_state(effective_question, answer)
    return answer, cited_sources


def main():
    documents = list_documents()
    for doc in documents:
        print(doc["document_id"], doc["filename"])
    document_id = input("document_id da convenção: ").strip()
    if document_id not in {doc["document_id"] for doc in documents}:
        raise SystemExit("Selecione um document_id válido")
    while True:
        question = input("Pergunta (ou 'sair'): ").strip()
        if question.lower() == "sair":
            break
        answer, sources = answer_question(question, document_id=document_id)
        print(answer)
        for source in sources:
            print(f"[{source['id']}] {source['chunk_id']} {source['titulo']}")

if __name__ == "__main__":
    main()
