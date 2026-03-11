import uuid
import streamlit as st

from rag_generator import answer_question
from observability.telemetry import telemetry
from observability.debug_store import init_db, save_query_log
from observability.prom_metrics import (
    start_metrics_server,
    rag_requests_total,
    rag_errors_total,
    rag_total_time_seconds,
    rag_retrieval_time_seconds,
    rag_generation_time_seconds,
    rag_chunks_retrieved,
    rag_chunks_used,
    rag_top_score,
    rag_avg_score,
)


PROJECT_NAME = "LexRAG — Assistente de Convenções Coletivas"

st.set_page_config(
    page_title=PROJECT_NAME,
    layout="wide",
    page_icon="📚"
)


SUGGESTED_QUESTIONS = [
    "Qual é a vigência do acordo coletivo?",
    "Existe cláusula de seguro obrigatório?",
    "Qual é o reajuste salarial?"
]


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Sobre o sistema")
        st.write(
            """
            O **LexRAG** utiliza **RAG (Retrieval-Augmented Generation)**
            para consultar **convenções coletivas de trabalho** e responder
            perguntas com base nos documentos indexados.
            """
        )

        st.divider()

        st.subheader("Tipos de perguntas")
        st.markdown(
            """
            Exemplos:

            • Qual é a vigência do acordo coletivo  
            • Qual o reajuste salarial previsto  
            • Existe cláusula de seguro obrigatório  
            • Qual o valor da contribuição sindical  
            • Quais descontos são previstos  
            """
        )

        st.divider()

        st.subheader("Status do sistema")
        st.success("RAG ativo")
        st.write("Embeddings + FAISS + Ollama")

        st.divider()

        if st.button("🗑 Limpar conversa", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["pending_question"] = None
            st.rerun()


def render_header() -> None:
    st.title("📚 LexRAG")
    st.markdown(
        """
        **Assistente de Convenções Coletivas**

        Faça perguntas sobre **acordos e convenções coletivas**.
        O sistema busca os **trechos mais relevantes** dos documentos e gera
        respostas fundamentadas nas fontes encontradas.

        A interface mantém um **histórico curto da conversa** para dar
        continuidade entre perguntas relacionadas.
        """
    )
    st.divider()


def render_suggested_questions() -> None:
    st.subheader("Perguntas sugeridas")
    cols = st.columns(len(SUGGESTED_QUESTIONS))

    for col, example in zip(cols, SUGGESTED_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state["pending_question"] = example
            st.rerun()


def render_chat_history() -> None:
    if not st.session_state["chat_history"]:
        return

    st.subheader("Histórico da conversa")

    for idx, item in enumerate(st.session_state["chat_history"], start=1):
        with st.chat_message("user", avatar="👤"):
            st.markdown(item["question"])

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(item["answer"])

            if item.get("sources"):
                with st.expander(f"Fontes da resposta {idx}", expanded=False):
                    render_sources(item["sources"])


def render_sources(sources) -> None:
    if not sources:
        st.warning("Nenhuma fonte encontrada.")
        return

    st.markdown("**Fontes consultadas:**")
    for file_name, excerpt in sources:
        with st.container(border=True):
            st.markdown(f"**Arquivo:** {file_name}")
            st.markdown(f"**Trecho:** {excerpt}")


def build_conversation_context(max_turns: int = 3) -> str:
    history = st.session_state.get("chat_history", [])
    if not history:
        return ""

    recent_history = history[-max_turns:]
    parts = []

    for idx, item in enumerate(recent_history, start=1):
        parts.append(
            f"Pergunta anterior {idx}: {item['question']}\n"
            f"Resposta anterior {idx}: {item['answer']}"
        )

    return "\n\n".join(parts)


def update_prometheus_metrics() -> None:
    rag_requests_total.inc()

    rag_total_time_seconds.observe(
        float(telemetry.metrics.get("total_time", 0) or 0)
    )

    rag_retrieval_time_seconds.observe(
        float(telemetry.metrics.get("retrieval_time", 0) or 0)
    )

    rag_generation_time_seconds.observe(
        float(telemetry.metrics.get("generation_time", 0) or 0)
    )

    rag_chunks_retrieved.set(
        float(telemetry.metrics.get("chunks_retrieved", 0) or 0)
    )

    rag_chunks_used.set(
        float(telemetry.metrics.get("chunks_used", 0) or 0)
    )

    rag_top_score.set(
        float(telemetry.metrics.get("top_score", 0) or 0)
    )

    rag_avg_score.set(
        float(telemetry.metrics.get("avg_score", 0) or 0)
    )


def process_question(question: str) -> None:
    telemetry.reset()
    telemetry.logs["question"] = question

    conversation_context = build_conversation_context()
    start = telemetry.start_timer()

    try:
        with st.spinner("Consultando documentos..."):
            answer, sources = answer_question(
                question=question,
                conversation_context=conversation_context
            )

        telemetry.metrics["total_time"] = telemetry.stop_timer(start)
        telemetry.logs["answer"] = answer
        telemetry.logs["sources"] = sources

        update_prometheus_metrics()

        st.session_state["chat_history"].append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
            }
        )

        save_query_log(
            session_id=st.session_state["session_id"],
            trace_id=telemetry.trace_id,
            question=telemetry.logs.get("question", ""),
            answer=telemetry.logs.get("answer", ""),
            sources=telemetry.logs.get("sources", []),
            context=telemetry.logs.get("context", ""),
            prompt=telemetry.logs.get("prompt", ""),
            metrics=telemetry.metrics,
            error=telemetry.metrics.get("error"),
        )

    except Exception as exc:
        telemetry.metrics["total_time"] = telemetry.stop_timer(start)
        telemetry.metrics["error"] = str(exc)

        rag_requests_total.inc()
        rag_errors_total.inc()

        save_query_log(
            session_id=st.session_state["session_id"],
            trace_id=telemetry.trace_id,
            question=telemetry.logs.get("question", question),
            answer=telemetry.logs.get("answer", ""),
            sources=telemetry.logs.get("sources", []),
            context=telemetry.logs.get("context", ""),
            prompt=telemetry.logs.get("prompt", ""),
            metrics=telemetry.metrics,
            error=str(exc),
        )

        st.error(f"Erro na consulta: {exc}")


def main() -> None:
    init_db()
    start_metrics_server(8000)
    init_session_state()

    render_sidebar()
    render_header()
    render_suggested_questions()
    render_chat_history()

    # 1) pergunta enviada pelas sugestões
    if st.session_state.get("pending_question"):
        question = st.session_state["pending_question"]
        st.session_state["pending_question"] = None
        process_question(question)
        st.rerun()

    # 2) pergunta digitada no chat_input
    question = st.chat_input("Digite sua pergunta sobre convenções coletivas...")

    if question:
        process_question(question)
        st.rerun()


if __name__ == "__main__":
    main()