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


st.set_page_config(
    page_title="Assistente de Convenções Coletivas",
    layout="wide",
    page_icon="📄"
)


SUGGESTED_QUESTIONS = [
    "Qual é a vigência do acordo coletivo?",
    "Existe cláusula de seguro obrigatório?",
    "Qual é o reajuste salarial?"
]


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Sobre o sistema")
        st.write(
            """
            Este assistente utiliza **RAG (Retrieval-Augmented Generation)**
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
            """
        )

        st.divider()

        st.subheader("Status do sistema")
        st.success("RAG ativo")
        st.write("Embeddings + FAISS")


def render_header() -> None:
    st.title("📄 Assistente de Convenções Coletivas")
    st.markdown(
        """
        Faça perguntas sobre **acordos e convenções coletivas**.

        O sistema busca os **trechos mais relevantes** dos documentos e gera
        uma resposta fundamentada nas fontes encontradas.
        """
    )
    st.divider()


def render_suggested_questions() -> None:
    st.subheader("Perguntas sugeridas")
    cols = st.columns(len(SUGGESTED_QUESTIONS))

    for col, example in zip(cols, SUGGESTED_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state["question"] = example


def render_question_input():
    st.subheader("Faça sua pergunta")

    question = st.text_input(
        "Digite sua pergunta:",
        value=st.session_state.get("question", ""),
        placeholder="Ex: Qual é a vigência deste acordo coletivo?"
    )

    col1, col2 = st.columns(2)
    search_clicked = col1.button("Consultar", use_container_width=True)
    clear_clicked = col2.button("Limpar pergunta", use_container_width=True)

    if clear_clicked:
        st.session_state["question"] = ""
        st.rerun()

    return search_clicked, question.strip()


def render_sources(sources) -> None:
    st.subheader("Fontes")

    if not sources:
        st.warning("Nenhuma fonte encontrada.")
        return

    for file_name, excerpt in sources:
        with st.container(border=True):
            st.markdown(f"**Arquivo:** {file_name}")
            st.markdown(f"**Trecho:** {excerpt}")


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


def main() -> None:
    init_db()
    start_metrics_server(8000)

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    render_sidebar()
    render_header()
    render_suggested_questions()

    search_clicked, question = render_question_input()

    if search_clicked and not question:
        st.warning("Digite uma pergunta para continuar.")
        return

    if search_clicked and question:
        telemetry.reset()
        telemetry.logs["question"] = question

        start = telemetry.start_timer()

        try:
            with st.spinner("Consultando documentos..."):
                answer, sources = answer_question(question)

            telemetry.metrics["total_time"] = telemetry.stop_timer(start)
            telemetry.logs["answer"] = answer
            telemetry.logs["sources"] = sources

            update_prometheus_metrics()

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

            st.divider()
            st.subheader("Resposta")
            st.write(answer)

            render_sources(sources)

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


if __name__ == "__main__":
    main()