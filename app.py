"""LexRAG user interface."""
import logging
import time
import uuid

import streamlit as st

from core.config import SELECT_DOCUMENT
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
from observability.telemetry import telemetry
from vectorstore.deploy import ensure_index_available
from vectorstore.faiss_store import FAISSStore


log = logging.getLogger(__name__)

st.set_page_config(
    page_title="LexRAG — Convenções Coletivas",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource
def catalog(index_version):
    return FAISSStore().load()


def fresh_conversation_state():
    return {
        "first_legal_question": None,
        "last_legal_question": None,
        "current_topic": None,
        "recent_legal_questions": [],
    }


def init_session():
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("selected_document", "")
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault(
        "conversation_state",
        fresh_conversation_state(),
    )


def source_preview_data(source):
    return {
        "document_title": source["document_title"],
        "filename": source["arquivo"],
        "clause_number": source["clause_number"],
        "clause_title": source["titulo"],
        "content": source["content"],
        "score": source["score"],
        "chunk_id": source["chunk_id"],
    }


def render_sources(sources):
    for source in sources:
        preview = source_preview_data(source)

        title = preview["clause_title"] or "Trecho da convenção"

        with st.expander(f"Fonte {source['id']} · {title}"):
            st.write(f"**Convenção:** {preview['document_title']}")
            st.write(f"**Arquivo:** {preview['filename']}")

            clause_number = preview["clause_number"] or "não identificada"

            st.write(
                f"**Cláusula:** {clause_number} — {title}"
            )

            content = preview["content"].replace(
                "\n",
                "\n> ",
            )

            st.markdown(f"> {content}")

            st.caption(
                f"Score vetorial: {preview['score']:.3f} · "
                f"Chunk: {preview['chunk_id']}"
            )


def conversation_context():
    rows = st.session_state["chat_history"][-4:]

    return "\n".join(
        (
            f"Pergunta anterior {i}: {row['question']}\n"
            f"Resposta anterior {i}: {row['answer']}"
        )
        for i, row in enumerate(rows, 1)
    )


def update_metrics():
    metrics = telemetry.metrics

    rag_requests_total.inc()

    rag_total_time_seconds.observe(
        float(metrics.get("total_time", 0))
    )

    rag_retrieval_time_seconds.observe(
        float(metrics.get("retrieval_time", 0))
    )

    rag_generation_time_seconds.observe(
        float(metrics.get("generation_time", 0))
    )

    rag_chunks_retrieved.set(
        float(metrics.get("chunks_retrieved", 0))
    )

    rag_chunks_used.set(
        float(metrics.get("chunks_used", 0))
    )

    rag_top_score.set(
        float(metrics.get("top_score", 0))
    )

    rag_avg_score.set(
        float(metrics.get("avg_score", 0))
    )


def process_question(question):
    from rag_generator import answer_question

    telemetry.reset()

    st.session_state["last_error"] = None

    document_id = (
        st.session_state["selected_document"]
        or None
    )

    previous = conversation_context()

    started = time.perf_counter()

    answer = ""
    sources = []
    error_type = None

    try:
        with st.status(
            "Preparando consulta...",
            expanded=False,
        ) as status:

            answer, sources = answer_question(
                question,
                previous,
                document_id=document_id,
                on_stage=lambda label: status.update(
                    label=label
                ),
            )

            status.update(
                label="Consulta concluída",
                state="complete",
            )

        st.session_state["chat_history"].append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
                "document_id": document_id,
            }
        )

    except Exception as exc:
        error_type = type(exc).__name__

        rag_errors_total.inc()

        log.exception("Query failed")

        st.session_state["last_error"] = (
            "Ocorreu um erro ao processar a consulta. "
            "Tente novamente."
        )

        answer = "Ocorreu um erro ao processar a consulta."

    finally:
        telemetry.metrics["total_time"] = round(
            time.perf_counter() - started,
            4,
        )

        for stage in (
            "total",
            "retrieval",
            "rewrite",
            "generation",
        ):
            telemetry.metrics[
                stage + "_ms"
            ] = round(
                1000
                * float(
                    telemetry.metrics.get(
                        stage + "_time",
                        0,
                    )
                ),
                2,
            )

        telemetry.metrics["document_id"] = document_id
        telemetry.metrics["error_type"] = error_type

        update_metrics()

        try:
            save_query_log(
                session_id=st.session_state[
                    "session_id"
                ],
                trace_id=telemetry.trace_id,
                question="",
                answer="",
                sources=[
                    {
                        "chunk_id": s["chunk_id"],
                        "document_id": s[
                            "document_id"
                        ],
                    }
                    for s in sources
                ],
                context="",
                prompt="",
                metrics=telemetry.metrics,
                error=error_type,
            )

        except Exception:
            log.exception(
                "Could not persist query telemetry"
            )


def main():
    init_session()
    init_db()

    try:
        start_metrics_server(8000)
    except OSError:
        log.warning(
            "Metrics port unavailable"
        )

    try:
        store = catalog(
            ensure_index_available()
        )

        documents = store.list_documents()

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ):
        log.exception("Index unavailable")

        st.error(
            "Índice indisponível. "
            "Um operador deve executar a indexação "
            "antes das consultas."
        )

        st.stop()

    # SIDEBAR
    with st.sidebar:

        st.title("LexRAG")

        if st.button(
            "Nova conversa",
            width="stretch",
        ):
            st.session_state[
                "chat_history"
            ] = []

            st.session_state[
                "session_id"
            ] = str(uuid.uuid4())

            st.session_state[
                "conversation_state"
            ] = fresh_conversation_state()

            st.rerun()

        st.divider()

        options = {
            "Selecionar convenção": ""
        }

        for doc in documents:
            label = (
                f"{doc['document_title']} · "
                f"{doc['filename']} · "
                f"{doc['document_id'][:8]}"
            )

            options[label] = doc[
                "document_id"
            ]

        values = list(
            options.values()
        )

        current_document = (
            st.session_state[
                "selected_document"
            ]
        )

        if current_document in values:
            index = values.index(
                current_document
            )
        else:
            index = 0

        selected = st.selectbox(
            "Convenção",
            list(options.keys()),
            index=index,
        )

        new_id = options[selected]

        # IMPORTANTE:
        # selecionar a primeira convenção
        # não apaga a conversa anterior.
        #
        # A conversa só é reiniciada
        # se o usuário trocar de uma
        # convenção já selecionada
        # para outra diferente.
        if (
            new_id
            != st.session_state[
                "selected_document"
            ]
        ):
            old_id = (
                st.session_state[
                    "selected_document"
                ]
            )

            st.session_state[
                "selected_document"
            ] = new_id

            if (
                old_id
                and new_id
                and old_id != new_id
            ):
                st.session_state[
                    "chat_history"
                ] = []

                st.session_state[
                    "conversation_state"
                ] = fresh_conversation_state()

                st.session_state[
                    "session_id"
                ] = str(uuid.uuid4())

            st.rerun()

        st.divider()

        st.caption(
            f"Índice carregado · "
            f"{len(documents)} convenções"
        )

        st.caption(
            "Versão: "
            f"{store.manifest['index_version'][:12]}"
        )

        st.caption(
            "Indexado em: "
            f"{store.manifest['created_at']}"
        )

        if new_id:
            doc = next(
                item
                for item in documents
                if item["document_id"]
                == new_id
            )

            st.write(
                f"**Documento:** "
                f"{doc['filename']}"
            )

            vigencia_inicio = (
                doc["vigencia_inicio"]
                or "não identificada"
            )

            vigencia_fim = (
                doc["vigencia_fim"]
                or "não identificada"
            )

            st.write(
                f"**Vigência:** "
                f"{vigencia_inicio} — "
                f"{vigencia_fim}"
            )

    # CONTEÚDO PRINCIPAL
    st.title(
        "Consulte sua convenção coletiva"
    )

    if new_id:
        st.write(
            "Faça perguntas sobre cláusulas, "
            "benefícios, adicionais, jornadas, "
            "pisos, reajustes e outras regras "
            "da convenção selecionada."
        )
    else:
        st.write(
            "Converse comigo ou selecione uma "
            "convenção coletiva para realizar "
            "consultas jurídicas específicas."
        )

    if st.session_state["last_error"]:
        st.error(
            st.session_state[
                "last_error"
            ]
        )

    if not new_id:
        st.info(
            "Para perguntas específicas sobre "
            "cláusulas, salários, benefícios, "
            "jornada ou outras regras jurídicas, "
            "selecione uma convenção na barra lateral."
        )

    # HISTÓRICO DO CHAT
    for row in st.session_state[
        "chat_history"
    ]:

        with st.chat_message("user"):
            st.write(
                row["question"]
            )

        with st.chat_message(
            "assistant"
        ):
            st.markdown(
                row["answer"]
            )

            if row["sources"]:
                render_sources(
                    row["sources"]
                )

    # INPUT
    if new_id:
        placeholder = (
            "Pergunte sobre a "
            "convenção selecionada"
        )
    else:
        placeholder = (
            "Converse comigo ou selecione "
            "uma convenção para consultas jurídicas"
        )

    question = st.chat_input(
        placeholder
    )

    if question:
        process_question(
            question
        )

        st.rerun()


if __name__ == "__main__":
    main()