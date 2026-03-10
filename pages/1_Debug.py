import streamlit as st
from observability.telemetry import telemetry

st.set_page_config(
    page_title="Debug do RAG",
    layout="wide",
    page_icon="🛠"
)


def safe_metric(key, default=0):
    return telemetry.metrics.get(key, default)


def safe_log(key, default=""):
    return telemetry.logs.get(key, default)


st.title("🛠 Debug do RAG")
st.caption("Página técnica para acompanhamento interno da execução.")

st.divider()

# -----------------------------
# MÉTRICAS PRINCIPAIS
# -----------------------------
st.subheader("Métricas principais")

col1, col2, col3, col4, col5 = st.columns(5)

total_time = safe_metric("total_time", 0)
retrieval_time = safe_metric("retrieval_time", 0)
generation_time = safe_metric("generation_time", 0)
chunks = safe_metric("chunks", 0)
trace_id = getattr(telemetry, "trace_id", "N/A")

col1.metric("Tempo total", f"{total_time:.2f} s" if isinstance(total_time, (int, float)) else str(total_time))
col2.metric("Retrieval", f"{retrieval_time:.2f} s" if isinstance(retrieval_time, (int, float)) else str(retrieval_time))
col3.metric("Geração", f"{generation_time:.2f} s" if isinstance(generation_time, (int, float)) else str(generation_time))
col4.metric("Chunks", chunks)
col5.metric("Trace ID", str(trace_id))

error_message = safe_metric("error", None)
if error_message:
    st.divider()
    st.subheader("Erro")
    st.error(error_message)

st.divider()

# -----------------------------
# ABAS TÉCNICAS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Execução", "Resposta", "Fontes", "Contexto", "Prompt"]
)

with tab1:
    st.subheader("Pergunta")
    st.code(safe_log("question", "Sem pergunta registrada."), language="text")

    st.subheader("Resumo da execução")
    st.markdown(
        f"""
        - **Trace ID:** `{trace_id}`
        - **Tempo total:** `{total_time}`
        - **Tempo de retrieval:** `{retrieval_time}`
        - **Tempo de geração:** `{generation_time}`
        - **Quantidade de chunks:** `{chunks}`
        """
    )

with tab2:
    st.subheader("Resposta gerada")
    answer = safe_log("answer", "")
    if answer:
        st.write(answer)
        st.caption(f"Tamanho da resposta: {len(answer)} caracteres")
    else:
        st.info("Sem resposta registrada.")

with tab3:
    st.subheader("Fontes retornadas")
    sources = safe_log("sources", [])

    if sources:
        for idx, source in enumerate(sources, start=1):
            try:
                file_name, excerpt = source
            except Exception:
                file_name, excerpt = str(source), ""

            with st.container(border=True):
                st.markdown(f"**Fonte {idx}**")
                st.write(f"**Arquivo:** {file_name}")
                st.write(f"**Trecho:** {excerpt}")
    else:
        st.info("Sem fontes registradas.")

with tab4:
    st.subheader("Contexto enviado ao modelo")
    st.text_area(
        "Contexto",
        value=safe_log("context", ""),
        height=350
    )

with tab5:
    st.subheader("Prompt enviado ao modelo")
    st.text_area(
        "Prompt",
        value=safe_log("prompt", ""),
        height=400
    )

st.divider()

with st.expander("JSON bruto da telemetria"):
    st.json(
        {
            "trace_id": trace_id,
            "metrics": telemetry.metrics,
            "logs": telemetry.logs,
        }
    )