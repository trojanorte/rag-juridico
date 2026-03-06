import streamlit as st
from observability.telemetry import telemetry

st.set_page_config(
    page_title="Debug do RAG",
    layout="wide"
)

st.title("🛠 Debug do RAG")
st.write("Página técnica para acompanhamento interno.")

st.subheader("Métricas")
col1, col2, col3 = st.columns(3)
col1.metric("Tempo total", f"{telemetry.metrics['total_time']} s")
col2.metric("Retrieval", f"{telemetry.metrics['retrieval_time']} s")
col3.metric("Geração", f"{telemetry.metrics['generation_time']} s")

col4, col5 = st.columns(2)
col4.metric("Chunks", telemetry.metrics["chunks"])
col5.metric("Trace ID", telemetry.trace_id)

if telemetry.metrics["error"]:
    st.subheader("Erro")
    st.error(telemetry.metrics["error"])

st.divider()

st.subheader("Pergunta")
st.code(telemetry.logs["question"] or "Sem pergunta registrada.")

st.subheader("Resposta")
st.write(telemetry.logs["answer"] or "Sem resposta registrada.")

st.subheader("Fontes")
if telemetry.logs["sources"]:
    for arquivo, titulo in telemetry.logs["sources"]:
        st.write(f"- **{arquivo}** | {titulo}")
else:
    st.write("Sem fontes registradas.")

st.subheader("Contexto enviado ao modelo")
st.text_area(
    "Contexto",
    value=telemetry.logs["context"],
    height=250
)

st.subheader("Prompt enviado ao modelo")
st.text_area(
    "Prompt",
    value=telemetry.logs["prompt"],
    height=300
)