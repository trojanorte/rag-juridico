import streamlit as st
from observability.debug_store import init_db, get_recent_logs, get_log_by_id

st.set_page_config(
    page_title="Debug do RAG",
    layout="wide",
    page_icon="🛠"
)

init_db()

st.title("🛠 Debug do RAG")
st.caption("Inspeção detalhada de consultas específicas registradas no histórico.")

limit = st.slider("Quantidade de registros carregados", 10, 200, 50, 10)
logs = get_recent_logs(limit=limit)

if not logs:
    st.info("Nenhum registro encontrado.")
    st.stop()

options = {}
for log in logs:
    label = f"#{log['id']} | {log['timestamp']} | {log['question'][:80]}"
    options[label] = log["id"]

selected_label = st.selectbox(
    "Selecione uma consulta para inspecionar",
    list(options.keys())
)

selected_id = options[selected_label]
selected_log = get_log_by_id(selected_id)

if not selected_log:
    st.error("Não foi possível carregar os detalhes da consulta.")
    st.stop()

metrics = selected_log.get("metrics", {})

st.divider()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("ID", selected_log["id"])
col2.metric("Trace ID", selected_log.get("trace_id", "N/A"))
col3.metric("Tempo total", str(metrics.get("total_time", 0)))
col4.metric("Retrieval", str(metrics.get("retrieval_time", 0)))
col5.metric("Geração", str(metrics.get("generation_time", 0)))

col6, col7, col8, col9 = st.columns(4)
col6.metric("Chunks recuperados", metrics.get("chunks_retrieved", 0))
col7.metric("Chunks usados", metrics.get("chunks_used", 0))
col8.metric("Top score", metrics.get("top_score", 0))
col9.metric("Avg score", metrics.get("avg_score", 0))

if selected_log.get("error"):
    st.divider()
    st.subheader("Erro")
    st.error(selected_log["error"])

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Pergunta", "Resposta", "Fontes", "Contexto", "Prompt"]
)

with tab1:
    st.subheader("Pergunta")
    st.code(selected_log.get("question", ""), language="text")

    st.write(f"**Timestamp:** {selected_log.get('timestamp', '')}")
    st.write(f"**Session ID:** {selected_log.get('session_id', '')}")

with tab2:
    st.subheader("Resposta")
    answer = selected_log.get("answer", "")
    if answer:
        st.write(answer)
        st.caption(f"Tamanho: {len(answer)} caracteres")
    else:
        st.info("Sem resposta registrada.")

with tab3:
    st.subheader("Fontes")
    sources = selected_log.get("sources", [])
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
        value=selected_log.get("context", ""),
        height=350
    )

with tab5:
    st.subheader("Prompt enviado ao modelo")
    st.text_area(
        "Prompt",
        value=selected_log.get("prompt", ""),
        height=400
    )

st.divider()

with st.expander("JSON bruto"):
    st.json(selected_log)