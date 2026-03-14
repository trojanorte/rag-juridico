import streamlit as st
from observability.debug_store import init_db, get_recent_logs

st.set_page_config(
    page_title="Histórico de Consultas",
    layout="wide",
    page_icon="🗂️"
)

init_db()

st.title("🗂️ Histórico de Consultas")
st.caption("Registro persistente das execuções do RAG.")

limit = st.slider("Quantidade de registros", min_value=10, max_value=200, value=50, step=10)

logs = get_recent_logs(limit=limit)

if not logs:
    st.info("Nenhum registro encontrado.")
else:
    for log in logs:
        title = f"#{log['id']} | {log['timestamp']} | {log['trace_id']}"
        with st.expander(title, expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Session ID:** {log['session_id']}")
            col2.write(f"**Trace ID:** {log['trace_id']}")
            col3.write(f"**Erro:** {log['error'] or 'Nenhum'}")

            st.markdown("**Pergunta**")
            st.code(log["question"] or "", language="text")

            st.markdown("**Resposta**")
            st.write(log["answer"] or "Sem resposta registrada.")

            st.markdown("**Fontes**")
            if log["sources"]:
                for idx, source in enumerate(log["sources"], start=1):
                    if isinstance(source, dict):
                        arquivo = source.get("arquivo", "arquivo_desconhecido")
                        titulo = source.get("titulo", "trecho_sem_titulo")
                        score = source.get("score", None)
                        label = source.get("label", f"Fonte {idx}")

                        with st.container(border=True):
                            st.markdown(f"**{label}**")
                            st.write(f"**Documento:** {arquivo}")
                            st.write(f"**Cláusula/Título:** {titulo}")
                            if score is not None:
                                st.write(f"**Relevância:** {score}")
                    else:
                        try:
                            file_name, excerpt = source
                        except Exception:
                            file_name, excerpt = str(source), ""

                        with st.container(border=True):
                            st.write(f"**Arquivo:** {file_name}")
                            st.write(f"**Trecho:** {excerpt}")
            else:
                st.write("Sem fontes.")

            st.markdown("**Métricas**")
            st.json(log["metrics"])