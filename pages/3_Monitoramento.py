import sqlite3
from pathlib import Path
import json

import pandas as pd
import streamlit as st


DB_PATH = Path("observability/debug_history.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def parse_metrics(metrics_json):
    if not metrics_json:
        return {}

    try:
        return json.loads(metrics_json)
    except Exception:
        return {}


def load_monitoring_data(limit: int = 500):
    if not DB_PATH.exists():
        return pd.DataFrame()

    with get_connection() as conn:
        query = """
        SELECT
            id,
            timestamp,
            session_id,
            trace_id,
            question,
            error,
            metrics_json
        FROM query_logs
        ORDER BY id DESC
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))

    if df.empty:
        return df

    metrics_expanded = df["metrics_json"].apply(parse_metrics).apply(pd.Series)
    df = pd.concat([df.drop(columns=["metrics_json"]), metrics_expanded], axis=1)

    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ===============================
# FUNÇÃO DE CORES PARA O DASHBOARD
# ===============================

def highlight_rows(row):

    # erro -> vermelho
    if str(row.get("error", "")).strip():
        return ["background-color: #ffcccc"] * len(row)

    # latência alta -> laranja
    if row.get("total_time", 0) > 5:
        return ["background-color: #ffe5b4"] * len(row)

    # retrieval fraco -> amarelo
    if row.get("top_score", 1) < 0.20:
        return ["background-color: #fff3cd"] * len(row)

    return [""] * len(row)


st.set_page_config(
    page_title="Monitoramento do RAG",
    layout="wide",
    page_icon="📈"
)

st.title("📈 Monitoramento do RAG")
st.caption("Painel operacional com métricas agregadas das consultas registradas.")

limit = st.slider("Quantidade de registros analisados", 50, 2000, 500, 50)

df = load_monitoring_data(limit=limit)

if df.empty:
    st.info("Nenhum dado encontrado no banco de histórico.")
    st.stop()

for col in [
    "total_time",
    "retrieval_time",
    "generation_time",
    "chunks_retrieved",
    "chunks_used",
    "top_score",
    "avg_score",
]:
    if col not in df.columns:
        df[col] = 0

if "error" not in df.columns:
    df["error"] = ""

df["error"] = df["error"].fillna("").astype(str)
df["has_error"] = df["error"].str.strip().ne("")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp")

# ===============================
# KPIs
# ===============================

total_queries = len(df)
total_errors = int(df["has_error"].sum())
error_rate = round((total_errors / total_queries) * 100, 2) if total_queries else 0
avg_total_time = round(df["total_time"].fillna(0).mean(), 3)
avg_retrieval_time = round(df["retrieval_time"].fillna(0).mean(), 3)
avg_generation_time = round(df["generation_time"].fillna(0).mean(), 3)
avg_top_score = round(df["top_score"].fillna(0).mean(), 4)
avg_avg_score = round(df["avg_score"].fillna(0).mean(), 4)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Consultas totais", total_queries)
col2.metric("Erros", total_errors)
col3.metric("Taxa de erro", f"{error_rate}%")
col4.metric("Tempo médio total", f"{avg_total_time}s")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Retrieval médio", f"{avg_retrieval_time}s")
col6.metric("Geração média", f"{avg_generation_time}s")
col7.metric("Top score médio", avg_top_score)
col8.metric("Avg score médio", avg_avg_score)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Latência", "Qualidade do Retrieval", "Erros", "Últimas Consultas"]
)

# ===============================
# LATÊNCIA
# ===============================

with tab1:
    st.subheader("Latência ao longo do tempo")

    latency_df = df[["timestamp", "total_time"]].dropna().set_index("timestamp")
    retrieval_df = df[["timestamp", "retrieval_time"]].dropna().set_index("timestamp")
    generation_df = df[["timestamp", "generation_time"]].dropna().set_index("timestamp")

    if not latency_df.empty:
        st.line_chart(latency_df)

    if not retrieval_df.empty:
        st.line_chart(retrieval_df)

    if not generation_df.empty:
        st.line_chart(generation_df)


# ===============================
# QUALIDADE DO RETRIEVAL
# ===============================

with tab2:
    st.subheader("Qualidade do retrieval")

    score_df = df[["timestamp", "top_score", "avg_score"]].dropna().set_index("timestamp")
    chunks_df = df[["timestamp", "chunks_retrieved", "chunks_used"]].dropna().set_index("timestamp")

    if not score_df.empty:
        st.line_chart(score_df)

    if not chunks_df.empty:
        st.line_chart(chunks_df)


# ===============================
# ERROS
# ===============================

with tab3:
    st.subheader("Consultas com erro")

    error_df = df[df["has_error"]].copy()

    if error_df.empty:
        st.success("Nenhum erro registrado nos dados carregados.")
    else:
        st.dataframe(
            error_df[["timestamp", "question", "error"]],
            use_container_width=True
        )


# ===============================
# ÚLTIMAS CONSULTAS
# ===============================

with tab4:
    st.subheader("Últimas consultas")

    preview_df = df[[
        "timestamp",
        "question",
        "total_time",
        "retrieval_time",
        "generation_time",
        "top_score",
        "avg_score",
        "error"
    ]].sort_values("timestamp", ascending=False)

    styled_df = preview_df.style.apply(highlight_rows, axis=1)

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )