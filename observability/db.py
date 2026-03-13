import os
from pathlib import Path
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

try:
    import streamlit as st
except Exception:
    st = None


SQLITE_PATH = Path("observability/debug_history.db")


def _get_database_url() -> str:
    # 1) tenta Streamlit secrets
    if st is not None:
        try:
            if "DATABASE_URL" in st.secrets:
                return st.secrets["DATABASE_URL"]
        except Exception:
            pass

    # 2) tenta variável de ambiente
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    # 3) fallback local para SQLite
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{SQLITE_PATH.as_posix()}"


def get_engine() -> Engine:
    db_url = _get_database_url()

    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    return create_engine(
        db_url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


@contextmanager
def get_connection():
    engine = get_engine()
    with engine.begin() as conn:
        yield conn


def init_db():
    engine = get_engine()

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    session_id TEXT,
                    trace_id TEXT,
                    question TEXT,
                    answer TEXT,
                    error TEXT,
                    context TEXT,
                    prompt TEXT,
                    sources_json TEXT,
                    metrics_json TEXT
                )
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    session_id TEXT,
                    trace_id TEXT,
                    question TEXT,
                    answer TEXT,
                    error TEXT,
                    context TEXT,
                    prompt TEXT,
                    sources_json TEXT,
                    metrics_json TEXT
                )
            """))
