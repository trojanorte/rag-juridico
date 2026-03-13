import json
import os
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, text


DB_PATH = Path("observability/debug_history.db")


def get_database_url():
    return os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")


def get_engine():
    db_url = get_database_url()

    if db_url.startswith("sqlite"):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            db_url,
            future=True,
            connect_args={"check_same_thread": False},
        )

    return create_engine(db_url, future=True, pool_pre_ping=True)


def get_connection():
    return get_engine().begin()


def init_db() -> None:
    engine = get_engine()

    if engine.dialect.name == "sqlite":
        create_sql = """
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT,
            trace_id TEXT,
            question TEXT,
            answer TEXT,
            sources_json TEXT,
            context TEXT,
            prompt TEXT,
            metrics_json TEXT,
            error TEXT
        )
        """
    else:
        create_sql = """
        CREATE TABLE IF NOT EXISTS query_logs (
            id SERIAL PRIMARY KEY,
            timestamp TEXT NOT NULL,
            session_id TEXT,
            trace_id TEXT,
            question TEXT,
            answer TEXT,
            sources_json TEXT,
            context TEXT,
            prompt TEXT,
            metrics_json TEXT,
            error TEXT
        )
        """

    with engine.begin() as conn:
        conn.execute(text(create_sql))


def save_query_log(
    session_id: str,
    trace_id: str,
    question: str,
    answer: str,
    sources,
    context: str,
    prompt: str,
    metrics: dict,
    error: str = None,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")

    insert_sql = text("""
        INSERT INTO query_logs (
            timestamp,
            session_id,
            trace_id,
            question,
            answer,
            sources_json,
            context,
            prompt,
            metrics_json,
            error
        )
        VALUES (
            :timestamp,
            :session_id,
            :trace_id,
            :question,
            :answer,
            :sources_json,
            :context,
            :prompt,
            :metrics_json,
            :error
        )
    """)

    payload = {
        "timestamp": timestamp,
        "session_id": session_id,
        "trace_id": trace_id,
        "question": question,
        "answer": answer,
        "sources_json": json.dumps(sources, ensure_ascii=False),
        "context": context,
        "prompt": prompt,
        "metrics_json": json.dumps(metrics, ensure_ascii=False),
        "error": error,
    }

    with get_engine().begin() as conn:
        conn.execute(insert_sql, payload)


def get_recent_logs(limit: int = 50):
    query = text("""
        SELECT
            id,
            timestamp,
            session_id,
            trace_id,
            question,
            answer,
            sources_json,
            metrics_json,
            error
        FROM query_logs
        ORDER BY id DESC
        LIMIT :limit
    """)

    with get_engine().begin() as conn:
        rows = conn.execute(query, {"limit": limit}).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "id": row[0],
                "timestamp": row[1],
                "session_id": row[2],
                "trace_id": row[3],
                "question": row[4],
                "answer": row[5],
                "sources": json.loads(row[6]) if row[6] else [],
                "metrics": json.loads(row[7]) if row[7] else {},
                "error": row[8],
            }
        )
    return results


def get_log_by_id(log_id: int):
    query = text("""
        SELECT
            id,
            timestamp,
            session_id,
            trace_id,
            question,
            answer,
            sources_json,
            context,
            prompt,
            metrics_json,
            error
        FROM query_logs
        WHERE id = :log_id
    """)

    with get_engine().begin() as conn:
        row = conn.execute(query, {"log_id": log_id}).fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "timestamp": row[1],
        "session_id": row[2],
        "trace_id": row[3],
        "question": row[4],
        "answer": row[5],
        "sources": json.loads(row[6]) if row[6] else [],
        "context": row[7] or "",
        "prompt": row[8] or "",
        "metrics": json.loads(row[9]) if row[9] else {},
        "error": row[10],
    }
