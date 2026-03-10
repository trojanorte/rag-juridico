import json
import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path("observability/debug_history.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
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
        )
        conn.commit()


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

    with get_connection() as conn:
        conn.execute(
            """
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                session_id,
                trace_id,
                question,
                answer,
                json.dumps(sources, ensure_ascii=False),
                context,
                prompt,
                json.dumps(metrics, ensure_ascii=False),
                error,
            ),
        )
        conn.commit()


def get_recent_logs(limit: int = 50):
    with get_connection() as conn:
        cursor = conn.execute(
            """
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
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

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