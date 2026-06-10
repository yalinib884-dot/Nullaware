"""
backend/database/sqlite_manager.py
SQLite manager for conversation history and dataset session storage.
"""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize all required tables."""
    conn = _get_conn()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                dataset_name TEXT NOT NULL,
                csv_path TEXT,
                json_path TEXT,
                html_path TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_id
            ON conversation_history(conversation_id)
        """)
    conn.close()
    logger.info("Database initialized.")


# ── Conversation History ─────────────────────────────────────────────────────

def save_message(conversation_id: str, role: str, content: str):
    """Append a message to the conversation history."""
    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT INTO conversation_history (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, datetime.utcnow().isoformat()),
        )
    conn.close()


def get_history(conversation_id: str, limit: int = 20) -> list[dict]:
    """Retrieve the last N messages for a conversation."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT role, content, timestamp FROM conversation_history
           WHERE conversation_id = ?
           ORDER BY id DESC LIMIT ?""",
        (conversation_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def clear_history(conversation_id: str):
    """Clear all messages for a conversation."""
    conn = _get_conn()
    with conn:
        conn.execute(
            "DELETE FROM conversation_history WHERE conversation_id = ?",
            (conversation_id,),
        )
    conn.close()


# ── Dataset Sessions ─────────────────────────────────────────────────────────

def save_session(session_id: str, dataset_name: str, csv_path: str,
                 json_path: str, html_path: str, metadata: dict):
    """Save or update a dataset session."""
    conn = _get_conn()
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO dataset_sessions
                (session_id, dataset_name, csv_path, json_path, html_path, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, dataset_name, csv_path, json_path, html_path,
            json.dumps(metadata, default=str),
            datetime.utcnow().isoformat(),
        ))
    conn.close()


def get_session(session_id: str) -> dict | None:
    """Retrieve a dataset session."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM dataset_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata_json", "{}"))
        return d
    return None


def list_sessions() -> list[dict]:
    """List all dataset sessions."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT session_id, dataset_name, created_at FROM dataset_sessions ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]