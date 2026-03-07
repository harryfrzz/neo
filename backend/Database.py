import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

_DB_DIR  = os.path.join(os.path.dirname(__file__), "data")
_DB_PATH = os.path.join(_DB_DIR, "neo_chat.db")
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(_DB_DIR, exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL DEFAULT 'New Chat',
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role          TEXT    NOT NULL,
    content       TEXT    NOT NULL,
    is_transcript INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE TABLE IF NOT EXISTS uploaded_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    filename    TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    chunks      INTEGER NOT NULL DEFAULT 0,
    detail      TEXT,
    file_size   INTEGER,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session      ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_files_session         ON uploaded_files(session_id);
"""


def init_db():
    try:
        conn = _get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        print(f"[db] SQLite schema initialised at {_DB_PATH}")
    except Exception as e:
        print(f"[db] ERROR initialising schema: {e}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _serialize_session(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id":         row["id"],
        "title":      row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _serialize_message(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id":            row["id"],
        "session_id":    row["session_id"],
        "role":          row["role"],
        "content":       row["content"],
        "is_transcript": bool(row["is_transcript"]),
        "created_at":    row["created_at"],
    }


def _serialize_file(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id":         row["id"],
        "session_id": row["session_id"],
        "filename":   row["filename"],
        "category":   row["category"],
        "status":     row["status"],
        "chunks":     row["chunks"],
        "detail":     row["detail"],
        "file_size":  row["file_size"],
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def create_session(title: str = "New Chat") -> Dict[str, Any]:
    conn = _get_conn()
    now = _now_iso()
    cur = conn.execute(
        "INSERT INTO chat_sessions (title, created_at, updated_at) VALUES (?, ?, ?)",
        (title, now, now),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM chat_sessions WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    return _serialize_session(_row_to_dict(row))


def list_sessions() -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM chat_sessions ORDER BY updated_at DESC"
    ).fetchall()
    return [_serialize_session(_row_to_dict(r)) for r in rows]


def get_session(session_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return _serialize_session(_row_to_dict(row)) if row else None


def delete_session(session_id: int) -> bool:
    conn = _get_conn()
    cur = conn.execute(
        "DELETE FROM chat_sessions WHERE id = ?", (session_id,)
    )
    conn.commit()
    return cur.rowcount > 0


def update_session_title(session_id: int, title: str) -> bool:
    conn = _get_conn()
    now = _now_iso()
    cur = conn.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
        (title, now, session_id),
    )
    conn.commit()
    return cur.rowcount > 0


def touch_session(session_id: int):
    """Bump updated_at after new messages are added."""
    conn = _get_conn()
    conn.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
        (_now_iso(), session_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def add_message(
    session_id: int,
    role: str,
    content: str,
    is_transcript: bool = False,
) -> Dict[str, Any]:
    conn = _get_conn()
    now = _now_iso()
    cur = conn.execute(
        """INSERT INTO messages (session_id, role, content, is_transcript, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, role, content, int(is_transcript), now),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    touch_session(session_id)
    return _serialize_message(_row_to_dict(row))


def get_messages(session_id: int) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC, id ASC",
        (session_id,),
    ).fetchall()
    return [_serialize_message(_row_to_dict(r)) for r in rows]


def delete_messages(session_id: int):
    conn = _get_conn()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Uploaded file helpers
# ---------------------------------------------------------------------------

def add_uploaded_file(
    session_id: int,
    filename: str,
    category: str,
    status: str,
    chunks: int = 0,
    detail: str = "",
    file_size: Optional[int] = None,
) -> Dict[str, Any]:
    conn = _get_conn()
    now = _now_iso()
    cur = conn.execute(
        """INSERT INTO uploaded_files
               (session_id, filename, category, status, chunks, detail, file_size, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, filename, category, status, chunks, detail, file_size, now),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM uploaded_files WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    return _serialize_file(_row_to_dict(row))


def get_uploaded_files(session_id: int) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM uploaded_files WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    return [_serialize_file(_row_to_dict(r)) for r in rows]
