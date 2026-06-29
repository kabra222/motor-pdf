from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DATA_DIR / "motor.db"
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(_DB_PATH))
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _init_tables(_local.conn)
    return _local.conn


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_vectors (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            page INTEGER DEFAULT 0,
            section TEXT,
            heading TEXT,
            chunk_index INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS agent_sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS agent_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES agent_sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_messages_session
            ON agent_messages(session_id, created_at);
    """)
    conn.commit()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class PersistentVectorStore:
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace

    def add(
        self,
        id: str,
        text: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
    ) -> None:
        meta = metadata or {}
        conn = _get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO agent_vectors
               (id, text, embedding, page, section, heading, chunk_index)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                f"{self.namespace}:{id}",
                text,
                json.dumps(embedding),
                meta.get("page", 0),
                meta.get("section"),
                meta.get("heading"),
                meta.get("chunk_index", 0),
            ),
        )
        conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM agent_vectors WHERE id LIKE ?",
            (f"{self.namespace}:%",),
        ).fetchall()
        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                scored.append((sim, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": r["id"].split(":", 1)[1] if ":" in r["id"] else r["id"],
                "text": r["text"],
                "score": round(s, 4),
                "metadata": {
                    "page": r["page"],
                    "section": r["section"],
                    "heading": r["heading"],
                    "chunk_index": r["chunk_index"],
                },
            }
            for s, r in scored[:top_k]
        ]

    def clear(self) -> None:
        conn = _get_conn()
        conn.execute(
            "DELETE FROM agent_vectors WHERE id LIKE ?",
            (f"{self.namespace}:%",),
        )
        conn.commit()

    @property
    def size(self) -> int:
        conn = _get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM agent_vectors WHERE id LIKE ?",
            (f"{self.namespace}:%",),
        ).fetchone()
        return row[0] if row else 0

    def get_entry(self, id: str) -> Optional[dict]:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM agent_vectors WHERE id = ?",
            (f"{self.namespace}:{id}",),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"].split(":", 1)[1] if ":" in row["id"] else row["id"],
            "text": row["text"],
            "metadata": {
                "page": row["page"],
                "section": row["section"],
                "heading": row["heading"],
                "chunk_index": row["chunk_index"],
            },
        }


# ── Session helpers ───────────────────────────────────────────────

def create_session() -> str:
    import uuid
    session_id = uuid.uuid4().hex[:12]
    conn = _get_conn()
    conn.execute(
        "INSERT INTO agent_sessions (id) VALUES (?)",
        (session_id,),
    )
    conn.commit()
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM agent_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return dict(row) if row else None


def add_message(session_id: str, role: str, content: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO agent_messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    conn.execute(
        "UPDATE agent_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (session_id,),
    )
    conn.commit()


def get_history(
    session_id: str, limit: int = 20
) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT role, content FROM agent_messages
           WHERE session_id = ? ORDER BY created_at ASC LIMIT ?""",
        (session_id, limit),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def list_sessions(limit: int = 10) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, title, created_at, updated_at
           FROM agent_sessions ORDER BY updated_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
