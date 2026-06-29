from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import uuid
from pathlib import Path

from app.models import DocumentJob, ExtractionResult, JobStatus

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DATA_DIR / "motor.db"


def _get_conn() -> sqlite3.Connection:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_sync() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'processing',
            result TEXT,
            error TEXT,
            progress INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    """)
    conn.commit()
    conn.close()


async def init_db() -> None:
    await asyncio.to_thread(_init_sync)


# ── Jobs ──────────────────────────────────────────────────────────


async def create_job() -> str:
    job_id = uuid.uuid4().hex[:12]

    def _create():
        conn = _get_conn()
        conn.execute(
            "INSERT INTO jobs (id, status) VALUES (?, ?)",
            (job_id, JobStatus.processing.value),
        )
        conn.commit()
        conn.close()

    await asyncio.to_thread(_create)
    return job_id


async def get_job(job_id: str) -> DocumentJob | None:
    def _get():
        conn = _get_conn()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        return row

    row = await asyncio.to_thread(_get)
    if not row:
        return None

    result = None
    if row["result"]:
        with contextlib.suppress(Exception):
            result = ExtractionResult(**json.loads(row["result"]))

    return DocumentJob(
        id=row["id"],
        status=JobStatus(row["status"]),
        result=result,
        error=row["error"],
        progress=row["progress"] or 0,
        total=row["total"] or 0,
    )


async def update_job(job_id: str, **kwargs) -> None:
    def _update():
        conn = _get_conn()
        sets = []
        values = []
        for k, v in kwargs.items():
            if k == "result":
                v = v.model_dump_json() if isinstance(v, ExtractionResult) else json.dumps(v) if not isinstance(v, str) else v
            elif isinstance(v, JobStatus):
                v = v.value
            sets.append(f"{k} = ?")
            values.append(v)
        values.append(job_id)
        conn.execute(
            f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", values
        )
        conn.commit()
        conn.close()

    await asyncio.to_thread(_update)


async def cleanup_stale_jobs(max_age_seconds: int = 300) -> int:
    def _clean():
        conn = _get_conn()
        cursor = conn.execute(
            """DELETE FROM jobs
               WHERE status IN ('done', 'error')
               AND created_at < datetime('now', ? || ' seconds')""",
            (f"-{max_age_seconds}",),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    return await asyncio.to_thread(_clean)


# ── Cache ──────────────────────────────────────────────────────────


async def cache_get(key: str) -> dict | None:
    def _get():
        conn = _get_conn()
        row = conn.execute(
            "SELECT data FROM cache WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        return row["data"] if row else None

    data = await asyncio.to_thread(_get)
    return json.loads(data) if data else None


async def cache_set(key: str, data: dict, ttl_seconds: int = 3600) -> None:
    def _set():
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, data) VALUES (?, ?)",
            (key, json.dumps(data)),
        )
        conn.commit()
        conn.close()

    await asyncio.to_thread(_set)


async def cache_clear() -> None:
    def _clear():
        conn = _get_conn()
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()

    await asyncio.to_thread(_clear)


async def cache_size() -> int:
    def _count():
        conn = _get_conn()
        row = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        conn.close()
        return row[0]

    return await asyncio.to_thread(_count)
