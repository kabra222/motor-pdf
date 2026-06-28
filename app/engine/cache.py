from __future__ import annotations

import hashlib
import json

from app.engine.storage import cache_clear, cache_get, cache_set, cache_size


class PDFCache:
    def make_key(self, file_bytes: bytes, params: dict) -> str:
        canonical = json.dumps(params, sort_keys=True, ensure_ascii=False)
        content = file_bytes + canonical.encode()
        return hashlib.sha256(content).hexdigest()

    async def get(self, key: str) -> dict | None:
        return await cache_get(key)

    async def set(self, key: str, data: dict) -> None:
        await cache_set(key, data)

    async def clear(self) -> None:
        await cache_clear()

    async def size(self) -> int:
        return await cache_size()
