from __future__ import annotations

import math
from typing import Optional


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorEntry:
    def __init__(self, id: str, text: str, embedding: list[float], metadata: dict | None = None):
        self.id = id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}


class VectorStore:
    def __init__(self):
        self._entries: list[VectorEntry] = []

    def add(self, id: str, text: str, embedding: list[float], metadata: dict | None = None) -> None:
        self._entries.append(VectorEntry(id, text, embedding, metadata))

    def add_batch(
        self, items: list[tuple[str, str, list[float], dict | None]]
    ) -> None:
        for id, text, embedding, metadata in items:
            self.add(id, text, embedding, metadata)

    def search(
        self, query_embedding: list[float], top_k: int = 5, threshold: float = 0.0
    ) -> list[dict]:
        scored = []
        for e in self._entries:
            sim = _cosine_similarity(query_embedding, e.embedding)
            if sim >= threshold:
                scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": e.id,
                "text": e.text,
                "score": round(s, 4),
                "metadata": e.metadata,
            }
            for s, e in scored[:top_k]
        ]

    def get_entry(self, id: str) -> Optional[VectorEntry]:
        for e in self._entries:
            if e.id == id:
                return e
        return None

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)
