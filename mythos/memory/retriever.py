"""Memory retrieval interface placeholder."""

from __future__ import annotations

from typing import Protocol


class Retriever(Protocol):
    def retrieve(self, query: str, *, k: int = 4) -> list[str]:
        """Return relevant text chunks for a query."""
