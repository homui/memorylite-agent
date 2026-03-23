from __future__ import annotations

from memorylite.config import MemoryAgentConfig
from memorylite.schema import RecallItem
from memorylite.store import SQLiteStore


class MemoryRetriever:
    def __init__(self, store: SQLiteStore, config: MemoryAgentConfig) -> None:
        self.store = store
        self.config = config

    def collect_candidates(
        self,
        query: str,
        scope_ids: dict[str, str],
        max_items: int,
    ) -> list[RecallItem]:
        limit = max(max_items, self.config.candidate_pool_size)
        items: list[RecallItem] = []
        seen_ids: set[int] = set()

        scope_order = []
        for scope_name in ("session", "user", "project", "global"):
            scope_id = scope_ids.get(scope_name)
            if scope_id:
                scope_order.append((scope_name, scope_id))

        for scope_filter in scope_order:
            found = self.store.search_memories(
                query=query,
                scope_filters=[scope_filter],
                target_kinds=None,
                limit=limit,
            )
            for item in found:
                if item.memory_id not in seen_ids:
                    items.append(item)
                    seen_ids.add(item.memory_id)
            if len(items) >= self.config.candidate_pool_size:
                break

        items.sort(key=lambda item: item.score, reverse=True)
        return items[: self.config.candidate_pool_size]
