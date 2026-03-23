from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class MemoryAgentConfig:
    root_dir: str | Path = ".memorylite"
    database_name: str = "memory.sqlite3"
    archive_dir_name: str = "archives"
    recent_window: int = 6
    recent_prompt_window: int = 4
    candidate_pool_size: int = 12
    max_recall_items: int = 4
    max_context_tokens: int = 1200
    background_write: bool = True
    auto_archive_messages: bool = True
    enable_fts: bool = True
    wal_mode: bool = True
    sqlite_cache_kb: int = 20000
    sqlite_mmap_bytes: int = 268435456
    default_event_importance: float = 0.35
    default_fact_importance: float = 0.75
    default_preference_importance: float = 0.82
    recall_recency_weight: float = 0.18
    recall_importance_weight: float = 0.32
    recall_keyword_weight: float = 0.5
    recent_cache_size: int = 128
    state_cache_size: int = 256
    semantic_search_enabled: bool = False
    semantic_embedding_model: str | None = None
    semantic_rerank_weight: float = 0.28
    semantic_query_cache_size: int = 128
    no_candidate_short_circuit_enabled: bool = True
    local_recall_fallback_enabled: bool = True
    local_recall_fallback_score_threshold: float = 1.1
    local_recall_fallback_max_items: int = 2
    maintenance_enabled: bool = True
    maintenance_interval_turns: int = 12
    keep_recent_event_memories: int = 24
    compact_event_batch_size: int = 12
    event_ttl_days: int = 30
    task_state_ttl_days: int = 14
    summary_max_chars: int = 800
    state_scope: str = "user"
    session_scope: str = "session"
    project_scope: str = "project"
    global_scope: str = "global"
    ignored_event_prefixes: tuple[str, ...] = field(
        default_factory=lambda: (
            "ok",
            "okay",
            "noted",
            "understood",
            "thanks",
        )
    )

    @property
    def root_path(self) -> Path:
        return Path(self.root_dir).expanduser().resolve()

    @property
    def database_path(self) -> Path:
        return self.root_path / self.database_name

    @property
    def archive_path(self) -> Path:
        return self.root_path / self.archive_dir_name
