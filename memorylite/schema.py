from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def now_ts() -> int:
    return int(time.time())


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
    created_at: int = field(default_factory=now_ts)
    message_id: int | None = None


@dataclass(slots=True)
class MemoryRecord:
    scope: str
    scope_id: str
    kind: str
    content: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.8
    created_at: int = field(default_factory=now_ts)
    updated_at: int = field(default_factory=now_ts)
    last_accessed_at: int | None = None
    expires_at: int | None = None
    supersedes_memory_id: int | None = None
    source_message_ids: list[int] = field(default_factory=list)
    memory_id: int | None = None


@dataclass(slots=True)
class RecallItem:
    memory_id: int
    scope: str
    scope_id: str
    kind: str
    content: str
    summary: str
    score: float
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RecallResult:
    query: str
    compiled_text: str
    triggered: bool
    reason: str
    items: list[RecallItem] = field(default_factory=list)
    recent_messages: list[ChatMessage] = field(default_factory=list)
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RecallDecision:
    should_recall: bool
    reason: str
    selected_memory_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ExtractionResult:
    memories: list[MemoryRecord] = field(default_factory=list)
    state_patch: dict[str, Any] = field(default_factory=dict)
    session_summary: str | None = None


@dataclass(slots=True)
class TurnResult:
    user_message: str
    assistant_message: str
    recall: RecallResult
