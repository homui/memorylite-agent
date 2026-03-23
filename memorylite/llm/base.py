from __future__ import annotations

from typing import Any, Protocol

from memorylite.schema import ChatMessage, ExtractionResult, RecallDecision, RecallItem


class MemoryController(Protocol):
    def decide_recall(
        self,
        query: str,
        recent_messages: list[ChatMessage],
        states: dict[str, dict[str, Any]],
        candidates: list[RecallItem],
        max_items: int,
    ) -> RecallDecision:
        ...

    def extract_memories(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        recent_messages: list[ChatMessage],
        existing_state: dict[str, Any],
        scope_ids: dict[str, str],
    ) -> ExtractionResult:
        ...


class JSONPromptClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


class EmbeddingClient(Protocol):
    model: str

    def embed_texts(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        ...


class ChatModel(Protocol):
    def generate(self, system_prompt: str, memory_context: str, user_message: str) -> str:
        ...
