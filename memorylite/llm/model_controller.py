from __future__ import annotations

import re
from typing import Any

from memorylite.config import MemoryAgentConfig
from memorylite.llm.base import JSONPromptClient
from memorylite.schema import ChatMessage, ExtractionResult, MemoryRecord, RecallDecision, RecallItem


class ModelMemoryController:
    VALID_SCOPE_KEYS = ("session", "user", "project")
    VALID_KINDS = ("fact", "preference", "event", "summary", "task_state")
    TECH_TERMS = (
        "python",
        "sqlite",
        "postgres",
        "postgresql",
        "mysql",
        "api",
        "database",
        "memory",
        "agent",
    )
    WEEKDAY_TERMS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

    def __init__(self, config: MemoryAgentConfig, client: JSONPromptClient) -> None:
        self.config = config
        self.client = client

    def decide_recall(
        self,
        query: str,
        recent_messages: list[ChatMessage],
        states: dict[str, dict[str, Any]],
        candidates: list[RecallItem],
        max_items: int,
    ) -> RecallDecision:
        compact_candidates = [
            {
                "id": item.memory_id,
                "kind": item.kind,
                "scope": item.scope,
                "summary": (item.summary or item.content[:140])[:140],
                "score": round(item.score, 4),
            }
            for item in candidates[: self.config.candidate_pool_size]
        ]
        try:
            payload = self.client.complete_json(
                system_prompt=(
                    "You are a lightweight memory recall controller. "
                    "You receive the current query, a short recent conversation window, compact state, "
                    "and a small candidate pool from local storage. "
                    "Decide whether long-term memory should be injected now, and if yes, choose the best candidate IDs. "
                    "Return strict JSON with keys: should_recall (bool), reason (str), selected_memory_ids (list[int]). "
                    "Keep the decision simple and conservative, but if the candidate pool already contains a clearly relevant memory, set should_recall=true. "
                    "Select at most max_items IDs and prefer exact topic matches over recent but unrelated items. "
                    "If no candidate is useful, return should_recall=false and an empty selected_memory_ids list."
                ),
                user_prompt=(
                    f"query={query!r}\n"
                    f"max_items={max_items}\n"
                    f"recent_messages={[{'role': m.role, 'content': m.content[:120]} for m in recent_messages[-self.config.recent_prompt_window:]]!r}\n"
                    f"states={self._compact_states(states)!r}\n"
                    f"candidates={compact_candidates!r}"
                ),
            )
        except Exception:
            fallback_ids = [item.memory_id for item in candidates[:max_items]]
            return RecallDecision(
                should_recall=bool(fallback_ids),
                reason="controller_fallback",
                selected_memory_ids=fallback_ids,
            )

        selected_memory_ids = self._coerce_int_list(payload.get("selected_memory_ids"))[:max_items]
        should_recall = bool(payload.get("should_recall", bool(selected_memory_ids)))
        reason = str(payload.get("reason", "model_recall"))
        if not should_recall:
            selected_memory_ids = []
        return RecallDecision(
            should_recall=should_recall,
            reason=reason,
            selected_memory_ids=selected_memory_ids,
        )

    def extract_memories(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        recent_messages: list[ChatMessage],
        existing_state: dict[str, Any],
        scope_ids: dict[str, str],
    ) -> ExtractionResult:
        try:
            payload = self.client.complete_json(
                system_prompt=(
                    "You are a lightweight memory-writing agent. "
                    "Extract only durable, useful memory from the latest conversation turn. "
                    "Return strict JSON with keys: state_patch (object), memories (list[object]). "
                    "Each memory object must include scope, scope_id_key, kind, content, summary, tags, importance, confidence. "
                    "scope_id_key must be one of these literal enum values only: session, user, project. "
                    "Never put real IDs like user-123 or project-abc into scope_id_key. "
                    "scope should normally match scope_id_key. If a project ID is not available in scope_ids, fall back to session or user. "
                    "Valid kind values: fact, preference, event, summary, task_state. "
                    "For content, preserve the user's original durable statement as literally as possible. "
                    "Do not rewrite a fact into a vague label like 'project stack is ...' if the user already stated the fact directly. "
                    "Preference memories should usually use user scope when a user scope exists. "
                    "Task state and personal plans should usually use session scope. "
                    "Write at most 2 memories unless the turn clearly contains multiple durable facts."
                ),
                user_prompt=(
                    f"user_message={user_message!r}\n"
                    f"assistant_message={assistant_message!r}\n"
                    f"recent_messages={[{'role': m.role, 'content': m.content[:120]} for m in recent_messages[-self.config.recent_prompt_window:]]!r}\n"
                    f"existing_state={existing_state!r}\n"
                    f"scope_ids={scope_ids!r}"
                ),
            )
        except Exception:
            return ExtractionResult()

        raw_memories = payload.get("memories", [])
        if not isinstance(raw_memories, list):
            raw_memories = []

        memories: list[MemoryRecord] = []
        for item in raw_memories:
            memory = self._build_memory_record(
                item=item,
                raw_memory_count=len(raw_memories),
                user_message=user_message,
                assistant_message=assistant_message,
                scope_ids=scope_ids,
            )
            if memory is not None:
                memories.append(memory)

        state_patch = payload.get("state_patch", {}) or {}
        if not isinstance(state_patch, dict):
            state_patch = {}
        return ExtractionResult(memories=memories, state_patch=state_patch)

    def _build_memory_record(
        self,
        item: Any,
        raw_memory_count: int,
        user_message: str,
        assistant_message: str,
        scope_ids: dict[str, str],
    ) -> MemoryRecord | None:
        normalized_scope_key = self._normalize_scope_key(
            item.get("scope_id_key"),
            item.get("scope"),
            scope_ids,
        )
        if not normalized_scope_key:
            return None

        kind = self._normalize_kind(item.get("kind"))
        raw_content = self._normalize_text_field(item.get("content", ""))
        raw_summary = self._normalize_text_field(item.get("summary", ""))

        content = self._stabilize_content(
            kind=kind,
            raw_content=raw_content,
            raw_summary=raw_summary,
            user_message=user_message,
            assistant_message=assistant_message,
            raw_memory_count=raw_memory_count,
        )
        if not content:
            return None

        normalized_scope = self._normalize_scope(item.get("scope"), normalized_scope_key, scope_ids)
        normalized_scope = self._stabilize_scope(
            kind=kind,
            content=content,
            summary=raw_summary,
            current_scope=normalized_scope,
            scope_ids=scope_ids,
        )
        if normalized_scope not in scope_ids:
            normalized_scope = normalized_scope_key
        scope_id = scope_ids.get(normalized_scope)
        if not scope_id:
            return None

        summary = self._stabilize_summary(kind=kind, content=content, raw_summary=raw_summary)
        tags = self._stabilize_tags(item.get("tags"), content, summary)

        return MemoryRecord(
            scope=normalized_scope,
            scope_id=scope_id,
            kind=kind,
            content=content[:500],
            summary=summary[:140],
            tags=tags,
            importance=self._normalize_score(item.get("importance"), 0.6),
            confidence=self._normalize_score(item.get("confidence"), 0.75),
        )

    def _compact_states(self, states: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        compact: dict[str, dict[str, Any]] = {}
        for scope, payload in states.items():
            if not payload:
                continue
            compact[scope] = {}
            for key, value in list(payload.items())[:8]:
                compact[scope][str(key)] = str(value)[:120]
        return compact

    def _coerce_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item)[:40] for item in value if str(item).strip()]
        return []

    def _coerce_int_list(self, value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    def _normalize_scope_key(
        self,
        raw_scope_key: Any,
        raw_scope: Any,
        scope_ids: dict[str, str],
    ) -> str | None:
        candidates = [
            str(raw_scope_key or "").strip().lower(),
            str(raw_scope or "").strip().lower(),
        ]
        for candidate in candidates:
            if candidate in self.VALID_SCOPE_KEYS and candidate in scope_ids:
                return candidate

        raw_scope_key_text = str(raw_scope_key or "").strip().lower()
        raw_scope_text = str(raw_scope or "").strip().lower()

        for candidate_key, candidate_value in scope_ids.items():
            value_lower = candidate_value.lower()
            if raw_scope_key_text == value_lower or raw_scope_text == value_lower:
                return candidate_key

        for token in self.VALID_SCOPE_KEYS:
            if token in raw_scope_key_text:
                if token == "project" and token not in scope_ids:
                    continue
                if token in scope_ids:
                    return token
            if token in raw_scope_text:
                if token == "project" and token not in scope_ids:
                    continue
                if token in scope_ids:
                    return token

        if "project" in scope_ids:
            return "project"
        if "user" in scope_ids:
            return "user"
        if "session" in scope_ids:
            return "session"
        return None

    def _normalize_scope(self, raw_scope: Any, normalized_scope_key: str, scope_ids: dict[str, str]) -> str:
        scope = str(raw_scope or "").strip().lower()
        if scope in self.VALID_SCOPE_KEYS and scope in scope_ids:
            return scope
        return normalized_scope_key

    def _stabilize_scope(
        self,
        kind: str,
        content: str,
        summary: str,
        current_scope: str,
        scope_ids: dict[str, str],
    ) -> str:
        text = f"{content} {summary}".lower()
        if kind == "preference" and "user" in scope_ids:
            return "user"
        if kind == "task_state" and current_scope == "user" and "user" in scope_ids:
            return "user"
        if kind in {"event", "task_state"} and "session" in scope_ids:
            return "session"
        if kind == "fact" and "project" in scope_ids and ("project" in text or "stack" in text or "database" in text):
            return "project"
        if current_scope in scope_ids:
            return current_scope
        if "session" in scope_ids:
            return "session"
        if "user" in scope_ids:
            return "user"
        if "project" in scope_ids:
            return "project"
        return current_scope

    def _normalize_kind(self, value: Any) -> str:
        kind = str(value or "event").strip().lower()
        if kind in self.VALID_KINDS:
            return kind
        if "task" in kind or "todo" in kind:
            return "task_state"
        if "prefer" in kind:
            return "preference"
        if "fact" in kind:
            return "fact"
        if "summary" in kind:
            return "summary"
        return "event"

    def _normalize_text_field(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.fullmatch(r"(user_message|assistant_message|content)\s*=\s*(['\"])(.*)\2", text, re.DOTALL)
        if match:
            return match.group(3).strip()
        return re.sub(r"\s+", " ", text).strip()

    def _stabilize_content(
        self,
        kind: str,
        raw_content: str,
        raw_summary: str,
        user_message: str,
        assistant_message: str,
        raw_memory_count: int,
    ) -> str:
        user_text = self._normalize_text_field(user_message)
        assistant_text = self._normalize_text_field(assistant_message)
        content = raw_content or raw_summary
        if not content:
            return ""

        if assistant_text and self._normalized_text_key(content) == self._normalized_text_key(assistant_text):
            content = user_text or content

        if user_text and kind in {"fact", "preference", "event", "task_state"}:
            overlap = self._text_overlap(user_text, f"{raw_content} {raw_summary}")
            if raw_memory_count == 1 and len(user_text) <= 320 and overlap >= 0.28:
                content = user_text
            elif self._looks_templated(raw_content or raw_summary) and overlap >= 0.18:
                content = user_text

        return self._normalize_text_field(content)

    def _stabilize_summary(self, kind: str, content: str, raw_summary: str) -> str:
        summary = self._normalize_text_field(raw_summary)
        if summary and not self._looks_templated(summary):
            return summary[:140]
        if kind == "preference":
            return self._trim_text(content, 140)
        if kind == "task_state":
            return self._trim_text(content, 140)
        if kind == "event":
            return self._trim_text(content, 140)
        if kind == "fact":
            return self._trim_text(content, 140)
        return self._trim_text(content, 140)

    def _stabilize_tags(self, raw_tags: Any, content: str, summary: str) -> list[str]:
        tags = self._coerce_list(raw_tags)
        inferred = self._infer_tags(f"{content} {summary}")
        merged: list[str] = []
        for tag in tags + inferred:
            cleaned = str(tag).strip().lower()
            if not cleaned or cleaned in merged:
                continue
            merged.append(cleaned[:40])
        return merged[:8]

    def _infer_tags(self, text: str) -> list[str]:
        lowered = text.lower()
        tags: list[str] = []
        for term in self.WEEKDAY_TERMS:
            if term in lowered:
                tags.append(term)
        for term in self.TECH_TERMS:
            if term in lowered:
                tags.append(term)
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,20}", lowered)
        for token in tokens:
            if token in {
                "please",
                "remember",
                "understood",
                "answers",
                "answer",
                "today",
                "week",
                "this",
                "that",
                "with",
            }:
                continue
            tags.append(token)
        result: list[str] = []
        for tag in tags:
            if tag not in result:
                result.append(tag)
        return result[:8]

    def _looks_templated(self, text: str) -> bool:
        lowered = self._normalize_text_field(text).lower()
        if not lowered:
            return False
        return any(
            pattern in lowered
            for pattern in (
                "project stack is",
                "user prefers",
                "preference:",
                "task state",
                "monday plan is",
                "plan is",
            )
        )

    def _normalized_text_key(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _text_overlap(self, lhs: str, rhs: str) -> float:
        lhs_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,20}", lhs.lower()))
        rhs_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,20}", rhs.lower()))
        if not lhs_terms or not rhs_terms:
            return 0.0
        return len(lhs_terms & rhs_terms) / max(1, min(len(lhs_terms), len(rhs_terms)))

    def _trim_text(self, text: str, limit: int) -> str:
        clean = self._normalize_text_field(text)
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3].rstrip() + "..."

    def _normalize_score(self, value: Any, default: float) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return default
        if score > 1.0:
            if score <= 5.0:
                score = score / 5.0
            else:
                score = 1.0
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score
