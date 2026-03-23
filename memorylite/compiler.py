from __future__ import annotations

import re
from typing import Any

from memorylite.schema import ChatMessage, RecallItem


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ContextCompiler:
    def compile(
        self,
        query: str,
        recent_messages: list[ChatMessage],
        states: dict[str, dict[str, Any]],
        recall_items: list[RecallItem],
        max_tokens: int,
    ) -> str:
        sections: list[str] = []
        used_tokens = 0

        guidance_block = (
            "[Memory Guidance]\n"
            "- Prefer the highest-scored memory lines first.\n"
            "- Treat the original memory statement as the most literal source of truth.\n"
            "- If memory and recent conversation disagree, prefer the more recent and higher-scored fact."
        )
        used_tokens += estimate_tokens(guidance_block)
        sections.append(guidance_block)

        state_lines: list[str] = []
        for scope_name, payload in states.items():
            if not payload:
                continue
            for key, value in list(payload.items())[:8]:
                state_lines.append(f"- {scope_name}.{key}: {value}")
        if state_lines:
            block = "[Active State]\n" + "\n".join(state_lines)
            used_tokens += estimate_tokens(block)
            sections.append(block)

        deduped_items = self._dedupe_items(recall_items)
        preference_items = [item for item in deduped_items if item.kind == "preference"]
        task_items = [item for item in deduped_items if item.kind == "task_state"]
        other_items = [item for item in deduped_items if item.kind not in {"preference", "task_state"}]

        preference_block, used_tokens = self._build_memory_block(
            title="[Important Preferences]",
            items=preference_items,
            used_tokens=used_tokens,
            max_tokens=max_tokens,
        )
        if preference_block:
            sections.append(preference_block)

        task_block, used_tokens = self._build_memory_block(
            title="[Relevant Task State]",
            items=task_items,
            used_tokens=used_tokens,
            max_tokens=max_tokens,
        )
        if task_block:
            sections.append(task_block)

        other_block, used_tokens = self._build_memory_block(
            title="[Relevant Memory]",
            items=other_items,
            used_tokens=used_tokens,
            max_tokens=max_tokens,
        )
        if other_block:
            sections.append(other_block)

        recent_lines = []
        for message in recent_messages[-8:]:
            line = f"- {message.role}: {self._trim(message.content, 240)}"
            projected = used_tokens + estimate_tokens("\n".join(recent_lines + [line]))
            if projected > max_tokens:
                break
            recent_lines.append(line)
        if recent_lines:
            recent_block = "[Recent Conversation]\n" + "\n".join(recent_lines)
            used_tokens += estimate_tokens(recent_block)
            sections.append(recent_block)

        sections.append(f"[Current Query]\n- user: {query}")
        return "\n\n".join(sections)

    def _build_memory_block(
        self,
        title: str,
        items: list[RecallItem],
        used_tokens: int,
        max_tokens: int,
    ) -> tuple[str, int]:
        if not items:
            return "", used_tokens
        lines: list[str] = []
        for item in items:
            for line in self._memory_lines(item):
                projected = used_tokens + estimate_tokens("\n".join(lines + [line]))
                if projected > max_tokens:
                    block = title + "\n" + "\n".join(lines) if lines else ""
                    return block, used_tokens + (estimate_tokens(block) if block else 0)
                lines.append(line)
        if not lines:
            return "", used_tokens
        block = title + "\n" + "\n".join(lines)
        used_tokens += estimate_tokens(block)
        return block, used_tokens

    def _memory_lines(self, item: RecallItem) -> list[str]:
        lines: list[str] = []
        header = f"- {item.kind.replace('_', ' ')} | {item.scope} | score={item.score:.2f}"
        lines.append(header)

        canonical = self._canonical_statement(item)
        if canonical:
            lines.append(f"  original: {canonical}")

        summary = self._normalize_inline(item.summary)
        if summary and self._normalize_text(summary) != self._normalize_text(canonical):
            lines.append(f"  summary: {self._trim(summary, 220)}")

        metadata = self._metadata_suffix(item)
        if metadata:
            lines.append(f"  metadata: {metadata}")
        return lines

    def _canonical_statement(self, item: RecallItem) -> str:
        content = self._normalize_inline(item.content)
        summary = self._normalize_inline(item.summary)
        if content:
            return self._trim(content, 260)
        return self._trim(summary, 260)

    def _metadata_suffix(self, item: RecallItem) -> str:
        parts: list[str] = []
        if item.tags:
            parts.append("tags=" + ", ".join(item.tags[:4]))
        if item.entities:
            parts.append("entities=" + ", ".join(item.entities[:4]))
        return " | ".join(parts)

    def _dedupe_items(self, items: list[RecallItem]) -> list[RecallItem]:
        seen: set[str] = set()
        result: list[RecallItem] = []
        for item in sorted(items, key=lambda current: current.score, reverse=True):
            key = self._normalize_text(" | ".join([item.kind, item.summary or "", item.content or ""]))
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _normalize_inline(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _trim(self, text: str, limit: int) -> str:
        text = self._normalize_inline(text)
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."
