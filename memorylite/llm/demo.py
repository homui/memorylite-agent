from __future__ import annotations

import ast
import re
from typing import Any


class DemoMemoryModelClient:
    """
    Deterministic stand-in for tests and local examples.
    It simulates a small memory model without pulling remote dependencies.
    """

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        prompt = system_prompt.lower()
        if "memory recall controller" in prompt:
            candidates = self._extract_python_value(user_prompt, "candidates") or []
            selected_ids = [item["id"] for item in candidates[:3] if "id" in item]
            return {
                "should_recall": bool(selected_ids),
                "reason": "demo_recall",
                "selected_memory_ids": selected_ids,
            }
        if "memory-writing agent" in prompt:
            user_message = self._extract_single_value(user_prompt, "user_message")
            memories = []
            tags = self._extract_tags(user_message)
            if user_message:
                memories.append(
                    {
                        "scope": "session",
                        "scope_id_key": "session",
                        "kind": "event",
                        "content": user_message[:400],
                        "summary": user_message[:140],
                        "tags": tags,
                        "importance": 0.5,
                        "confidence": 0.8,
                    }
                )
            lowered = user_message.lower()
            if "prefer" in lowered or "偏好" in user_message:
                memories.append(
                    {
                        "scope": "user",
                        "scope_id_key": "user",
                        "kind": "preference",
                        "content": user_message[:300],
                        "summary": user_message[:120],
                        "tags": tags,
                        "importance": 0.82,
                        "confidence": 0.86,
                    }
                )
            if "project" in lowered or "python" in lowered or "sqlite" in lowered or "memory agent" in lowered:
                memories.append(
                    {
                        "scope": "session",
                        "scope_id_key": "session",
                        "kind": "fact",
                        "content": user_message[:300],
                        "summary": user_message[:120],
                        "tags": tags,
                        "importance": 0.72,
                        "confidence": 0.8,
                    }
                )
            state_patch: dict[str, Any] = {}
            if "todo:" in lowered:
                state_patch["todo"] = user_message.split(":", 1)[-1].strip()[:200]
            return {"state_patch": state_patch, "memories": memories[:2]}
        return {}

    def _extract_single_value(self, text: str, key: str) -> str:
        match = re.search(rf"{key}=(.+)", text)
        if not match:
            return ""
        raw = match.group(1).splitlines()[0].strip()
        try:
            value = ast.literal_eval(raw)
        except Exception:
            value = raw.strip("'\"")
        return str(value)

    def _extract_python_value(self, text: str, key: str) -> Any:
        match = re.search(rf"{key}=(.+)", text, re.DOTALL)
        if not match:
            return None
        raw = match.group(1).strip()
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None

    def _extract_tags(self, text: str) -> list[str]:
        terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,20}", text.lower())
        return list(dict.fromkeys(terms[:8]))
