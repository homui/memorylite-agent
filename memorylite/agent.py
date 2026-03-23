from __future__ import annotations

import math
import queue
import re
import sys
import threading
import time
from collections import OrderedDict
from typing import Any

from memorylite.archive import ArchiveStore
from memorylite.compiler import ContextCompiler
from memorylite.config import MemoryAgentConfig
from memorylite.llm.base import ChatModel, EmbeddingClient, JSONPromptClient, MemoryController
from memorylite.llm.model_controller import ModelMemoryController
from memorylite.retriever import MemoryRetriever
from memorylite.schema import ChatMessage, MemoryRecord, RecallDecision, RecallItem, RecallResult, TurnResult, now_ts
from memorylite.store import SQLiteStore


class MemoryAgent:
    def __init__(
        self,
        config: MemoryAgentConfig | None = None,
        controller: MemoryController | None = None,
        client: JSONPromptClient | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config or MemoryAgentConfig()
        self.embedding_client = embedding_client
        if self.config.semantic_search_enabled and self.embedding_client is None:
            raise ValueError("semantic_search_enabled=True requires an embedding_client.")
        self.store = SQLiteStore(self.config)
        self.archive = ArchiveStore(self.config.archive_path)
        if controller is not None:
            self.controller = controller
        elif client is not None:
            self.controller = ModelMemoryController(self.config, client)
        else:
            raise ValueError("MemoryAgent now requires a model client or a custom memory controller.")
        self.retriever = MemoryRetriever(self.store, self.config)
        self.compiler = ContextCompiler()
        self._recent_cache: OrderedDict[str, list[ChatMessage]] = OrderedDict()
        self._state_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._queue: queue.Queue[dict[str, Any] | None] | None = None
        self._worker: threading.Thread | None = None
        self._background_error_count = 0
        self._last_background_error: str | None = None
        self._turns_since_maintenance = 0
        if self.config.background_write:
            self._queue = queue.Queue()
            self._worker = threading.Thread(target=self._background_loop, daemon=True)
            self._worker.start()

    def run_turn(
        self,
        chat_model: ChatModel | Any,
        user_message: str,
        session_id: str,
        user_id: str | None = None,
        project_id: str | None = None,
        system_prompt: str = "",
        max_items: int | None = None,
        max_tokens: int | None = None,
        sync_remember: bool = False,
    ) -> TurnResult:
        recall = self.recall(
            query=user_message,
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            max_items=max_items,
            max_tokens=max_tokens,
        )
        assistant_message = chat_model.generate(
            system_prompt=system_prompt,
            memory_context=recall.compiled_text,
            user_message=user_message,
        )
        self.remember(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            user_message=user_message,
            assistant_message=assistant_message,
            sync=sync_remember,
        )
        return TurnResult(
            user_message=user_message,
            assistant_message=assistant_message,
            recall=recall,
        )

    def remember(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        user_id: str | None = None,
        project_id: str | None = None,
        sync: bool = False,
    ) -> dict[str, int]:
        created_at = now_ts()
        user_message_id, assistant_message_id = self._store_messages(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            created_at=created_at,
        )
        self._append_recent_message(session_id, "user", user_message, created_at, user_message_id)
        self._append_recent_message(session_id, "assistant", assistant_message, created_at, assistant_message_id)
        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "project_id": project_id,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "message_ids": [user_message_id, assistant_message_id],
        }
        if sync or not self._queue:
            self._postprocess_turn(payload)
        else:
            self._queue.put(payload)
        return {"user_message_id": user_message_id, "assistant_message_id": assistant_message_id}

    def recall(
        self,
        query: str,
        session_id: str,
        user_id: str | None = None,
        project_id: str | None = None,
        max_items: int | None = None,
        max_tokens: int | None = None,
    ) -> RecallResult:
        timings_ms: dict[str, float] = {}
        total_started = time.perf_counter()
        item_limit = max_items or self.config.max_recall_items

        preload_started = time.perf_counter()
        recent_messages = self._get_recent_messages(session_id)
        states = self._load_states(session_id, user_id, project_id)
        scope_ids = self._scope_ids(session_id, user_id, project_id)
        timings_ms["preload_ms"] = (time.perf_counter() - preload_started) * 1000.0

        candidate_started = time.perf_counter()
        candidates = self.retriever.collect_candidates(
            query=query,
            scope_ids=scope_ids,
            max_items=item_limit,
        )
        timings_ms["candidate_ms"] = (time.perf_counter() - candidate_started) * 1000.0

        timings_ms["semantic_ms"] = 0.0
        if self.config.semantic_search_enabled and candidates:
            semantic_started = time.perf_counter()
            candidates = self._semantic_rerank(query, candidates)
            timings_ms["semantic_ms"] = (time.perf_counter() - semantic_started) * 1000.0

        controller_started = time.perf_counter()
        if self.config.no_candidate_short_circuit_enabled and not candidates:
            decision = RecallDecision(
                should_recall=False,
                reason="no_candidate_short_circuit",
                selected_memory_ids=[],
            )
        else:
            decision = self.controller.decide_recall(
                query=query,
                recent_messages=recent_messages,
                states=states,
                candidates=candidates,
                max_items=item_limit,
            )
        timings_ms["controller_ms"] = (time.perf_counter() - controller_started) * 1000.0

        selected_items: list[RecallItem] = []
        if decision.should_recall and decision.selected_memory_ids:
            selected_id_set = set(decision.selected_memory_ids[:item_limit])
            selected_items = [item for item in candidates if item.memory_id in selected_id_set][:item_limit]
        elif decision.should_recall:
            selected_items = candidates[:item_limit]

        if not selected_items:
            selected_items, decision = self._apply_local_recall_fallback(query, candidates, item_limit, decision)

        compile_started = time.perf_counter()
        compiled = self.compiler.compile(
            query=query,
            recent_messages=recent_messages,
            states=states,
            recall_items=selected_items,
            max_tokens=max_tokens or self.config.max_context_tokens,
        )
        timings_ms["compile_ms"] = (time.perf_counter() - compile_started) * 1000.0
        timings_ms["total_ms"] = (time.perf_counter() - total_started) * 1000.0

        return RecallResult(
            query=query,
            compiled_text=compiled,
            triggered=decision.should_recall,
            reason=decision.reason,
            items=selected_items,
            recent_messages=recent_messages,
            states=states,
            timings_ms={key: round(value, 3) for key, value in timings_ms.items()},
        )

    def run_maintenance(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, int]:
        scope_ids: dict[str, str] = {}
        if session_id:
            scope_ids["session"] = session_id
        if user_id:
            scope_ids["user"] = user_id
        if project_id:
            scope_ids["project"] = project_id
        return self.store.run_maintenance(scope_ids)

    def get_state(self, scope: str, scope_id: str) -> dict[str, Any]:
        cache_key = (scope, scope_id)
        cached = self._state_cache.get(cache_key)
        if cached is not None:
            self._state_cache.move_to_end(cache_key)
            return dict(cached)
        state = self.store.get_state(scope, scope_id)
        self._set_state_cache(scope, scope_id, state)
        return dict(state)

    def flush(self) -> None:
        if not self._queue:
            return
        self._queue.join()

    def close(self) -> None:
        if self._queue:
            self._queue.put(None)
            self._queue.join()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)
        self.store.close()

    def _background_loop(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                try:
                    self._postprocess_turn(item)
                except Exception as exc:
                    self._background_error_count += 1
                    self._last_background_error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[memory][warning] background memory write failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
            finally:
                self._queue.task_done()

    def _postprocess_turn(self, payload: dict[str, Any]) -> None:
        session_id = payload["session_id"]
        user_id = payload.get("user_id")
        project_id = payload.get("project_id")
        recent_messages = self._get_recent_messages(session_id)
        scope_ids = self._scope_ids(session_id, user_id, project_id)
        existing_state = self.get_state("session", session_id)
        extraction = self.controller.extract_memories(
            session_id=session_id,
            user_message=payload["user_message"],
            assistant_message=payload["assistant_message"],
            recent_messages=recent_messages,
            existing_state=existing_state,
            scope_ids=scope_ids,
        )
        for memory in extraction.memories:
            memory.source_message_ids = list(payload["message_ids"])

        state_updates: list[tuple[str, str, dict[str, Any]]] = []
        if extraction.state_patch:
            state_updates.append(("session", session_id, extraction.state_patch))
            merged = dict(existing_state)
            merged.update(extraction.state_patch)
            self._set_state_cache("session", session_id, merged)

        memory_embeddings: list[list[float] | None] | None = None
        embedding_model = None
        if self.config.semantic_search_enabled and extraction.memories:
            memory_embeddings = self._embed_memory_records(extraction.memories)
            embedding_model = self._embedding_model_name()

        if extraction.memories or state_updates:
            self.store.persist_extraction(
                extraction.memories,
                state_updates,
                embeddings_model=embedding_model,
                memory_embeddings=memory_embeddings,
            )

        self._maybe_run_maintenance(scope_ids)

    def _scope_ids(
        self,
        session_id: str,
        user_id: str | None,
        project_id: str | None,
    ) -> dict[str, str]:
        scope_ids = {"session": session_id}
        if user_id:
            scope_ids["user"] = user_id
        if project_id:
            scope_ids["project"] = project_id
        return scope_ids

    def _load_states(
        self,
        session_id: str,
        user_id: str | None,
        project_id: str | None,
    ) -> dict[str, dict[str, Any]]:
        result = {"session": self.get_state("session", session_id)}
        if user_id:
            result["user"] = self.get_state("user", user_id)
        if project_id:
            result["project"] = self.get_state("project", project_id)
        return result

    def _store_messages(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        created_at: int,
    ) -> tuple[int, int]:
        messages: list[dict[str, Any]] = []
        for role, content in (("user", user_message), ("assistant", assistant_message)):
            archive_path: str | None = None
            archive_offset: int | None = None
            if self.config.auto_archive_messages:
                archive_path, archive_offset = self.archive.append_message(
                    session_id,
                    {
                        "session_id": session_id,
                        "role": role,
                        "content": content,
                        "created_at": created_at,
                    },
                )
            messages.append(
                {
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "created_at": created_at,
                    "archive_path": archive_path,
                    "archive_offset": archive_offset,
                }
            )
        ids = self.store.insert_messages(messages)
        return ids[0], ids[1]

    def _get_recent_messages(self, session_id: str) -> list[ChatMessage]:
        cached = self._recent_cache.get(session_id)
        if cached is not None:
            self._recent_cache.move_to_end(session_id)
            return list(cached)
        recent_messages = self.store.get_recent_messages(session_id, self.config.recent_window)
        self._recent_cache[session_id] = list(recent_messages)
        self._recent_cache.move_to_end(session_id)
        while len(self._recent_cache) > self.config.recent_cache_size:
            self._recent_cache.popitem(last=False)
        return list(recent_messages)

    def _append_recent_message(
        self,
        session_id: str,
        role: str,
        content: str,
        created_at: int,
        message_id: int,
    ) -> None:
        messages = list(self._recent_cache.get(session_id, []))
        messages.append(
            ChatMessage(
                role=role,
                content=content,
                created_at=created_at,
                message_id=message_id,
            )
        )
        self._recent_cache[session_id] = messages[-self.config.recent_window :]
        self._recent_cache.move_to_end(session_id)
        while len(self._recent_cache) > self.config.recent_cache_size:
            self._recent_cache.popitem(last=False)

    def _set_state_cache(self, scope: str, scope_id: str, value: dict[str, Any]) -> None:
        cache_key = (scope, scope_id)
        self._state_cache[cache_key] = dict(value)
        self._state_cache.move_to_end(cache_key)
        while len(self._state_cache) > self.config.state_cache_size:
            self._state_cache.popitem(last=False)

    def _maybe_run_maintenance(self, scope_ids: dict[str, str]) -> None:
        if not self.config.maintenance_enabled:
            return
        self._turns_since_maintenance += 1
        if self._turns_since_maintenance < max(1, self.config.maintenance_interval_turns):
            return
        self._turns_since_maintenance = 0
        try:
            self.store.run_maintenance(scope_ids)
        except Exception as exc:
            self._background_error_count += 1
            self._last_background_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[memory][warning] maintenance failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _apply_local_recall_fallback(
        self,
        query: str,
        candidates: list[RecallItem],
        item_limit: int,
        decision: RecallDecision,
    ) -> tuple[list[RecallItem], RecallDecision]:
        if not self.config.local_recall_fallback_enabled:
            return [], decision
        if not candidates:
            return [], decision

        query_terms = self._search_terms(query)
        strong_matches: list[RecallItem] = []
        for candidate in candidates:
            relevance = self._fallback_relevance(query_terms, query, candidate)
            if candidate.score >= self.config.local_recall_fallback_score_threshold or relevance >= 0.55:
                strong_matches.append(candidate)

        if not strong_matches:
            return [], decision

        fallback_limit = min(item_limit, max(1, self.config.local_recall_fallback_max_items))
        selected_items = strong_matches[:fallback_limit]
        fallback_reason = "local_score_fallback"
        if any(self._fallback_relevance(query_terms, query, item) >= 0.55 for item in selected_items):
            fallback_reason = "local_relevance_fallback"
        fallback_decision = RecallDecision(
            should_recall=True,
            reason=f"{decision.reason}|{fallback_reason}",
            selected_memory_ids=[item.memory_id for item in selected_items],
        )
        return selected_items, fallback_decision

    def _fallback_relevance(self, query_terms: list[str], query: str, item: RecallItem) -> float:
        source = " ".join(
            part for part in [item.summary, item.content, " ".join(item.tags), " ".join(item.entities)] if part
        ).lower()
        if not source:
            return 0.0
        normalized_query = query.strip().lower()
        if normalized_query and normalized_query in source:
            return 1.0
        if not query_terms:
            return 0.0
        source_terms = set(self._search_terms(source))
        if not source_terms:
            return 0.0
        overlap = len(set(query_terms) & source_terms) / max(1, min(len(set(query_terms)), len(source_terms)))
        return overlap

    def _search_terms(self, text: str) -> list[str]:
        cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
        latin_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,20}", text.lower())
        return list(dict.fromkeys(cjk_terms[:12] + latin_terms[:12]))

    def _semantic_rerank(self, query: str, candidates: list[RecallItem]) -> list[RecallItem]:
        if not self.embedding_client or not candidates:
            return candidates
        query_vector = self._get_query_embedding(query)
        if not query_vector:
            return candidates
        embedding_map = self.store.load_memory_embeddings(
            [item.memory_id for item in candidates],
            self._embedding_model_name(),
        )
        if not embedding_map:
            return candidates

        reranked: list[RecallItem] = []
        for item in candidates:
            vector = embedding_map.get(item.memory_id)
            if not vector:
                reranked.append(item)
                continue
            semantic_similarity = self._cosine_similarity(query_vector, vector)
            semantic_score = max(0.0, min(1.0, (semantic_similarity + 1.0) / 2.0))
            reranked.append(
                RecallItem(
                    memory_id=item.memory_id,
                    scope=item.scope,
                    scope_id=item.scope_id,
                    kind=item.kind,
                    content=item.content,
                    summary=item.summary,
                    score=item.score + (self.config.semantic_rerank_weight * semantic_score),
                    tags=list(item.tags),
                    entities=list(item.entities),
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    def _embed_memory_records(self, memories: list[MemoryRecord]) -> list[list[float] | None]:
        if not self.embedding_client or not memories:
            return [None for _ in memories]
        texts = [self._memory_embedding_text(memory) for memory in memories]
        try:
            vectors = self.embedding_client.embed_texts(texts, input_type="document")
        except Exception:
            return [None for _ in memories]
        result: list[list[float] | None] = []
        for index in range(len(memories)):
            if index < len(vectors):
                result.append([float(value) for value in vectors[index]])
            else:
                result.append(None)
        return result

    def _get_query_embedding(self, query: str) -> list[float]:
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            self._query_embedding_cache.move_to_end(query)
            return list(cached)
        if not self.embedding_client:
            return []
        try:
            vectors = self.embedding_client.embed_texts([query], input_type="query")
        except Exception:
            return []
        if not vectors:
            return []
        vector = [float(value) for value in vectors[0]]
        self._query_embedding_cache[query] = vector
        self._query_embedding_cache.move_to_end(query)
        while len(self._query_embedding_cache) > self.config.semantic_query_cache_size:
            self._query_embedding_cache.popitem(last=False)
        return list(vector)

    def _memory_embedding_text(self, memory: MemoryRecord) -> str:
        parts = [memory.kind, memory.summary or memory.content, " ".join(memory.tags), " ".join(memory.entities)]
        return "\n".join(part.strip() for part in parts if part and part.strip())

    def _embedding_model_name(self) -> str:
        if self.config.semantic_embedding_model:
            return self.config.semantic_embedding_model
        if self.embedding_client is not None:
            return getattr(self.embedding_client, "model", "default")
        return "default"

    def _cosine_similarity(self, lhs: list[float], rhs: list[float]) -> float:
        if not lhs or not rhs:
            return 0.0
        length = min(len(lhs), len(rhs))
        if length == 0:
            return 0.0
        dot = sum(lhs[index] * rhs[index] for index in range(length))
        lhs_norm = math.sqrt(sum(value * value for value in lhs[:length]))
        rhs_norm = math.sqrt(sum(value * value for value in rhs[:length]))
        if lhs_norm == 0.0 or rhs_norm == 0.0:
            return 0.0
        return dot / (lhs_norm * rhs_norm)
