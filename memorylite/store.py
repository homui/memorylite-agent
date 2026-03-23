from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from typing import Any

from memorylite.config import MemoryAgentConfig
from memorylite.schema import ChatMessage, MemoryRecord, RecallItem


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


class SQLiteStore:
    def __init__(self, config: MemoryAgentConfig) -> None:
        self.config = config
        config.root_path.mkdir(parents=True, exist_ok=True)
        config.database_path.parent.mkdir(parents=True, exist_ok=True)
        if not config.database_path.exists():
            with open(config.database_path, "ab"):
                pass
        self.conn = sqlite3.connect(str(config.database_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    def _configure(self) -> None:
        if self.config.wal_mode:
            self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute(f"PRAGMA cache_size=-{self.config.sqlite_cache_kb};")
        self.conn.execute(f"PRAGMA mmap_size={self.config.sqlite_mmap_bytes};")

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                archive_path TEXT,
                archive_offset INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_created
            ON messages(session_id, created_at DESC, id DESC);

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                scope TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                tags TEXT,
                entities TEXT,
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.8,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_accessed_at INTEGER,
                expires_at INTEGER,
                supersedes_memory_id INTEGER,
                source_message_ids TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_scope
            ON memories(scope, scope_id, kind, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memories_expiry
            ON memories(expires_at);

            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(memory_id, model),
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model
            ON memory_embeddings(model, memory_id);

            CREATE TABLE IF NOT EXISTS states (
                scope TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(scope, scope_id)
            );
            """
        )
        if self.config.enable_fts:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    summary,
                    tags,
                    entities
                );
                """
            )
        self.conn.commit()

    def insert_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        cursor = self.conn.cursor()
        ids: list[int] = []
        cursor.execute("BEGIN")
        for message in messages:
            cursor.execute(
                """
                INSERT INTO messages(session_id, role, content, created_at, archive_path, archive_offset)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    message["session_id"],
                    message["role"],
                    message["content"],
                    message["created_at"],
                    message.get("archive_path"),
                    message.get("archive_offset"),
                ),
            )
            ids.append(int(cursor.lastrowid))
        self.conn.commit()
        return ids

    def get_recent_messages(self, session_id: str, limit: int) -> list[ChatMessage]:
        rows = self.conn.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        messages = [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
                message_id=row["id"],
            )
            for row in rows
        ]
        messages.reverse()
        return messages

    def get_state(self, scope: str, scope_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT state_json FROM states WHERE scope = ? AND scope_id = ?",
            (scope, scope_id),
        ).fetchone()
        if not row:
            return {}
        return _json_loads(row["state_json"], {})

    def persist_extraction(
        self,
        memories: list[MemoryRecord],
        state_updates: list[tuple[str, str, dict[str, Any]]],
        embeddings_model: str | None = None,
        memory_embeddings: list[list[float] | None] | None = None,
    ) -> None:
        cursor = self.conn.cursor()
        current_states = {(scope, scope_id): self.get_state(scope, scope_id) for scope, scope_id, _ in state_updates}
        cursor.execute("BEGIN")
        for index, memory in enumerate(memories):
            self._apply_memory_defaults(memory)
            embedding_vector = None
            if memory_embeddings and index < len(memory_embeddings):
                embedding_vector = memory_embeddings[index]
            duplicate_row = self._find_duplicate_memory(cursor, memory)
            if duplicate_row is not None:
                memory_id = self._update_duplicate_memory(cursor, duplicate_row, memory, embeddings_model, embedding_vector)
            else:
                memory_id = self._insert_memory(cursor, memory, embeddings_model, embedding_vector)
            memory.memory_id = memory_id
        for scope, scope_id, patch in state_updates:
            current = dict(current_states[(scope, scope_id)])
            current.update(patch)
            cursor.execute(
                """
                INSERT INTO states(scope, scope_id, state_json, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(scope, scope_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (scope, scope_id, _json_dumps(current), int(time.time())),
            )
        self.conn.commit()

    def load_memory_embeddings(self, memory_ids: list[int], model: str) -> dict[int, list[float]]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self.conn.execute(
            f"""
            SELECT memory_id, vector_json
            FROM memory_embeddings
            WHERE model = ?
              AND memory_id IN ({placeholders})
            """,
            [model, *memory_ids],
        ).fetchall()
        result: dict[int, list[float]] = {}
        for row in rows:
            vector = _json_loads(row["vector_json"], [])
            if isinstance(vector, list):
                result[int(row["memory_id"])] = [float(value) for value in vector]
        return result

    def run_maintenance(self, scope_ids: dict[str, str] | None = None) -> dict[str, int]:
        scope_filters = list(scope_ids.items()) if scope_ids else []
        expired_deleted = self.prune_expired_memories()
        duplicate_merged = self.merge_duplicate_memories(scope_filters)
        compacted_events = self.compact_old_events(scope_filters)
        return {
            "expired_deleted": expired_deleted,
            "duplicate_merged": duplicate_merged,
            "compacted_events": compacted_events,
        }

    def prune_expired_memories(self) -> int:
        now = int(time.time())
        rows = self.conn.execute(
            """
            SELECT id
            FROM memories
            WHERE expires_at IS NOT NULL
              AND expires_at <= ?
            """,
            (now,),
        ).fetchall()
        memory_ids = [int(row["id"]) for row in rows]
        if not memory_ids:
            return 0
        self._delete_memories(memory_ids)
        return len(memory_ids)

    def merge_duplicate_memories(self, scope_filters: list[tuple[str, str]]) -> int:
        if not scope_filters:
            return 0
        merged_groups = 0
        cursor = self.conn.cursor()
        cursor.execute("BEGIN")
        try:
            for scope, scope_id in scope_filters:
                rows = self.conn.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE scope = ?
                      AND scope_id = ?
                      AND kind IN ('event', 'fact', 'preference', 'task_state')
                      AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY updated_at DESC, id DESC
                    """,
                    (scope, scope_id, int(time.time())),
                ).fetchall()
                groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
                for row in rows:
                    normalized = self._normalized_memory_text(row["summary"] or row["content"])
                    if not normalized:
                        continue
                    groups.setdefault((row["kind"], normalized), []).append(row)
                for group_rows in groups.values():
                    if len(group_rows) < 2:
                        continue
                    keep_row = group_rows[0]
                    merged_groups += 1
                    self._merge_memory_group(cursor, keep_row, group_rows[1:])
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return merged_groups

    def compact_old_events(self, scope_filters: list[tuple[str, str]]) -> int:
        if not scope_filters:
            return 0
        compacted = 0
        cursor = self.conn.cursor()
        cursor.execute("BEGIN")
        try:
            for scope, scope_id in scope_filters:
                rows = self.conn.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE scope = ?
                      AND scope_id = ?
                      AND kind = 'event'
                      AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY updated_at DESC, id DESC
                    """,
                    (scope, scope_id, int(time.time())),
                ).fetchall()
                threshold = self.config.keep_recent_event_memories + self.config.compact_event_batch_size
                if len(rows) <= threshold:
                    continue
                start = self.config.keep_recent_event_memories
                end = start + self.config.compact_event_batch_size
                to_compact = rows[start:end]
                if len(to_compact) < 2:
                    continue
                summary_memory = self._build_compaction_summary(scope, scope_id, to_compact)
                self._insert_memory(cursor, summary_memory, None, None)
                self._delete_memories_with_cursor(cursor, [int(row["id"]) for row in to_compact])
                compacted += len(to_compact)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return compacted

    def search_memories(
        self,
        query: str,
        scope_filters: list[tuple[str, str]],
        target_kinds: list[str] | None,
        limit: int,
    ) -> list[RecallItem]:
        candidates = self._load_candidates(query, scope_filters, target_kinds, max(limit * 2, 12))
        scored = self._score_candidates(query, candidates)
        top = sorted(scored, key=lambda item: item.score, reverse=True)[:limit]
        if top:
            self.touch_memories([item.memory_id for item in top])
        return top

    def _insert_memory(
        self,
        cursor: sqlite3.Cursor,
        memory: MemoryRecord,
        embeddings_model: str | None,
        embedding_vector: list[float] | None,
    ) -> int:
        cursor.execute(
            """
            INSERT INTO memories(
                scope, scope_id, kind, content, summary, tags, entities,
                importance, confidence, created_at, updated_at, last_accessed_at,
                expires_at, supersedes_memory_id, source_message_ids
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.scope,
                memory.scope_id,
                memory.kind,
                memory.content,
                memory.summary,
                _json_dumps(memory.tags),
                _json_dumps(memory.entities),
                memory.importance,
                memory.confidence,
                memory.created_at,
                memory.updated_at,
                memory.last_accessed_at,
                memory.expires_at,
                memory.supersedes_memory_id,
                _json_dumps(memory.source_message_ids),
            ),
        )
        memory_id = int(cursor.lastrowid)
        self._upsert_fts_row(cursor, memory_id, memory.content, memory.summary, memory.tags, memory.entities)
        self._upsert_embedding_row(cursor, memory_id, embeddings_model, embedding_vector)
        return memory_id

    def _find_duplicate_memory(self, cursor: sqlite3.Cursor, memory: MemoryRecord) -> sqlite3.Row | None:
        normalized = self._normalized_memory_text(memory.summary or memory.content)
        if not normalized or memory.kind == "summary":
            return None
        rows = cursor.execute(
            """
            SELECT *
            FROM memories
            WHERE scope = ?
              AND scope_id = ?
              AND kind = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT 16
            """,
            (memory.scope, memory.scope_id, memory.kind),
        ).fetchall()
        for row in rows:
            if self._normalized_memory_text(row["summary"] or row["content"]) == normalized:
                return row
        return None

    def _update_duplicate_memory(
        self,
        cursor: sqlite3.Cursor,
        existing_row: sqlite3.Row,
        memory: MemoryRecord,
        embeddings_model: str | None,
        embedding_vector: list[float] | None,
    ) -> int:
        memory_id = int(existing_row["id"])
        existing_tags = _json_loads(existing_row["tags"], [])
        existing_entities = _json_loads(existing_row["entities"], [])
        existing_sources = _json_loads(existing_row["source_message_ids"], [])
        merged_tags = self._merge_unique(existing_tags, memory.tags)
        merged_entities = self._merge_unique(existing_entities, memory.entities)
        merged_sources = self._merge_unique(existing_sources, memory.source_message_ids)
        content = self._prefer_richer_text(existing_row["content"], memory.content)
        summary = self._prefer_richer_text(existing_row["summary"] or "", memory.summary or "")
        updated_at = max(int(existing_row["updated_at"]), memory.updated_at, int(time.time()))
        expires_at = self._max_optional_int(existing_row["expires_at"], memory.expires_at)
        cursor.execute(
            """
            UPDATE memories
            SET content = ?,
                summary = ?,
                tags = ?,
                entities = ?,
                importance = ?,
                confidence = ?,
                updated_at = ?,
                expires_at = ?,
                source_message_ids = ?
            WHERE id = ?
            """,
            (
                content,
                summary,
                _json_dumps(merged_tags),
                _json_dumps(merged_entities),
                max(float(existing_row["importance"]), memory.importance),
                max(float(existing_row["confidence"]), memory.confidence),
                updated_at,
                expires_at,
                _json_dumps(merged_sources),
                memory_id,
            ),
        )
        self._upsert_fts_row(cursor, memory_id, content, summary, merged_tags, merged_entities)
        self._upsert_embedding_row(cursor, memory_id, embeddings_model, embedding_vector)
        return memory_id

    def _merge_memory_group(
        self,
        cursor: sqlite3.Cursor,
        keep_row: sqlite3.Row,
        duplicate_rows: list[sqlite3.Row],
    ) -> None:
        keep_id = int(keep_row["id"])
        tags = _json_loads(keep_row["tags"], [])
        entities = _json_loads(keep_row["entities"], [])
        sources = _json_loads(keep_row["source_message_ids"], [])
        content = keep_row["content"]
        summary = keep_row["summary"] or ""
        importance = float(keep_row["importance"])
        confidence = float(keep_row["confidence"])
        expires_at = keep_row["expires_at"]
        duplicate_ids: list[int] = []
        for row in duplicate_rows:
            duplicate_ids.append(int(row["id"]))
            tags = self._merge_unique(tags, _json_loads(row["tags"], []))
            entities = self._merge_unique(entities, _json_loads(row["entities"], []))
            sources = self._merge_unique(sources, _json_loads(row["source_message_ids"], []))
            content = self._prefer_richer_text(content, row["content"])
            summary = self._prefer_richer_text(summary, row["summary"] or "")
            importance = max(importance, float(row["importance"]))
            confidence = max(confidence, float(row["confidence"]))
            expires_at = self._max_optional_int(expires_at, row["expires_at"])
        cursor.execute(
            """
            UPDATE memories
            SET content = ?,
                summary = ?,
                tags = ?,
                entities = ?,
                importance = ?,
                confidence = ?,
                updated_at = ?,
                expires_at = ?,
                source_message_ids = ?
            WHERE id = ?
            """,
            (
                content,
                summary,
                _json_dumps(tags),
                _json_dumps(entities),
                importance,
                confidence,
                int(time.time()),
                expires_at,
                _json_dumps(sources),
                keep_id,
            ),
        )
        self._upsert_fts_row(cursor, keep_id, content, summary, tags, entities)
        self._delete_memories_with_cursor(cursor, duplicate_ids)

    def _build_compaction_summary(
        self,
        scope: str,
        scope_id: str,
        rows: list[sqlite3.Row],
    ) -> MemoryRecord:
        snippets: list[str] = []
        tags: list[str] = []
        entities: list[str] = []
        source_ids: list[int] = []
        oldest_created = int(time.time())
        for row in rows:
            snippet = (row["summary"] or row["content"]).strip().replace("\n", " ")
            if snippet:
                snippets.append(snippet[:120])
            tags = self._merge_unique(tags, _json_loads(row["tags"], []))
            entities = self._merge_unique(entities, _json_loads(row["entities"], []))
            source_ids = self._merge_unique(source_ids, _json_loads(row["source_message_ids"], []))
            oldest_created = min(oldest_created, int(row["created_at"]))
        joined = "\n".join(f"- {snippet}" for snippet in snippets[: self.config.compact_event_batch_size])
        content = f"Compacted older events:\n{joined}".strip()[: self.config.summary_max_chars]
        summary = f"Compacted {len(rows)} older events"
        return MemoryRecord(
            scope=scope,
            scope_id=scope_id,
            kind="summary",
            content=content,
            summary=summary,
            tags=tags[:12],
            entities=entities[:12],
            importance=0.46,
            confidence=0.92,
            created_at=oldest_created,
            updated_at=int(time.time()),
            source_message_ids=source_ids[:32],
        )

    def _load_candidates(
        self,
        query: str,
        scope_filters: list[tuple[str, str]],
        target_kinds: list[str] | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        clauses = []
        params: list[Any] = []
        for scope, scope_id in scope_filters:
            clauses.append("(scope = ? AND scope_id = ?)")
            params.extend([scope, scope_id])
        if not clauses:
            return []
        kind_clause = ""
        if target_kinds:
            placeholders = ",".join("?" for _ in target_kinds)
            kind_clause = f" AND kind IN ({placeholders})"
            params.extend(target_kinds)
        base_sql = f"""
            SELECT *
            FROM memories
            WHERE ({' OR '.join(clauses)})
              AND (expires_at IS NULL OR expires_at > ?)
              {kind_clause}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        base_params = [*params, int(time.time()), limit]
        rows = list(self.conn.execute(base_sql, base_params).fetchall())
        seen = {row["id"] for row in rows}
        for extra_rows in (
            self._load_fts_candidates(query, scope_filters, target_kinds, limit),
            self._load_exact_candidates(query, scope_filters, target_kinds, limit),
            self._load_token_overlap_candidates(query, scope_filters, target_kinds, limit),
        ):
            for row in extra_rows:
                if row["id"] not in seen:
                    rows.append(row)
                    seen.add(row["id"])
        return rows

    def _load_fts_candidates(
        self,
        query: str,
        scope_filters: list[tuple[str, str]],
        target_kinds: list[str] | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        if not self.config.enable_fts:
            return []
        keywords = self._search_terms(query)
        if not keywords:
            return []
        fts_query = " OR ".join(f'\"{term}\"' for term in keywords[:6])
        clauses = []
        params: list[Any] = [fts_query]
        for scope, scope_id in scope_filters:
            clauses.append("(m.scope = ? AND m.scope_id = ?)")
            params.extend([scope, scope_id])
        kind_clause = ""
        if target_kinds:
            placeholders = ",".join("?" for _ in target_kinds)
            kind_clause = f" AND m.kind IN ({placeholders})"
            params.extend(target_kinds)
        sql = f"""
            SELECT m.*
            FROM memories_fts f
            JOIN memories m ON m.id = f.rowid
            WHERE f.memories_fts MATCH ?
              AND ({' OR '.join(clauses)})
              AND (m.expires_at IS NULL OR m.expires_at > {int(time.time())})
              {kind_clause}
            LIMIT ?
        """
        params.append(limit)
        return list(self.conn.execute(sql, params).fetchall())

    def _load_exact_candidates(
        self,
        query: str,
        scope_filters: list[tuple[str, str]],
        target_kinds: list[str] | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        normalized_query = query.strip().lower()
        if len(normalized_query) < 3:
            return []
        clauses = []
        params: list[Any] = []
        for scope, scope_id in scope_filters:
            clauses.append("(scope = ? AND scope_id = ?)")
            params.extend([scope, scope_id])
        text_clause = "(lower(content) LIKE ? OR lower(summary) LIKE ? OR lower(tags) LIKE ? OR lower(entities) LIKE ?)"
        params.extend([f"%{normalized_query}%"] * 4)
        kind_clause = ""
        if target_kinds:
            placeholders = ",".join("?" for _ in target_kinds)
            kind_clause = f" AND kind IN ({placeholders})"
            params.extend(target_kinds)
        sql = f"""
            SELECT *
            FROM memories
            WHERE ({' OR '.join(clauses)})
              AND {text_clause}
              AND (expires_at IS NULL OR expires_at > {int(time.time())})
              {kind_clause}
            LIMIT ?
        """
        params.append(limit)
        return list(self.conn.execute(sql, params).fetchall())

    def _load_token_overlap_candidates(
        self,
        query: str,
        scope_filters: list[tuple[str, str]],
        target_kinds: list[str] | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        terms = self._search_terms(query)
        if not terms:
            return []
        clauses = []
        params: list[Any] = []
        for scope, scope_id in scope_filters:
            clauses.append("(scope = ? AND scope_id = ?)")
            params.extend([scope, scope_id])
        overlap_clauses = []
        for term in terms[:4]:
            overlap_clauses.append("(lower(content) LIKE ? OR lower(summary) LIKE ? OR lower(tags) LIKE ? OR lower(entities) LIKE ?)")
            params.extend([f"%{term.lower()}%"] * 4)
        kind_clause = ""
        if target_kinds:
            placeholders = ",".join("?" for _ in target_kinds)
            kind_clause = f" AND kind IN ({placeholders})"
            params.extend(target_kinds)
        sql = f"""
            SELECT *
            FROM memories
            WHERE ({' OR '.join(clauses)})
              AND ({' OR '.join(overlap_clauses)})
              AND (expires_at IS NULL OR expires_at > {int(time.time())})
              {kind_clause}
            LIMIT ?
        """
        params.append(limit)
        return list(self.conn.execute(sql, params).fetchall())

    def _score_candidates(self, query: str, rows: list[sqlite3.Row]) -> list[RecallItem]:
        query_terms = set(self._search_terms(query))
        query_ngrams = set(self._char_ngrams(query))
        normalized_query = query.strip().lower()
        now = int(time.time())
        scored: list[RecallItem] = []
        for row in rows:
            summary = row["summary"] or ""
            content = row["content"]
            tags = _json_loads(row["tags"], [])
            entities = _json_loads(row["entities"], [])
            text = " ".join(
                str(part)
                for part in [content, summary, " ".join(tags), " ".join(entities)]
                if part
            )
            text_lower = text.lower()
            candidate_terms = set(self._search_terms(text))
            candidate_ngrams = set(self._char_ngrams(text))
            tags_lower = {str(tag).lower() for tag in tags}
            entities_lower = {str(entity).lower() for entity in entities}
            term_overlap = self._overlap_score(query_terms, candidate_terms)
            ngram_overlap = self._overlap_score(query_ngrams, candidate_ngrams)
            exact_bonus = 1.0 if normalized_query and normalized_query in text_lower else 0.0
            tag_bonus = 1.0 if any(term in tags_lower for term in query_terms) else 0.0
            entity_bonus = 1.0 if any(term in entities_lower for term in query_terms) else 0.0
            recency_days = max(0.0, (now - row["updated_at"]) / 86400.0)
            recency_score = math.exp(-recency_days / 14.0)
            scope_bonus = 0.12 if row["scope"] == "session" else (0.06 if row["scope"] == "user" else 0.0)
            score = (
                0.34 * term_overlap
                + 0.18 * ngram_overlap
                + 0.14 * min(1.0, exact_bonus + tag_bonus * 0.5 + entity_bonus * 0.5)
                + self.config.recall_importance_weight * float(row["importance"])
                + self.config.recall_recency_weight * recency_score
                + scope_bonus
            )
            scored.append(
                RecallItem(
                    memory_id=int(row["id"]),
                    scope=row["scope"],
                    scope_id=row["scope_id"],
                    kind=row["kind"],
                    content=content,
                    summary=summary,
                    score=score,
                    tags=tags,
                    entities=entities,
                )
            )
        return scored

    def _apply_memory_defaults(self, memory: MemoryRecord) -> None:
        if memory.expires_at is not None:
            return
        if memory.kind == "event":
            memory.expires_at = memory.created_at + (self.config.event_ttl_days * 86400)
        elif memory.kind == "task_state":
            memory.expires_at = memory.created_at + (self.config.task_state_ttl_days * 86400)

    def _upsert_fts_row(self, cursor: sqlite3.Cursor, memory_id: int, content: str, summary: str, tags: list[str], entities: list[str]) -> None:
        if not self.config.enable_fts:
            return
        cursor.execute("DELETE FROM memories_fts WHERE rowid = ?", (memory_id,))
        cursor.execute(
            """
            INSERT INTO memories_fts(rowid, content, summary, tags, entities)
            VALUES(?, ?, ?, ?, ?)
            """,
            (memory_id, content, summary, " ".join(str(tag) for tag in tags), " ".join(str(entity) for entity in entities)),
        )

    def _upsert_embedding_row(
        self,
        cursor: sqlite3.Cursor,
        memory_id: int,
        embeddings_model: str | None,
        embedding_vector: list[float] | None,
    ) -> None:
        if not embeddings_model or not embedding_vector:
            return
        cursor.execute(
            """
            INSERT INTO memory_embeddings(memory_id, model, vector_json, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(memory_id, model) DO UPDATE SET
                vector_json = excluded.vector_json,
                updated_at = excluded.updated_at
            """,
            (memory_id, embeddings_model, _json_dumps(embedding_vector), int(time.time())),
        )

    def _delete_memories(self, memory_ids: list[int]) -> None:
        cursor = self.conn.cursor()
        cursor.execute("BEGIN")
        self._delete_memories_with_cursor(cursor, memory_ids)
        self.conn.commit()

    def _delete_memories_with_cursor(self, cursor: sqlite3.Cursor, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        cursor.execute(f"DELETE FROM memory_embeddings WHERE memory_id IN ({placeholders})", memory_ids)
        if self.config.enable_fts:
            cursor.execute(f"DELETE FROM memories_fts WHERE rowid IN ({placeholders})", memory_ids)
        cursor.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", memory_ids)

    def _normalized_memory_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _prefer_richer_text(self, current: str, incoming: str) -> str:
        current = (current or "").strip()
        incoming = (incoming or "").strip()
        if not incoming:
            return current
        if not current:
            return incoming
        if incoming in current:
            return current
        if current in incoming:
            return incoming
        return incoming if len(incoming) > len(current) else current

    def _merge_unique(self, left: list[Any], right: list[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for value in [*left, *right]:
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    def _max_optional_int(self, lhs: Any, rhs: Any) -> int | None:
        values = [int(value) for value in (lhs, rhs) if value is not None]
        return max(values) if values else None

    def _search_terms(self, text: str) -> list[str]:
        cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
        latin_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,20}", text.lower())
        return list(dict.fromkeys(cjk_terms[:12] + latin_terms[:12]))

    def _char_ngrams(self, text: str, n: int = 2) -> list[str]:
        compact = re.sub(r"\s+", "", text.lower())
        if len(compact) < n:
            return [compact] if compact else []
        return [compact[i : i + n] for i in range(len(compact) - n + 1)]

    def _overlap_score(self, lhs: set[str], rhs: set[str]) -> float:
        if not lhs or not rhs:
            return 0.0
        intersection = len(lhs & rhs)
        if intersection == 0:
            return 0.0
        return intersection / max(1, min(len(lhs), len(rhs)))

    def touch_memories(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        self.conn.execute(
            f"UPDATE memories SET last_accessed_at = ? WHERE id IN ({placeholders})",
            [int(time.time()), *memory_ids],
        )
        self.conn.commit()
