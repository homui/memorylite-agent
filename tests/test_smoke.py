import os
import shutil
import unittest

from memorylite import MemoryAgent, MemoryAgentConfig
from memorylite.compiler import ContextCompiler
from memorylite.llm import DemoMemoryModelClient
from memorylite.llm.model_controller import ModelMemoryController
from memorylite.schema import ChatMessage, MemoryRecord, RecallDecision, RecallItem, now_ts


class FakeEmbeddingClient:
    model = "fake-small"

    def embed_texts(self, texts, input_type="document"):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "python" in lowered else 0.0,
                    1.0 if "sqlite" in lowered or "database" in lowered else 0.0,
                    1.0 if "concise" in lowered else 0.0,
                ]
            )
        return vectors


class StaticJSONClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def complete_json(self, system_prompt: str, user_prompt: str):
        if not self.payloads:
            return {}
        return self.payloads.pop(0)


class ConservativeController:
    def decide_recall(self, query, recent_messages, states, candidates, max_items):
        return RecallDecision(should_recall=False, reason="model_said_no", selected_memory_ids=[])

    def extract_memories(self, session_id, user_message, assistant_message, recent_messages, existing_state, scope_ids):
        return type("ExtractionResultProxy", (), {"memories": [], "state_patch": {}})()


class SpyController(ConservativeController):
    def __init__(self):
        self.decide_calls = 0

    def decide_recall(self, query, recent_messages, states, candidates, max_items):
        self.decide_calls += 1
        return super().decide_recall(query, recent_messages, states, candidates, max_items)


class MemoryLiteSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = os.path.join(os.getcwd(), "memorylite-test-data")
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.makedirs(self.tmpdir, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_remember_and_recall(self) -> None:
        agent = MemoryAgent(
            MemoryAgentConfig(
                root_dir=self.tmpdir,
                background_write=False,
            ),
            client=DemoMemoryModelClient(),
        )
        agent.remember(
            session_id="s1",
            user_id="u1",
            user_message="Please remember that the project is a memory agent built with Python.",
            assistant_message="Understood. I will keep the project goal in memory.",
        )
        result = agent.recall(
            query="Do you remember what project I am building?",
            session_id="s1",
            user_id="u1",
        )
        agent.close()

        self.assertTrue(result.triggered)
        self.assertIn("memory agent", result.compiled_text.lower())
        self.assertIn("total_ms", result.timings_ms)

    def test_semantic_embeddings_are_persisted(self) -> None:
        agent = MemoryAgent(
            MemoryAgentConfig(
                root_dir=self.tmpdir,
                background_write=False,
                semantic_search_enabled=True,
                semantic_embedding_model="fake-small",
            ),
            client=DemoMemoryModelClient(),
            embedding_client=FakeEmbeddingClient(),
        )
        agent.remember(
            session_id="s2",
            user_id="u2",
            user_message="Please remember that I want Python with SQLite for local storage.",
            assistant_message="I will keep your Python and SQLite preference in memory.",
        )
        result = agent.recall(
            query="Which database stack did I prefer for local work?",
            session_id="s2",
            user_id="u2",
        )
        row = agent.store.conn.execute("SELECT COUNT(*) AS count FROM memory_embeddings").fetchone()
        agent.close()

        self.assertGreater(row["count"], 0)
        self.assertTrue(result.triggered)
        self.assertGreaterEqual(result.timings_ms.get("semantic_ms", 0.0), 0.0)

    def test_duplicate_memories_are_merged_on_write(self) -> None:
        agent = MemoryAgent(
            MemoryAgentConfig(
                root_dir=self.tmpdir,
                background_write=False,
                maintenance_enabled=False,
            ),
            client=DemoMemoryModelClient(),
        )
        for _ in range(2):
            agent.remember(
                session_id="s3",
                user_id="u3",
                user_message="Please remember that I prefer concise answers.",
                assistant_message="Understood. I will keep answers concise.",
            )
        count = agent.store.conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()["count"]
        agent.close()

        self.assertEqual(count, 2)

    def test_manual_maintenance_compacts_and_prunes(self) -> None:
        agent = MemoryAgent(
            MemoryAgentConfig(
                root_dir=self.tmpdir,
                background_write=False,
                maintenance_enabled=False,
                keep_recent_event_memories=2,
                compact_event_batch_size=3,
            ),
            client=DemoMemoryModelClient(),
        )
        for index in range(6):
            agent.store.persist_extraction(
                [
                    MemoryRecord(
                        scope="session",
                        scope_id="s4",
                        kind="event",
                        content=f"Event number {index}",
                        summary=f"Event number {index}",
                    )
                ],
                [],
            )
        expired = MemoryRecord(
            scope="session",
            scope_id="s4",
            kind="event",
            content="Expired event",
            summary="Expired event",
            created_at=now_ts() - 100,
            expires_at=now_ts() - 10,
        )
        agent.store.persist_extraction([expired], [])
        stats = agent.run_maintenance(session_id="s4")
        event_count = agent.store.conn.execute(
            "SELECT COUNT(*) AS count FROM memories WHERE scope = 'session' AND scope_id = 's4' AND kind = 'event'"
        ).fetchone()["count"]
        summary_count = agent.store.conn.execute(
            "SELECT COUNT(*) AS count FROM memories WHERE scope = 'session' AND scope_id = 's4' AND kind = 'summary'"
        ).fetchone()["count"]
        agent.close()

        self.assertGreaterEqual(stats["expired_deleted"], 1)
        self.assertGreaterEqual(stats["compacted_events"], 3)
        self.assertGreaterEqual(summary_count, 1)
        self.assertLess(event_count, 6)

    def test_controller_normalizes_real_ids_in_scope_key(self) -> None:
        controller = ModelMemoryController(
            MemoryAgentConfig(),
            StaticJSONClient(
                [
                    {
                        "state_patch": {},
                        "memories": [
                            {
                                "scope": "user",
                                "scope_id_key": "user-pref-1",
                                "kind": "preference",
                                "content": "The user prefers concise answers.",
                                "summary": "User prefers concise answers.",
                                "tags": ["concise"],
                                "importance": 3,
                                "confidence": 5,
                            }
                        ],
                    }
                ]
            ),
        )
        extraction = controller.extract_memories(
            session_id="s5",
            user_message="Please remember that I prefer concise answers.",
            assistant_message="Understood.",
            recent_messages=[
                ChatMessage(role="user", content="Please remember that I prefer concise answers."),
                ChatMessage(role="assistant", content="Understood."),
            ],
            existing_state={},
            scope_ids={"session": "s5", "user": "u5"},
        )

        self.assertEqual(len(extraction.memories), 1)
        self.assertEqual(extraction.memories[0].scope, "user")
        self.assertEqual(extraction.memories[0].scope_id, "u5")
        self.assertAlmostEqual(extraction.memories[0].importance, 0.6)
        self.assertAlmostEqual(extraction.memories[0].confidence, 1.0)

    def test_controller_falls_back_when_project_scope_is_unavailable(self) -> None:
        controller = ModelMemoryController(
            MemoryAgentConfig(),
            StaticJSONClient(
                [
                    {
                        "state_patch": {},
                        "memories": [
                            {
                                "scope": "project",
                                "scope_id_key": "session",
                                "kind": "fact",
                                "content": "The project uses Python and SQLite.",
                                "summary": "Project stack is Python and SQLite.",
                                "tags": ["python", "sqlite"],
                                "importance": 4,
                                "confidence": 4,
                            }
                        ],
                    }
                ]
            ),
        )
        extraction = controller.extract_memories(
            session_id="s6",
            user_message="The project uses Python and SQLite.",
            assistant_message="Got it.",
            recent_messages=[
                ChatMessage(role="user", content="The project uses Python and SQLite."),
                ChatMessage(role="assistant", content="Got it."),
            ],
            existing_state={},
            scope_ids={"session": "s6", "user": "u6"},
        )

        self.assertEqual(len(extraction.memories), 1)
        self.assertEqual(extraction.memories[0].scope, "session")
        self.assertEqual(extraction.memories[0].scope_id, "s6")

    def test_controller_strips_wrapped_content_fields(self) -> None:
        controller = ModelMemoryController(
            MemoryAgentConfig(),
            StaticJSONClient(
                [
                    {
                        "state_patch": {},
                        "memories": [
                            {
                                "scope": "user",
                                "scope_id_key": "user_stack_1",
                                "kind": "task",
                                "content": "user_message='There is a lot to organize.'",
                                "summary": "assistant_message='There is a lot to organize.'",
                                "tags": ["organization"],
                                "importance": 2,
                                "confidence": 3,
                            }
                        ],
                    }
                ]
            ),
        )
        extraction = controller.extract_memories(
            session_id="s7",
            user_message="There is a lot to organize.",
            assistant_message="Noted.",
            recent_messages=[
                ChatMessage(role="user", content="There is a lot to organize."),
                ChatMessage(role="assistant", content="Noted."),
            ],
            existing_state={},
            scope_ids={"session": "s7", "user": "u7"},
        )

        self.assertEqual(len(extraction.memories), 1)
        self.assertEqual(extraction.memories[0].scope, "user")
        self.assertEqual(extraction.memories[0].scope_id, "u7")
        self.assertEqual(extraction.memories[0].kind, "task_state")
        self.assertEqual(extraction.memories[0].content, "There is a lot to organize.")
        self.assertEqual(extraction.memories[0].summary, "There is a lot to organize.")

    def test_local_recall_fallback_uses_high_score_candidates(self) -> None:
        agent = MemoryAgent(
            MemoryAgentConfig(
                root_dir=self.tmpdir,
                background_write=False,
                local_recall_fallback_enabled=True,
                local_recall_fallback_score_threshold=0.8,
                local_recall_fallback_max_items=1,
            ),
            controller=ConservativeController(),
        )
        agent.store.persist_extraction(
            [
                MemoryRecord(
                    scope="session",
                    scope_id="s8",
                    kind="fact",
                    content="The project uses Python and SQLite.",
                    summary="Project stack is Python and SQLite.",
                    tags=["python", "sqlite", "project"],
                    importance=0.95,
                    confidence=0.95,
                )
            ],
            [],
        )
        result = agent.recall(
            query="What stack does the project use?",
            session_id="s8",
        )
        agent.close()

        self.assertTrue(result.triggered)
        self.assertEqual(len(result.items), 1)
        self.assertIn("local_score_fallback", result.reason)
        self.assertIn("python", result.compiled_text.lower())

    def test_no_candidate_short_circuit_skips_controller(self) -> None:
        controller = SpyController()
        agent = MemoryAgent(
            MemoryAgentConfig(
                root_dir=self.tmpdir,
                background_write=False,
                no_candidate_short_circuit_enabled=True,
            ),
            controller=controller,
        )
        result = agent.recall(
            query="What did I say before?",
            session_id="s9",
        )
        agent.close()

        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "no_candidate_short_circuit")
        self.assertEqual(controller.decide_calls, 0)
        self.assertLess(result.timings_ms.get("controller_ms", 0.0), 0.05)

    def test_context_compiler_groups_memory_sections(self) -> None:
        compiler = ContextCompiler()
        compiled = compiler.compile(
            query="What should you remember?",
            recent_messages=[ChatMessage(role="user", content="keep it concise")],
            states={"session": {"todo": "review draft"}},
            recall_items=[
                RecallItem(
                    memory_id=1,
                    scope="user",
                    scope_id="u1",
                    kind="preference",
                    content="The user prefers concise answers.",
                    summary="User prefers concise answers.",
                    score=1.2,
                ),
                RecallItem(
                    memory_id=2,
                    scope="session",
                    scope_id="s1",
                    kind="task_state",
                    content="Review the draft today.",
                    summary="Need to review the draft today.",
                    score=1.1,
                ),
                RecallItem(
                    memory_id=3,
                    scope="session",
                    scope_id="s1",
                    kind="fact",
                    content="The stack is Python and SQLite.",
                    summary="Project stack is Python and SQLite.",
                    score=1.0,
                ),
            ],
            max_tokens=1200,
        )

        self.assertIn("[Important Preferences]", compiled)
        self.assertIn("[Relevant Task State]", compiled)
        self.assertIn("[Relevant Memory]", compiled)
        self.assertIn("[Memory Guidance]", compiled)
        self.assertIn("original: The stack is Python and SQLite.", compiled)

    def test_context_compiler_prefers_original_content_for_literal_matching(self) -> None:
        compiler = ContextCompiler()
        compiled = compiler.compile(
            query="What style do I prefer for answers?",
            recent_messages=[],
            states={},
            recall_items=[
                RecallItem(
                    memory_id=4,
                    scope="user",
                    scope_id="u2",
                    kind="preference",
                    content="Please remember that I prefer bullet points and concise answers.",
                    summary="User prefers structured concise answers.",
                    score=1.3,
                )
            ],
            max_tokens=800,
        )

        self.assertIn("prefer bullet points and concise answers", compiled.lower())
        self.assertIn("summary: User prefers structured concise answers.", compiled)

    def test_local_recall_fallback_uses_relevance_overlap(self) -> None:
        agent = MemoryAgent(
            MemoryAgentConfig(
                root_dir=self.tmpdir,
                background_write=False,
                local_recall_fallback_enabled=True,
                local_recall_fallback_score_threshold=2.0,
                local_recall_fallback_max_items=1,
            ),
            controller=ConservativeController(),
        )
        agent.store.persist_extraction(
            [
                MemoryRecord(
                    scope="session",
                    scope_id="s10",
                    kind="event",
                    content="On Monday I will have dinner with friends.",
                    summary="Monday plan is dinner with friends.",
                    tags=["monday", "dinner", "friends"],
                    importance=0.4,
                    confidence=0.9,
                )
            ],
            [],
        )
        result = agent.recall(
            query="What will I do on Monday?",
            session_id="s10",
        )
        agent.close()

        self.assertTrue(result.triggered)
        self.assertEqual(len(result.items), 1)
        self.assertIn("local_relevance_fallback", result.reason)
        self.assertIn("monday", result.compiled_text.lower())


if __name__ == "__main__":
    unittest.main()
