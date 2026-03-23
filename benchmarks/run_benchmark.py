from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memorylite import MemoryAgent, MemoryAgentConfig
from memorylite.llm import (
    DemoMemoryModelClient,
    OpenAICompatibleEmbeddingClient,
    OpenAICompatibleJSONClient,
)


DEFAULT_DATASET = PROJECT_ROOT / "benchmarks" / "datasets" / "memory_benchmark_v1.json"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen2.5-7b-instruct"
DEFAULT_QWEN_EMBED_MODEL = "text-embedding-v4"


@dataclass(slots=True)
class SampleResult:
    sample_id: str
    category: str
    recall_ms: float
    preload_ms: float
    candidate_ms: float
    semantic_ms: float
    controller_ms: float
    compile_ms: float
    triggered: bool
    expected_triggered: bool | None
    trigger_correct: bool | None
    memory_hit: bool | None
    context_hit: bool | None
    item_count: int
    recent_count: int
    memory_count: int
    db_size_bytes: int


@dataclass(slots=True)
class BackendSettings:
    backend: str
    model: str
    base_url: str | None
    timeout_seconds: float
    semantic_mode: str
    embedding_model: str | None
    embedding_base_url: str | None
    api_key_env: str | None


class FakeEmbeddingClient:
    model = "benchmark-fake-embedding"

    def embed_texts(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "python" in lowered else 0.0,
                    1.0 if "sqlite" in lowered else 0.0,
                    1.0 if "postgres" in lowered else 0.0,
                    1.0 if "prefer" in lowered or "concise" in lowered else 0.0,
                    1.0 if "todo" in lowered else 0.0,
                    1.0 if "monday" in lowered else 0.0,
                ]
            )
        return vectors


class LoggingOpenAICompatibleJSONClient(OpenAICompatibleJSONClient):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 45.0,
        debug_path: Path | None = None,
        sample_id: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        self.debug_path = debug_path
        self.sample_id = sample_id

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        raw_response_body = ""
        raw_content: Any = None
        event: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "sample_id": self.sample_id,
            "phase": self._infer_phase(system_prompt),
            "model": self.model,
            "base_url": self.base_url,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "request_payload": payload,
        }

        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response_body = response.read().decode("utf-8")
            event["raw_response_body"] = raw_response_body
            parsed_response = json.loads(raw_response_body)
            raw_content = parsed_response["choices"][0]["message"]["content"]
            event["raw_content"] = raw_content
            parsed_payload = self._parse_json_payload(raw_content)
            event["parsed_payload"] = parsed_payload
            self._write_debug_event(event)
            return parsed_payload
        except Exception as exc:
            event["raw_response_body"] = raw_response_body
            event["raw_content"] = raw_content
            event["error_type"] = type(exc).__name__
            event["error"] = str(exc)
            self._write_debug_event(event)
            raise

    def _infer_phase(self, system_prompt: str) -> str:
        lowered = system_prompt.lower()
        if "memory recall controller" in lowered:
            return "decide_recall"
        if "memory-writing agent" in lowered:
            return "extract_memories"
        return "unknown"

    def _write_debug_event(self, event: dict[str, Any]) -> None:
        if not self.debug_path:
            return
        self.debug_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    lowered = text.lower()
    compact = re.sub(r"\s+", " ", lowered).strip()
    return compact


def any_gold_match(texts: list[str], gold_substrings: list[str]) -> bool | None:
    if not gold_substrings:
        return None
    normalized_texts = [normalize(text) for text in texts if text]
    normalized_gold = [normalize(item) for item in gold_substrings if item]
    for gold in normalized_gold:
        for text in normalized_texts:
            if gold in text:
                return True
    return False


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_backend_settings(args: argparse.Namespace) -> BackendSettings:
    backend = args.backend
    if backend == "qwen":
        return BackendSettings(
            backend="qwen",
            model=args.model or os.getenv("QWEN_BENCHMARK_MODEL") or os.getenv("QWEN_MEMORY_MODEL") or DEFAULT_QWEN_MODEL,
            base_url=args.base_url or os.getenv("QWEN_BASE_URL") or DEFAULT_QWEN_BASE_URL,
            timeout_seconds=args.timeout_seconds or float(os.getenv("QWEN_MEMORY_TIMEOUT_SECONDS", "60")),
            semantic_mode="real" if args.semantic_real else ("fake" if args.semantic_fake else "off"),
            embedding_model=args.embed_model or os.getenv("QWEN_EMBED_MODEL") or DEFAULT_QWEN_EMBED_MODEL,
            embedding_base_url=args.embed_base_url or os.getenv("QWEN_EMBED_BASE_URL") or args.base_url or os.getenv("QWEN_BASE_URL") or DEFAULT_QWEN_BASE_URL,
            api_key_env=args.api_key_env or "DASHSCOPE_API_KEY",
        )
    if backend == "openai_compat":
        return BackendSettings(
            backend="openai_compat",
            model=args.model or os.getenv("OPENAI_COMPAT_BENCHMARK_MODEL") or DEFAULT_QWEN_MODEL,
            base_url=args.base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or DEFAULT_QWEN_BASE_URL,
            timeout_seconds=args.timeout_seconds or float(os.getenv("OPENAI_COMPAT_MEMORY_TIMEOUT_SECONDS", "60")),
            semantic_mode="real" if args.semantic_real else ("fake" if args.semantic_fake else "off"),
            embedding_model=args.embed_model or os.getenv("OPENAI_COMPAT_EMBED_MODEL") or DEFAULT_QWEN_EMBED_MODEL,
            embedding_base_url=args.embed_base_url or os.getenv("OPENAI_COMPAT_EMBED_BASE_URL") or args.base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or DEFAULT_QWEN_BASE_URL,
            api_key_env=args.api_key_env or "OPENAI_API_KEY",
        )
    return BackendSettings(
        backend="demo",
        model="demo-memory-model",
        base_url=None,
        timeout_seconds=0.0,
        semantic_mode="fake" if args.semantic_fake else "off",
        embedding_model="benchmark-fake-embedding" if args.semantic_fake else None,
        embedding_base_url=None,
        api_key_env=None,
    )


def build_agent(
    sample_root: Path,
    sample_id: str,
    settings: BackendSettings,
    args: argparse.Namespace,
) -> MemoryAgent:
    semantic_enabled = settings.semantic_mode in {"fake", "real"}
    embedding_client: Any = None

    if settings.backend == "demo":
        client = DemoMemoryModelClient()
        if settings.semantic_mode == "fake":
            embedding_client = FakeEmbeddingClient()
    else:
        api_key = args.api_key or os.getenv(settings.api_key_env or "")
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set {settings.api_key_env} or pass --api-key when using backend={settings.backend}."
            )
        client_cls = LoggingOpenAICompatibleJSONClient if args.debug_model_io else OpenAICompatibleJSONClient
        client_kwargs = {
            "model": settings.model,
            "base_url": settings.base_url or DEFAULT_QWEN_BASE_URL,
            "api_key": api_key,
            "timeout_seconds": settings.timeout_seconds,
        }
        if args.debug_model_io:
            client_kwargs["debug_path"] = sample_root / "model_debug.jsonl"
            client_kwargs["sample_id"] = sample_id
        client = client_cls(**client_kwargs)
        if settings.semantic_mode == "real":
            embedding_client = OpenAICompatibleEmbeddingClient(
                model=settings.embedding_model or DEFAULT_QWEN_EMBED_MODEL,
                base_url=settings.embedding_base_url or settings.base_url or DEFAULT_QWEN_BASE_URL,
                api_key=args.embed_api_key or api_key,
            )
        elif settings.semantic_mode == "fake":
            embedding_client = FakeEmbeddingClient()

    config = MemoryAgentConfig(
        root_dir=sample_root,
        background_write=False,
        semantic_search_enabled=semantic_enabled,
        semantic_embedding_model=settings.embedding_model if settings.semantic_mode == "real" else ("benchmark-fake-embedding" if settings.semantic_mode == "fake" else None),
        maintenance_enabled=False,
    )
    return MemoryAgent(config=config, client=client, embedding_client=embedding_client)


def _timing(recall: Any, name: str) -> float:
    return float((getattr(recall, "timings_ms", {}) or {}).get(name, 0.0))


def run_sample(
    sample: dict[str, Any],
    runs_root: Path,
    settings: BackendSettings,
    args: argparse.Namespace,
) -> SampleResult:
    sample_root = runs_root / sample["id"]
    shutil.rmtree(sample_root, ignore_errors=True)
    sample_root.mkdir(parents=True, exist_ok=True)

    agent = build_agent(sample_root, sample_id=sample["id"], settings=settings, args=args)
    session_id = sample["session_id"]
    query_session_id = sample.get("query_session_id", session_id)
    user_id = sample.get("user_id")
    project_id = sample.get("project_id")

    try:
        for turn in sample.get("turns", []):
            agent.remember(
                session_id=session_id,
                user_id=user_id,
                project_id=project_id,
                user_message=turn["user"],
                assistant_message=turn.get("assistant", ""),
                sync=True,
            )

        started = time.perf_counter()
        recall = agent.recall(
            query=sample["query"],
            session_id=query_session_id,
            user_id=user_id,
            project_id=project_id,
        )
        recall_ms = (time.perf_counter() - started) * 1000.0

        item_texts: list[str] = []
        for item in recall.items:
            item_texts.append(item.summary or "")
            item_texts.append(item.content or "")
        context_texts = [recall.compiled_text]
        memory_hit = any_gold_match(item_texts, sample.get("gold_memory_substrings", []))
        context_hit = any_gold_match(
            context_texts,
            sample.get("gold_context_substrings", sample.get("gold_memory_substrings", [])),
        )
        expected_triggered = sample.get("expected_triggered")
        trigger_correct = None if expected_triggered is None else (bool(expected_triggered) == recall.triggered)
        memory_count = int(agent.store.conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()["count"])
        db_size_bytes = sample_root.joinpath("memory.sqlite3").stat().st_size

        return SampleResult(
            sample_id=sample["id"],
            category=sample["category"],
            recall_ms=recall_ms,
            preload_ms=_timing(recall, "preload_ms"),
            candidate_ms=_timing(recall, "candidate_ms"),
            semantic_ms=_timing(recall, "semantic_ms"),
            controller_ms=_timing(recall, "controller_ms"),
            compile_ms=_timing(recall, "compile_ms"),
            triggered=recall.triggered,
            expected_triggered=expected_triggered,
            trigger_correct=trigger_correct,
            memory_hit=memory_hit,
            context_hit=context_hit,
            item_count=len(recall.items),
            recent_count=len(recall.recent_messages),
            memory_count=memory_count,
            db_size_bytes=db_size_bytes,
        )
    finally:
        agent.close()


def _mean_attr(results: list[SampleResult], name: str) -> float:
    return round(statistics.mean(getattr(item, name) for item in results), 2) if results else 0.0


def summarize_results(results: list[SampleResult]) -> dict[str, Any]:
    recall_latencies = [item.recall_ms for item in results]
    trigger_scores = [item.trigger_correct for item in results if item.trigger_correct is not None]
    memory_scores = [item.memory_hit for item in results if item.memory_hit is not None]
    context_scores = [item.context_hit for item in results if item.context_hit is not None]

    by_category: dict[str, list[SampleResult]] = {}
    for item in results:
        by_category.setdefault(item.category, []).append(item)

    category_summary: dict[str, Any] = {}
    for category, items in by_category.items():
        memory_denom = sum(sample.memory_hit is not None for sample in items)
        context_denom = sum(sample.context_hit is not None for sample in items)
        category_summary[category] = {
            "samples": len(items),
            "avg_recall_ms": round(statistics.mean(sample.recall_ms for sample in items), 2),
            "memory_hit_rate": round(
                sum(bool(sample.memory_hit) for sample in items if sample.memory_hit is not None) / max(1, memory_denom),
                3,
            ),
            "context_hit_rate": round(
                sum(bool(sample.context_hit) for sample in items if sample.context_hit is not None) / max(1, context_denom),
                3,
            ),
        }

    return {
        "samples": len(results),
        "avg_recall_ms": round(statistics.mean(recall_latencies), 2) if recall_latencies else 0.0,
        "p95_recall_ms": round(percentile(recall_latencies, 0.95), 2) if recall_latencies else 0.0,
        "avg_preload_ms": _mean_attr(results, "preload_ms"),
        "avg_candidate_ms": _mean_attr(results, "candidate_ms"),
        "avg_semantic_ms": _mean_attr(results, "semantic_ms"),
        "avg_controller_ms": _mean_attr(results, "controller_ms"),
        "avg_compile_ms": _mean_attr(results, "compile_ms"),
        "p95_controller_ms": round(percentile([item.controller_ms for item in results], 0.95), 2) if results else 0.0,
        "p95_semantic_ms": round(percentile([item.semantic_ms for item in results], 0.95), 2) if results else 0.0,
        "trigger_accuracy": round(sum(bool(item) for item in trigger_scores) / len(trigger_scores), 3) if trigger_scores else None,
        "memory_hit_rate": round(sum(bool(item) for item in memory_scores) / len(memory_scores), 3) if memory_scores else None,
        "context_hit_rate": round(sum(bool(item) for item in context_scores) / len(context_scores), 3) if context_scores else None,
        "avg_memory_count": round(statistics.mean(item.memory_count for item in results), 2) if results else 0.0,
        "avg_db_size_kb": round(statistics.mean(item.db_size_bytes for item in results) / 1024.0, 2) if results else 0.0,
        "by_category": category_summary,
    }


def print_report(
    results: list[SampleResult],
    summary: dict[str, Any],
    settings: BackendSettings,
    dataset_name: str,
    runs_root: Path,
    debug_model_io: bool,
) -> None:
    print("=" * 96)
    print("memorylite benchmark")
    print("=" * 96)
    print(
        "backend: "
        f"name={settings.backend} "
        f"model={settings.model} "
        f"semantic={settings.semantic_mode} "
        f"dataset={dataset_name}"
    )
    if settings.base_url:
        print(f"endpoint: base_url={settings.base_url}")
    if settings.semantic_mode == "real" and settings.embedding_model:
        print(
            "embedding: "
            f"model={settings.embedding_model} "
            f"base_url={settings.embedding_base_url}"
        )
    if debug_model_io:
        print(f"debug: model_io_logs={runs_root}")
    print(
        "overall: "
        f"samples={summary['samples']} "
        f"avg_recall_ms={summary['avg_recall_ms']} "
        f"p95_recall_ms={summary['p95_recall_ms']} "
        f"trigger_accuracy={summary['trigger_accuracy']} "
        f"memory_hit_rate={summary['memory_hit_rate']} "
        f"context_hit_rate={summary['context_hit_rate']}"
    )
    print(
        "stages: "
        f"avg_preload_ms={summary['avg_preload_ms']} "
        f"avg_candidate_ms={summary['avg_candidate_ms']} "
        f"avg_semantic_ms={summary['avg_semantic_ms']} "
        f"avg_controller_ms={summary['avg_controller_ms']} "
        f"avg_compile_ms={summary['avg_compile_ms']} "
        f"p95_controller_ms={summary['p95_controller_ms']} "
        f"p95_semantic_ms={summary['p95_semantic_ms']}"
    )
    print(
        "storage: "
        f"avg_memory_count={summary['avg_memory_count']} "
        f"avg_db_size_kb={summary['avg_db_size_kb']}"
    )
    print("-" * 96)
    print(
        f"{'sample_id':28} {'cat':10} {'ms':>8} {'ctrl':>8} {'sem':>8} {'cand':>8} "
        f"{'trig':>6} {'items':>6} {'recent':>6}"
    )
    for item in results:
        print(
            f"{item.sample_id[:28]:28} {item.category[:10]:10} {item.recall_ms:8.2f} "
            f"{item.controller_ms:8.2f} {item.semantic_ms:8.2f} {item.candidate_ms:8.2f} "
            f"{str(item.triggered):>6} {item.item_count:6d} {item.recent_count:6d}"
        )
    print("-" * 96)
    print("by_category:")
    for category, payload in summary["by_category"].items():
        print(
            f"  {category:12} "
            f"samples={payload['samples']} "
            f"avg_recall_ms={payload['avg_recall_ms']} "
            f"memory_hit_rate={payload['memory_hit_rate']} "
            f"context_hit_rate={payload['context_hit_rate']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a benchmark for memorylite with demo or real OpenAI-compatible memory models.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to a benchmark dataset JSON file.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / ".runs",
        help="Directory for temporary benchmark databases.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write the raw benchmark report JSON.")
    parser.add_argument(
        "--backend",
        choices=("demo", "qwen", "openai_compat"),
        default="demo",
        help="Memory-model backend. Use 'qwen' for DashScope Qwen models.",
    )
    parser.add_argument("--model", type=str, default=None, help="Memory model name. For Qwen benchmark the default is qwen2.5-7b-instruct.")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL for the memory model.")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the memory model backend.")
    parser.add_argument("--api-key-env", type=str, default=None, help="Environment variable name to read the API key from.")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Timeout in seconds for the memory model JSON call.")
    parser.add_argument("--semantic-fake", action="store_true", help="Enable deterministic fake embeddings to exercise the semantic rerank path.")
    parser.add_argument("--semantic-real", action="store_true", help="Enable a real embedding model through an OpenAI-compatible embeddings endpoint.")
    parser.add_argument("--embed-model", type=str, default=None, help="Embedding model to use when --semantic-real is enabled.")
    parser.add_argument("--embed-base-url", type=str, default=None, help="Embeddings endpoint base URL when --semantic-real is enabled.")
    parser.add_argument("--embed-api-key", type=str, default=None, help="Optional API key override for the embedding endpoint.")
    parser.add_argument(
        "--debug-model-io",
        action="store_true",
        help="Write raw memory-model request/response logs to <runs-root>/<sample_id>/model_debug.jsonl.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.semantic_fake and args.semantic_real:
        parser.error("--semantic-fake and --semantic-real cannot be used together.")

    dataset = load_dataset(args.dataset)
    settings = resolve_backend_settings(args)
    args.runs_root.mkdir(parents=True, exist_ok=True)

    results = [run_sample(sample, args.runs_root, settings=settings, args=args) for sample in dataset["samples"]]
    summary = summarize_results(results)
    print_report(
        results,
        summary,
        settings=settings,
        dataset_name=dataset["name"],
        runs_root=args.runs_root,
        debug_model_io=args.debug_model_io,
    )

    if args.json_out:
        report = {
            "dataset": dataset["name"],
            "backend": asdict(settings),
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
