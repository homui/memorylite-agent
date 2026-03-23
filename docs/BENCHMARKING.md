# Benchmarking & Evaluation Guide 📏📊

[中文版本 / Chinese Version](BENCHMARKING.zh-CN.md)

This document explains how to evaluate `memorylite`, how the built-in benchmark works, and how to construct useful datasets for memory-agent evaluation.

## 1. Why benchmark a memory agent?

A memory system can fail in different ways:

- memory is written incorrectly
- memory is stored in the wrong scope
- the right candidate is never retrieved
- the controller refuses to recall
- the memory is recalled but not used well by the final prompt

Because of that, memory evaluation should not be treated as a single score.

`memorylite` separates evaluation into multiple layers.

## 2. Core Metrics

### 2.1 Trigger Accuracy

Whether the system correctly decided:

- recall should happen
- or recall should not happen

### 2.2 Memory Hit Rate

Whether the selected memory items contain the expected gold memory content.

This measures:

- retrieval quality
- recall controller selection quality

### 2.3 Context Hit Rate

Whether the final compiled context contains the expected gold fact.

This measures:

- memory selection
- compiler quality
- whether the final prompt preserves useful information

### 2.4 Recall Latency

Total recall latency, plus stage timings:

- `preload_ms`
- `candidate_ms`
- `semantic_ms`
- `controller_ms`
- `compile_ms`

These stage metrics are critical for optimization work.

## 3. Built-in Benchmark Files

Main files:

- benchmark runner: [../benchmarks/run_benchmark.py](../benchmarks/run_benchmark.py)
- default dataset: [../benchmarks/datasets/memory_benchmark_v1.json](../benchmarks/datasets/memory_benchmark_v1.json)

## 4. How the Built-in Benchmark Works

For each sample:

1. replay the historical turns with `remember(sync=True)`
2. run `recall(...)` on the benchmark query
3. evaluate:
   - `triggered`
   - selected memory items
   - final compiled context
   - stage timings
   - memory count / db size

This means the benchmark exercises the real memory pipeline, not just one isolated retrieval function.

## 5. Dataset Format

Each sample in the dataset has a structure like:

```json
{
  "id": "stack_long_gap_fact",
  "category": "fact",
  "session_id": "stack-s1",
  "user_id": "user-stack-1",
  "turns": [
    {
      "user": "The project uses Python and SQLite.",
      "assistant": "Got it."
    }
  ],
  "query": "What stack does the project use?",
  "gold_memory_substrings": [
    "project uses python and sqlite"
  ],
  "gold_context_substrings": [
    "project uses python and sqlite"
  ],
  "expected_triggered": true
}
```

### Key fields

- `id`
- `category`
- `session_id`
- `query_session_id` (optional)
- `user_id`
- `turns`
- `query`
- `gold_memory_substrings`
- `gold_context_substrings`
- `expected_triggered`

## 6. Recommended Scenario Categories

Your benchmark suite should include at least:

- `preference`
- `fact`
- `task_state`
- `event`
- `conflict`
- `negative`

### Preference

Tests whether the system remembers user style, formatting, or behavior preferences.

### Fact

Tests stable factual memory such as project stack, identity, or constraints.

### Task State

Tests actionable state such as TODOs or work-in-progress tasks.

### Event

Tests temporal or episodic memory such as plans or past actions.

### Conflict

Tests whether new memory overrides old memory correctly.

### Negative

Tests whether the system avoids unnecessary recall.

## 7. How to Run

### Offline demo mode

```bash
python .\benchmarks\run_benchmark.py
```

### Qwen / DashScope mode

```bash
python .\benchmarks\run_benchmark.py --backend qwen
```

Optional semantic rerank:

```bash
python .\benchmarks\run_benchmark.py --backend qwen --semantic-real --embed-model text-embedding-v4
```

### OpenAI-compatible mode

```bash
python .\benchmarks\run_benchmark.py --backend openai_compat --model your-model-name --base-url http://127.0.0.1:8000/v1
```

### Debug model I/O

```bash
python .\benchmarks\run_benchmark.py --backend qwen --debug-model-io
```

This writes raw request/response logs to:

`benchmarks/.runs/<sample_id>/model_debug.jsonl`

## 8. How to Interpret Results

### Case A

- high `memory_hit_rate`
- low `context_hit_rate`

Meaning:

- retrieval worked
- compiler or prompt organization is weak

### Case B

- low `memory_hit_rate`
- low `context_hit_rate`

Meaning:

- the real issue is upstream
- writing, retrieval, or controller selection is unstable

### Case C

- high `trigger_accuracy`
- low `memory_hit_rate`

Meaning:

- the system knows recall should happen
- but selects the wrong items or writes poor memory

### Case D

- strong quality
- very high `controller_ms`

Meaning:

- recall quality is acceptable
- latency bottleneck is the memory model backend

## 9. Recommended Evaluation Strategy

Use two benchmark modes:

### 9.1 Regression benchmark

Use the built-in demo backend.

Goal:

- check whether code changes break the pipeline
- maintain stable local regression signals

### 9.2 Real-model benchmark

Use Qwen / DashScope or another OpenAI-compatible memory model.

Goal:

- evaluate real extraction stability
- evaluate real recall-controller behavior
- measure realistic latency

## 10. How to Build Better Datasets

### 10.1 Start small

Start with 20 to 50 hand-written samples.

### 10.2 Add long-gap samples

Insert many irrelevant turns between the original memory and the query.

### 10.3 Add conflict samples

Example:

- old: project uses SQLite
- new: project uses Postgres now

### 10.4 Add noise-heavy samples

Large numbers of similar but irrelevant memories are useful for testing ranking and selection.

### 10.5 Add cross-session samples

Write memory in one session and query in another session with the same `user_id`.

## 11. What to Improve First When Scores Are Bad

Recommended debugging order:

1. inspect memory writing
2. inspect memory scope
3. inspect candidate pool quality
4. inspect controller decisions
5. inspect compiler output

## 12. Practical Success Targets

Useful targets for a lightweight production-oriented memory layer:

- `trigger_accuracy >= 0.9`
- `memory_hit_rate >= 0.8`
- `context_hit_rate >= 0.8`
- bounded recall latency
- acceptable negative-sample latency
