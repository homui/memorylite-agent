# memorylite 🧠⚡

[中文文档 / Chinese README](README.zh-CN.md)

**Ultra-light local memory agent framework for Python LLM apps.**  
`memorylite` helps you add **long-term memory**, **fast recall**, **SQLite-first storage**, **OpenAI-compatible model integration**, and **lightweight retrieval** to chatbots, copilots, agents, and RAG-style applications without bringing in a heavy orchestration stack.

**Search-friendly keywords:** `memory agent`, `long-term memory`, `LLM memory`, `agent memory framework`, `SQLite memory`, `RAG`, `retrieval`, `semantic rerank`, `OpenAI-compatible`, `Qwen`, `local-first`, `Python agent framework`.

## Why memorylite? 🚀

Most memory systems for LLM apps become heavy very quickly:

- too many services
- too many dependencies
- too much orchestration
- too much latency

`memorylite` takes a different path:

- **SQLite-first** structured memory and state
- **Local JSONL archives** for raw transcripts
- **Small-model memory controller** for planning and writing
- **Fast lexical retrieval** with optional lightweight semantic rerank
- **OpenAI-compatible integration** for Qwen, DeepSeek, vLLM, LM Studio, Ollama gateways, and similar backends
- **Low-latency Python API** for real chat loops
- **Built-in maintenance** for dedupe, TTL pruning, and event compaction

If you are building a **chat memory layer**, **agent memory system**, **persistent conversation memory**, or **lightweight long-context helper** for Python, this project is designed for that exact use case.

## Feature Highlights ✨

- 🗂️ **Structured memory**: facts, preferences, events, summaries, task state
- ⚡ **Fast recall path**: caches + SQLite + tiny candidate pools
- 🤖 **Model-driven memory agent**: a small model always participates in memory planning
- 🔎 **Optional semantic rerank**: lightweight embedding support without a heavy vector database
- 🏠 **Local-first architecture**: works well with local files and simple deployments
- 🔌 **OpenAI-compatible backends**: easy integration with Qwen / DashScope, DeepSeek, and custom endpoints
- 📏 **Benchmarking support**: built-in benchmark script and dataset
- 📦 **Pip-installable library**: designed to be embedded into Python applications

## Documentation 📚

- [Chinese README](README.zh-CN.md)
- [Retrieval & Memory Architecture](docs/RETRIEVAL.md)
- [中文：检索与记忆架构](docs/RETRIEVAL.zh-CN.md)
- [Benchmarking & Evaluation Guide](docs/BENCHMARKING.md)
- [中文：评测与数据集说明](docs/BENCHMARKING.zh-CN.md)

## Install

```bash
pip install -e .
```

Install from another local Python project:

```bash
pip install /path/to/memory_agent
```

Install optional OpenAI-compatible example support:

```bash
pip install "memorylite[openai]"
```

Check the installed version:

```python
import memorylite
print(memorylite.__version__)
```

## Quick Start

```python
from memorylite import MemoryAgent, MemoryAgentConfig
from memorylite.llm import DemoMemoryModelClient

agent = MemoryAgent(
    MemoryAgentConfig(root_dir="./.memorylite"),
    client=DemoMemoryModelClient(),
)

agent.remember(
    session_id="demo-session",
    user_id="alice",
    user_message="Please remember that I prefer concise answers.",
    assistant_message="Understood. I will keep answers concise.",
)

recall = agent.recall(
    query="Do you remember my response preference?",
    session_id="demo-session",
    user_id="alice",
)

print(recall.compiled_text)
agent.close()
```

## One-Call Turn API

If you already have a chat model wrapper, `run_turn()` gives you a clean:

`recall -> chat -> remember`

pipeline in one call.

```python
turn = agent.run_turn(
    chat_model=chat_model,
    user_message="What stack did I say I want?",
    session_id="demo-session",
    user_id="alice",
    system_prompt="You are a helpful assistant.",
)

print(turn.assistant_message)
print(turn.recall.items)
```

## Supported Integration Patterns

`memorylite` is especially useful for:

- chatbots with persistent user memory
- coding copilots with project memory
- task-oriented assistants with task state
- lightweight agent frameworks
- local-first RAG and memory systems
- OpenAI-compatible LLM backends

Runnable examples:

- [examples/basic_usage.py](examples/basic_usage.py)
- [examples/chat_loop.py](examples/chat_loop.py)
- [examples/deepseek_chat.py](examples/deepseek_chat.py)
- [examples/openai_compatible_chat.py](examples/openai_compatible_chat.py)
- [examples/qwen_chat.py](examples/qwen_chat.py)

## Real Model Examples

Generic local example:

```bash
python examples/chat_loop.py
```

Ollama memory model:

```bash
set OLLAMA_MEMORY_MODEL=qwen2.5:3b-instruct
python examples/chat_loop.py
```

Ollama memory + chat:

```bash
set OLLAMA_MEMORY_MODEL=qwen2.5:3b-instruct
set OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct
python examples/chat_loop.py
```

DeepSeek for chat + memory:

```bash
set DEEPSEEK_API_KEY=your_key_here
python examples/deepseek_chat.py
```

OpenAI-compatible endpoint:

```bash
set OPENAI_COMPAT_BASE_URL=http://127.0.0.1:8000/v1
set OPENAI_COMPAT_MEMORY_MODEL=qwen2.5-3b-instruct
set OPENAI_COMPAT_CHAT_MODEL=qwen2.5-7b-instruct
python examples/openai_compatible_chat.py
```

Qwen / DashScope compatible mode:

```bash
set DASHSCOPE_API_KEY=your_key_here
set QWEN_CHAT_MODEL=qwen2.5-7b-instruct
set QWEN_MEMORY_MODEL=qwen2.5-3b-instruct
set QWEN_EMBED_MODEL=text-embedding-v4
python examples/qwen_chat.py
```

## Architecture Overview

The current design is a **lightweight memory agent**, not a heavy autonomous multi-step planner.

1. **Preload**: recent messages + compact state are loaded from cache / SQLite
2. **Candidate retrieval**: local lexical search gathers a very small candidate pool
3. **Semantic rerank** (optional): rerank only the small candidate pool
4. **Recall controller**: the small model decides whether memory is needed and selects item IDs
5. **Context compiler**: selected memories are compiled into prompt-ready context
6. **Writer**: after each turn, the small model extracts durable memories and state patches
7. **Maintenance**: dedupe, TTL pruning, and event compaction keep storage lightweight

For deeper details, see [docs/RETRIEVAL.md](docs/RETRIEVAL.md).

## Optional Semantic Retrieval

Semantic retrieval is optional and intentionally lightweight:

- no heavy vector database
- no full-library ANN stack
- only query embedding + small candidate rerank

```python
from memorylite import MemoryAgent, MemoryAgentConfig
from memorylite.llm import OpenAICompatibleEmbeddingClient, OllamaJSONClient

config = MemoryAgentConfig(
    root_dir="./.memorylite",
    semantic_search_enabled=True,
    semantic_embedding_model="BAAI/bge-small-zh-v1.5",
)

memory = MemoryAgent(
    config,
    client=OllamaJSONClient(model="qwen2.5:3b-instruct"),
    embedding_client=OpenAICompatibleEmbeddingClient(
        model="BAAI/bge-small-zh-v1.5",
        base_url="http://127.0.0.1:8000/v1",
    ),
)
```

## Benchmarking

The repository includes:

- a benchmark runner: [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py)
- a benchmark dataset: [benchmarks/datasets/memory_benchmark_v1.json](benchmarks/datasets/memory_benchmark_v1.json)

Run the default offline benchmark:

```bash
python .\benchmarks\run_benchmark.py
```

Run Qwen benchmark:

```bash
python .\benchmarks\run_benchmark.py --backend qwen
```

Detailed benchmark design, metrics, and dataset format are documented in [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## Main Components

- `MemoryAgent`: main Python entry point
- `ModelMemoryController`: small-model controller for recall decisions and memory writing
- `MemoryRetriever`: local candidate collector
- `SQLiteStore`: structured memories, states, messages, and optional embeddings
- `ArchiveStore`: raw JSONL message archives
- `ContextCompiler`: builds final prompt-ready context
- `OpenAICompatibleJSONClient`: OpenAI-style memory model adapter
- `OpenAICompatibleEmbeddingClient`: OpenAI-style embedding adapter
- `OllamaJSONClient`: thin adapter for local Ollama-compatible memory endpoints

## Design Principles

- **Use memory only when needed**
- **Prefer small candidate pools**
- **Keep storage local and inspectable**
- **Make the memory controller simple**
- **Keep the recall path bounded**
- **Treat long-term memory as infrastructure, not magic**

## Packaging & Publishing

Build:

```bash
python -m build
```

Install wheel locally:

```bash
pip install dist/memorylite-0.1.0-py3-none-any.whl
```

Upload:

```bash
python -m twine upload dist/*
```
