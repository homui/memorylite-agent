# memorylite 🧠⚡

[English README](README.md)

**面向 Python LLM 应用的超轻量、本地优先 memory agent 框架。**  
`memorylite` 可以帮助你在聊天机器人、Copilot、Agent、RAG 应用里增加：

- 长期记忆
- 快速召回
- SQLite 优先的本地存储
- OpenAI 协议兼容模型接入
- 轻量检索与可选语义重排

## 为什么是 memorylite？🚀

很多 LLM memory 系统一旦开始落地，就会迅速变重：

- 服务越来越多
- 依赖越来越多
- 编排越来越复杂
- 延迟越来越高

`memorylite` 的路线不同：

- **SQLite-first** 的结构化记忆与状态存储
- **本地 JSONL** 归档原始对话
- **小模型 memory controller** 负责 recall 决策和 memory writing
- **轻量 lexical 检索**，再可选做小范围 semantic rerank
- **兼容 OpenAI 协议**，方便接 Qwen、DeepSeek、vLLM、自建网关等
- **低延迟 Python API**，方便嵌入真实聊天程序
- **内置维护机制**，支持去重、TTL 清理、事件压缩

## 功能亮点 ✨

- 🗂️ **结构化记忆**：fact、preference、event、summary、task_state
- ⚡ **快速 recall 路径**：cache + SQLite + 小候选池
- 🤖 **模型驱动 memory agent**：小模型持续参与 recall 和写入
- 🔎 **可选语义重排**：不依赖重型向量数据库
- 🏠 **本地优先架构**：适合简单部署和本地运行
- 🔌 **OpenAI-compatible 支持**：方便接 Qwen / DashScope / DeepSeek / 自建后端
- 📏 **内置 benchmark**：方便持续评测效果
- 📦 **可通过 pip 使用**：适合作为 Python 库集成

## 文档导航 📚

- [English README](README.md)
- [检索与记忆架构](docs/RETRIEVAL.zh-CN.md)
- [Retrieval & Memory Architecture](docs/RETRIEVAL.md)
- [评测与数据集说明](docs/BENCHMARKING.zh-CN.md)
- [Benchmarking & Evaluation Guide](docs/BENCHMARKING.md)

## 安装

```bash
pip install -e .
```

在其他本地 Python 项目中安装：

```bash
pip install /path/to/memory_agent
```

如果需要运行 OpenAI-compatible 示例：

```bash
pip install "memorylite[openai]"
```

查看版本：

```python
import memorylite
print(memorylite.__version__)
```

## 快速开始

### 1. 零依赖 Demo 快速体验

> `DemoMemoryModelClient` 是一个**零依赖、本地快速体验用的 demo client，不是生产环境 memory model**。

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
    user_message="请记住我喜欢简洁回答。",
    assistant_message="好的，我会尽量简洁。",
)

recall = agent.recall(
    query="你还记得我的回答偏好吗？",
    session_id="demo-session",
    user_id="alice",
)

print(recall.compiled_text)
agent.close()
```

### 2. 使用 `OpenAICompatibleJSONClient` 接真实 memory model

```python
import os

from memorylite import MemoryAgent, MemoryAgentConfig
from memorylite.llm import OpenAICompatibleJSONClient

agent = MemoryAgent(
    MemoryAgentConfig(root_dir="./.memorylite"),
    client=OpenAICompatibleJSONClient(
        model="qwen2.5-7b-instruct",
        base_url=os.getenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("OPENAI_COMPAT_API_KEY"),
    ),
)
```

这类接法适合：

- DashScope compatible mode
- DeepSeek compatible endpoints
- vLLM
- LM Studio
- 自建 OpenAI-compatible 网关

### 3. 使用 `OllamaJSONClient` 接真实 memory model

```python
from memorylite import MemoryAgent, MemoryAgentConfig
from memorylite.llm import OllamaJSONClient

agent = MemoryAgent(
    MemoryAgentConfig(root_dir="./.memorylite"),
    client=OllamaJSONClient(
        model="qwen2.5:7b-instruct",
        base_url="http://127.0.0.1:11434",
    ),
)
```

## 一步式对话 API

如果你已经有自己的聊天模型封装，可以直接使用 `run_turn()`，把：

`recall -> chat -> remember`

压成一次调用。

```python
turn = agent.run_turn(
    chat_model=chat_model,
    user_message="我之前说想用什么技术栈？",
    session_id="demo-session",
    user_id="alice",
    system_prompt="You are a helpful assistant.",
)

print(turn.assistant_message)
print(turn.recall.items)
```

## 适合的接入场景

`memorylite` 特别适合：

- 带用户长期记忆的聊天机器人
- 带项目记忆的编程 Copilot
- 带任务状态的任务型助手
- 轻量 Agent 框架
- 本地优先 RAG / Memory 系统
- 需要兼容 OpenAI 协议后端的 Python 项目

可运行示例：

- [examples/basic_usage.py](examples/basic_usage.py)
- [examples/chat_loop.py](examples/chat_loop.py)
- [examples/deepseek_chat.py](examples/deepseek_chat.py)
- [examples/openai_compatible_chat.py](examples/openai_compatible_chat.py)
- [examples/qwen_chat.py](examples/qwen_chat.py)

## 真实模型示例

通用本地示例：

```bash
python examples/chat_loop.py
```

用 Ollama 做 memory model：

```bash
set OLLAMA_MEMORY_MODEL=qwen2.5:3b-instruct
python examples/chat_loop.py
```

同时用 Ollama 做 memory + chat：

```bash
set OLLAMA_MEMORY_MODEL=qwen2.5:3b-instruct
set OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct
python examples/chat_loop.py
```

用 DeepSeek 做 chat + memory：

```bash
set DEEPSEEK_API_KEY=your_key_here
python examples/deepseek_chat.py
```

用任意 OpenAI-compatible endpoint：

```bash
set OPENAI_COMPAT_BASE_URL=http://127.0.0.1:8000/v1
set OPENAI_COMPAT_MEMORY_MODEL=qwen2.5-3b-instruct
set OPENAI_COMPAT_CHAT_MODEL=qwen2.5-7b-instruct
python examples/openai_compatible_chat.py
```

用 Qwen / DashScope compatible mode：

```bash
set DASHSCOPE_API_KEY=your_key_here
set QWEN_CHAT_MODEL=qwen2.5-7b-instruct
set QWEN_MEMORY_MODEL=qwen2.5-3b-instruct
set QWEN_EMBED_MODEL=text-embedding-v4
python examples/qwen_chat.py
```

## 架构概览

当前设计是一个**轻量 memory agent**，不是重型的多步自主 agent 平台。

1. **Preload**：从缓存 / SQLite 读取 recent messages 和 compact state
2. **Candidate retrieval**：先做本地 lexical 检索，拿到很小的候选池
3. **Semantic rerank（可选）**：只对小候选池做语义重排
4. **Recall controller**：小模型决定是否需要 long-term memory，并选出 memory IDs
5. **Context compiler**：把选中的记忆编译成 prompt-ready context
6. **Writer**：每轮结束后，小模型提取 durable memories 和 state patches
7. **Maintenance**：去重、TTL 清理、事件压缩，保持系统轻量

更详细的说明见 [检索与记忆架构文档](docs/RETRIEVAL.zh-CN.md)。

## 可选语义检索

语义检索是可选的，而且刻意做得很轻：

- 不引入重型向量数据库
- 不做全库向量检索
- 只做 query embedding + 小候选池 rerank

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

## 评测

仓库内已经包含：

- benchmark runner: [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py)
- benchmark dataset: [benchmarks/datasets/memory_benchmark_v1.json](benchmarks/datasets/memory_benchmark_v1.json)

运行默认离线 benchmark：

```bash
python .\benchmarks\run_benchmark.py
```

运行 Qwen benchmark：

```bash
python .\benchmarks\run_benchmark.py --backend qwen
```

详细的评测指标、数据集格式和 benchmark 说明见 [评测文档](docs/BENCHMARKING.zh-CN.md)。

## 主要组件

- `MemoryAgent`：主入口
- `ModelMemoryController`：小模型 memory controller，负责 recall 决策和 memory writing
- `MemoryRetriever`：本地候选召回器
- `SQLiteStore`：结构化记忆、状态、消息、可选 embedding 存储
- `ArchiveStore`：原始消息 JSONL 归档
- `ContextCompiler`：生成最终 prompt-ready context
- `OpenAICompatibleJSONClient`：OpenAI 风格 memory model 适配器
- `OpenAICompatibleEmbeddingClient`：OpenAI 风格 embedding 适配器
- `OllamaJSONClient`：Ollama 风格本地 memory 接口适配器

## 设计原则

- **只在需要时使用 memory**
- **尽量保持候选池很小**
- **让存储本地、可检查、可维护**
- **让 memory controller 保持简单**
- **让 recall 路径保持边界清晰**
- **把长期记忆当成基础设施，而不是魔法**

## 打包与发布

构建：

```bash
python -m build
```

本地安装 wheel：

```bash
pip install dist/memorylite-0.1.0-py3-none-any.whl
```

上传：

```bash
python -m twine upload dist/*
```
