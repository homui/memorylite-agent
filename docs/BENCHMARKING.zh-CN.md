# 评测与数据集说明 📏📊

[English Version](BENCHMARKING.md)

这份文档说明如何评测 `memorylite`，以及如何构造适合 memory agent 的测试数据集。

## 1. 为什么要评测 memory agent

一个 memory 系统可能在很多层出问题：

- 记忆写入错了
- 记忆写到了错误的 scope
- 正确候选没有被召回出来
- recall controller 拒绝触发
- 记忆虽然召回了，但最终 prompt 没把它表达好

所以 memory agent 不能只看一个总分。

## 2. 核心指标

### 2.1 Trigger Accuracy

系统是否正确判断：

- 这轮应该 recall
- 或这轮不应该 recall

### 2.2 Memory Hit Rate

选中的 `recall.items` 是否包含 gold memory。

它主要衡量：

- 候选召回质量
- recall controller 选择质量

### 2.3 Context Hit Rate

最终 `compiled_text` 是否包含 gold fact。

它主要衡量：

- memory 选择是否正确
- compiler 是否把记忆表达清楚
- 最终 prompt 是否真的保留了有用信息

### 2.4 Recall Latency

除了总 recall 时间，还要看阶段耗时：

- `preload_ms`
- `candidate_ms`
- `semantic_ms`
- `controller_ms`
- `compile_ms`

这些阶段指标对性能优化非常关键。

## 3. 内置 Benchmark 文件

主要文件：

- benchmark runner: [../benchmarks/run_benchmark.py](../benchmarks/run_benchmark.py)
- 默认数据集: [../benchmarks/datasets/memory_benchmark_v1.json](../benchmarks/datasets/memory_benchmark_v1.json)

## 4. Benchmark 的工作方式

对每个 sample：

1. 用 `remember(sync=True)` 回放历史 turns
2. 在 query turn 调用 `recall(...)`
3. 统计：
   - `triggered`
   - `selected memory items`
   - 最终 `compiled context`
   - 阶段耗时
   - memory 数量 / 数据库大小

因此它测的是完整的 memory pipeline，不是单点函数。

## 5. 数据集格式

每条 sample 的结构类似：

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

关键字段：

- `id`
- `category`
- `session_id`
- `query_session_id`（可选）
- `user_id`
- `turns`
- `query`
- `gold_memory_substrings`
- `gold_context_substrings`
- `expected_triggered`

## 6. 推荐的场景类别

一套合格的 memory benchmark 至少要覆盖：

- `preference`
- `fact`
- `task_state`
- `event`
- `conflict`
- `negative`

### Preference

测试用户风格、格式或行为偏好记忆。

### Fact

测试稳定事实记忆，比如项目技术栈、身份、固定约束。

### Task State

测试可执行状态，比如 TODO、当前进行中的任务。

### Event

测试时间性或事件性记忆，比如计划、过去发生的事。

### Conflict

测试新记忆是否能正确覆盖旧记忆。

### Negative

测试系统是否能避免不必要的 recall。

## 7. 如何运行

### 离线 demo 模式

```bash
python .\benchmarks\run_benchmark.py
```

### Qwen / DashScope 模式

```bash
python .\benchmarks\run_benchmark.py --backend qwen
```

可选开启语义重排：

```bash
python .\benchmarks\run_benchmark.py --backend qwen --semantic-real --embed-model text-embedding-v4
```

### OpenAI-compatible 模式

```bash
python .\benchmarks\run_benchmark.py --backend openai_compat --model your-model-name --base-url http://127.0.0.1:8000/v1
```

### 调试模型输入输出

```bash
python .\benchmarks\run_benchmark.py --backend qwen --debug-model-io
```

原始请求/响应会写到：

`benchmarks/.runs/<sample_id>/model_debug.jsonl`

## 8. 如何解读结果

### 情况 A

- `memory_hit_rate` 高
- `context_hit_rate` 低

说明：

- recall 基本找对了
- 但 compiler 或 prompt 组织方式偏弱

### 情况 B

- `memory_hit_rate` 低
- `context_hit_rate` 低

说明：

- 问题更偏上游
- 可能是写入、候选召回或 controller 选择不稳定

### 情况 C

- `trigger_accuracy` 高
- `memory_hit_rate` 低

说明：

- 系统知道应该 recall
- 但没有选到正确 memory，或者 memory 写入质量差

### 情况 D

- 效果还可以
- 但 `controller_ms` 很高

说明：

- recall 质量尚可
- 真正瓶颈在 memory model 后端，而不是本地检索

## 9. 推荐的评测策略

建议保留两套 benchmark：

### 9.1 回归 benchmark

使用内置 demo backend。

目标：

- 检查代码变更是否破坏 pipeline
- 保持稳定的本地回归信号

### 9.2 真实模型 benchmark

使用 Qwen / DashScope 或其他 OpenAI-compatible memory model。

目标：

- 评估真实 memory writing 稳定性
- 评估真实 recall-controller 表现
- 测真实延迟

## 10. 如何构造更好的数据集

### 10.1 先从小样本开始

先写 20 到 50 条高质量手工样本。

### 10.2 增加长间隔样本

在原始记忆和 query 之间插入大量无关轮次。

### 10.3 增加冲突样本

例如：

- 旧：项目用 SQLite
- 新：项目现在改用 Postgres

### 10.4 增加高噪声样本

加入大量相似但不相关的 memory，用来测试排序和选择。

### 10.5 增加跨 session 样本

在一个 session 写入 memory，在另一个 session 用同一 `user_id` 查询。

## 11. 当指标不好时，优先该查什么

推荐排查顺序：

1. 先看 memory writing
2. 再看 memory scope
3. 再看 candidate pool 质量
4. 再看 controller 决策
5. 最后看 compiler 输出

## 12. 实用目标值

对于轻量、面向生产的 memory layer，可以把这些当作参考目标：

- `trigger_accuracy >= 0.9`
- `memory_hit_rate >= 0.8`
- `context_hit_rate >= 0.8`
- recall 延迟可控
- negative sample 延迟足够低
