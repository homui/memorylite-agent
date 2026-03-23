# 检索与记忆架构 🔎🧠

[English Version](RETRIEVAL.md)

这份文档说明 `memorylite` 的记忆写入、候选召回、语义重排、上下文编译和维护机制。

## 1. 目标

`memorylite` 的目标是做一个：

- 本地优先
- 轻量可控
- 延迟可控
- 易于嵌入 Python LLM 项目

的 memory layer。

它不是：

- 重型自主 Agent 运行时
- 完整向量数据库平台
- 超长上下文的替代品

它更适合被理解为：

- 记忆层
- 检索中间层
- 轻量长期记忆系统

## 2. 核心数据类型

系统存的不是纯原始文本，而是结构化记忆对象。

主要 `kind`：

- `fact`
- `preference`
- `event`
- `summary`
- `task_state`

主要 `scope`：

- `session`
- `user`
- `project`

这种分层的好处是：

- `fact` 和 `event` 不会混在一起
- `preference` 不会和 `task_state` 混在一起
- `user` 记忆不会和 `session` 记忆混用

## 3. 写入路径

每轮对话结束后会执行：

1. 写入 user / assistant 原始消息
2. 让 memory controller 提取 durable memory
3. 在落库前做本地归一化
4. 写入 SQLite
5. 如果开启语义重排，再一次性写入 embedding

### 3.1 写入什么

写入阶段主要关注：

- 稳定事实
- 持久偏好
- 有意义的计划或事件
- 可执行的任务状态

不会把每一句闲聊都写成长期记忆。

### 3.2 当前的稳定化逻辑

在真正落库前，`memorylite` 现在会做一层本地后处理：

- 归一化 `scope` / `scope_id_key`
- 归一化 `kind`
- 清理像 `user_message='...'` 这种包装脏字段
- 当用户原始表述更直白、更可检索时，优先保留原始 phrasing
- 修复模板化 summary
- 自动补 weekday、技术词等 tags

这一步尤其重要，因为很多 OpenAI-compatible 小模型虽然能输出 JSON，但字段质量并不稳定。

## 4. Recall 路径

Recall 被刻意做成“有边界、低延迟”的流程。

### 4.1 阶段划分

1. **Preload**
   - recent messages
   - active state

2. **Candidate retrieval**
   - 从 SQLite 做本地 lexical 搜索
   - FTS / lexical overlap / scoring
   - scope short-circuiting

3. **Optional semantic rerank**
   - 只对小候选池做语义重排
   - 不做全库向量检索

4. **Recall controller**
   - 小模型决定这轮要不要 long-term memory
   - 并选出候选 memory IDs

5. **Local fallback**
   - 如果模型过于保守
   - 本地高分或高重合候选仍可直接兜底

6. **Context compiler**
   - 生成最终可直接拼进 prompt 的 memory context

## 5. 候选召回

`memorylite` 先做便宜的本地候选召回，再让 memory controller 参与。

当前候选分数会综合这些信号：

- lexical term overlap
- 字符 n-gram overlap
- exact substring bonus
- tag bonus
- entity bonus
- importance
- recency
- scope bonus

这意味着本地候选召回具备这些特点：

- 快
- 本地可调试
- 可解释
- 可控

### 5.1 Scope short-circuiting

当前 scope 是有优先顺序的：

- `session`
- `user`
- `project`

这样能减少噪声，避免每次都把所有范围一视同仁地查一遍。

## 6. 语义重排

语义检索是可选的。

开启后：

- 写入 durable memory 时会一次性存 embedding
- recall 时只做 query embedding
- 只对很小的候选池做 semantic rerank

这样可以避免引入很重的向量检索系统。

## 7. Recall Controller

Recall controller 是一个**小 memory model**，不是主聊天模型。

它的职责被刻意限制为：

- 决定要不要 recall
- 从候选池里选 memory IDs

它**不直接做全库数据库搜索**。

所以当前 `memorylite` 更准确的定位是：

`本地检索找候选，小模型做轻量判断`

也就是一个轻量 memory agent，而不是重型多步自治 agent。

## 8. 本地 Recall Fallback

真实小模型经常会过于保守。

为了提升 recall 质量，`memorylite` 加了本地兜底：

- 如果模型说 `should_recall=false`
- 但候选非常强
- 系统仍然可以把它注入上下文

当前 fallback 的触发依据包括：

- 本地候选高分
- query 与 candidate 的 lexical / topic overlap 很强

## 9. Context Compiler

Compiler 负责把选中的记忆翻译成 prompt-ready context。

当前会生成这些结构块：

- `[Memory Guidance]`
- `[Active State]`
- `[Important Preferences]`
- `[Relevant Task State]`
- `[Relevant Memory]`
- `[Recent Conversation]`
- `[Current Query]`

为什么 compiler 重要？

因为即使 recall 选对了，也可能出现：

- 记忆找到了
- 但主模型没用好

原因通常是：

- 写得太模糊
- 转述太多
- 把原始事实藏起来了

## 10. Maintenance

长期运行的 memory 系统如果只追加不整理，一定会退化。

`memorylite` 当前已经内置：

- 去重合并
- TTL 清理
- 旧 event 压缩

这样可以保持：

- 空间更小
- 检索更快
- 候选噪声更低

## 11. 实际调优建议

如果你要提**准确率**，建议顺序是：

1. 先提升写入质量
2. 再提升候选召回质量
3. 再优化 compiler
4. 最后再增加 semantic 能力

如果你要提**速度**，建议顺序是：

1. 减少 memory-model 调用次数
2. 缩短 controller prompt
3. 保持候选池很小
4. 多利用本地短路和 fallback

## 12. 推荐排查顺序

如果 recall 质量变差，建议按这个顺序排查：

1. 先看 memory 有没有被正确写进去
2. 再看正确 memory 有没有进入候选池
3. 再看 controller 有没有把正确 memory 选出来
4. 最后再看 compiler 有没有把它表达好

真实项目里，很多问题最终都不是 compiler 本身，而是：

- 写入质量差
- scope 不稳定
- recall controller 过于保守
