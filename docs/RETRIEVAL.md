# Retrieval & Memory Architecture 🔎🧠

[中文版本 / Chinese Version](RETRIEVAL.zh-CN.md)

This document explains how `memorylite` performs memory writing, memory retrieval, context compilation, and maintenance.

## 1. Goal

`memorylite` is designed for **fast, bounded, local-first memory recall** in Python LLM applications.

It is not trying to be:

- a massive autonomous agent runtime
- a full vector database platform
- an unbounded long-context replacement

Instead, it aims to be:

- a **memory layer**
- a **retrieval middleware**
- a **lightweight long-term memory system**

for chatbots, copilots, task agents, and local-first AI applications.

## 2. Core Data Types

The system stores structured memories rather than raw text only.

Main memory kinds:

- `fact`
- `preference`
- `event`
- `summary`
- `task_state`

Main scopes:

- `session`
- `user`
- `project`

This separation improves retrieval quality and keeps memory usage predictable.

## 3. Write Path

After each conversation turn:

1. user/assistant messages are stored
2. the memory controller extracts durable memory
3. extracted memory is normalized locally
4. the result is written into SQLite
5. optional embeddings are stored once for later semantic rerank

### 3.1 What gets written

The writer focuses on:

- stable facts
- durable preferences
- meaningful plans or events
- actionable task state

It intentionally avoids treating every sentence as long-term memory.

### 3.2 Current stabilization logic

Before memory is persisted, `memorylite` now does local post-processing:

- normalize `scope` / `scope_id_key`
- normalize `kind`
- strip malformed wrapped fields like `user_message='...'`
- preserve original user phrasing when it is more literal and more searchable
- repair templated summaries
- infer tags such as weekdays or technical keywords

This is especially important for OpenAI-compatible small models, which may return valid JSON but still produce noisy memory fields.

## 4. Recall Path

Recall is intentionally bounded and low-latency.

### 4.1 Stages

1. **Preload**
   - recent messages
   - active state

2. **Candidate retrieval**
   - local lexical search from SQLite
   - FTS / lexical overlap / scoring
   - scope short-circuiting

3. **Optional semantic rerank**
   - rerank only a small candidate pool
   - no full vector DB required

4. **Recall controller**
   - small model decides whether recall is needed
   - chooses candidate IDs

5. **Local fallback**
   - if the model is too conservative
   - high-score or high-overlap candidates can still be selected locally

6. **Context compiler**
   - builds final prompt-ready memory context

## 5. Candidate Retrieval

`memorylite` first performs cheap local retrieval before calling the memory controller.

Signals used for candidate scoring include:

- lexical term overlap
- character n-gram overlap
- exact substring bonus
- tag bonus
- entity bonus
- importance
- recency
- scope bonus

This makes candidate retrieval:

- fast
- local
- inspectable
- predictable

### 5.1 Scope short-circuiting

The system prioritizes scopes in a bounded way:

- `session`
- `user`
- `project`

This reduces noise and avoids scanning everything equally.

## 6. Semantic Rerank

Semantic retrieval is optional.

When enabled:

- memory embeddings are stored once at write time
- a query embedding is computed on recall
- only the small candidate pool is reranked semantically

This avoids a heavy full-corpus vector search architecture.

## 7. Recall Controller

The recall controller is a **small memory model**, not the main chat model.

Its job is intentionally limited:

- decide whether long-term memory is needed
- select memory IDs from the candidate pool

It does **not** perform full DB search on its own.

This is why `memorylite` is best described as a **lightweight memory agent**:

- local retrieval finds candidates
- the memory model makes a simple selection decision

## 8. Local Recall Fallback

Real small models can be too conservative.

To improve practical recall quality, `memorylite` includes local fallback:

- if the model says `should_recall=false`
- but a candidate is obviously strong
- the system can still inject it

Fallback triggers include:

- high local candidate score
- strong lexical/topic overlap

## 9. Context Compiler

The compiler translates selected memories into prompt-ready context.

Current layout includes:

- `[Memory Guidance]`
- `[Active State]`
- `[Important Preferences]`
- `[Relevant Task State]`
- `[Relevant Memory]`
- `[Recent Conversation]`
- `[Current Query]`

Compiler quality matters because you can have:

- correct memory selection
- but poor final answers

if the compiled context is vague or paraphrased too aggressively.

## 10. Maintenance

Long-running memory systems degrade if they only append.

`memorylite` includes lightweight maintenance:

- duplicate merge
- TTL pruning
- event compaction

This helps keep:

- storage smaller
- retrieval faster
- memory noise lower

## 11. Practical Tuning Guide

If you want better memory quality:

1. improve writing quality first
2. improve candidate retrieval second
3. improve compiler third
4. only then add more semantic power

If you want better latency:

1. reduce memory-model calls
2. shorten controller prompts
3. keep candidate pools small
4. rely on local short-circuit paths

## 12. Recommended Debugging Workflow

When retrieval quality is poor:

1. inspect written memories first
2. check whether the right memory reached the candidate pool
3. inspect controller selection decisions
4. inspect the final compiled context

Very often the real issue is:

- bad write quality
- wrong scope
- over-conservative recall controller

not the compiler itself.
