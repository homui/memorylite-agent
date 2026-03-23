from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memorylite import MemoryAgent, MemoryAgentConfig
from memorylite.llm import (
    OpenAICompatibleChatModel,
    OpenAICompatibleEmbeddingClient,
    OpenAICompatibleJSONClient,
)


SYSTEM_PROMPT = """\
You are a helpful assistant.
Use the provided memory context when it is relevant.
If the memory context contains stable user preferences or task state, follow them.
Keep answers practical and grounded in the current user request.
"""


def build_memory_agent() -> MemoryAgent:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY environment variable.")

    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    memory_model = os.getenv("QWEN_MEMORY_MODEL", "qwen2.5-3b-instruct")
    embed_model = os.getenv("QWEN_EMBED_MODEL", "text-embedding-v4")
    semantic_enabled = os.getenv("QWEN_ENABLE_SEMANTIC", "1") not in {"0", "false", "False"}
    memory_timeout_seconds = float(os.getenv("QWEN_MEMORY_TIMEOUT_SECONDS", "60"))

    embedding_client = None
    if semantic_enabled:
        embedding_client = OpenAICompatibleEmbeddingClient(
            model=embed_model,
            base_url=base_url,
            api_key=api_key,
        )

    return MemoryAgent(
        MemoryAgentConfig(
            root_dir="./qwen-memory-data",
            background_write=True,
            semantic_search_enabled=semantic_enabled,
            semantic_embedding_model=embed_model if semantic_enabled else None,
        ),
        client=OpenAICompatibleJSONClient(
            model=memory_model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=memory_timeout_seconds,
        ),
        embedding_client=embedding_client,
    )


def build_chat_model() -> OpenAICompatibleChatModel:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY environment variable.")
    return OpenAICompatibleChatModel(
        model=os.getenv("QWEN_CHAT_MODEL", "qwen2.5-7b-instruct"),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=api_key,
    )


def print_banner() -> None:
    print("=" * 72)
    print("memorylite + Qwen chat example")
    print("Type 'exit' to quit.")
    print("Required env: DASHSCOPE_API_KEY")
    print("Optional env: QWEN_CHAT_MODEL, QWEN_MEMORY_MODEL, QWEN_EMBED_MODEL, QWEN_BASE_URL")
    print("Optional env: QWEN_ENABLE_SEMANTIC=0 to disable semantic rerank")
    print("Optional env: QWEN_MEMORY_TIMEOUT_SECONDS=60 to tune memory model timeout")
    print("=" * 72)


def print_memory_log(turn_result) -> None:
    recall = turn_result.recall
    used_recent_context = bool(recall.recent_messages)
    used_long_term_memory = bool(recall.items)
    used_state = any(bool(payload) for payload in recall.states.values())

    print("\n[memory] recall finished")
    print(
        "[memory] sources: "
        f"recent_context={'yes' if used_recent_context else 'no'}, "
        f"long_term_memory={'yes' if used_long_term_memory else 'no'}, "
        f"state={'yes' if used_state else 'no'}"
    )
    print(
        "[memory] long_term_recall: "
        f"triggered={recall.triggered} "
        f"items={len(recall.items)} "
        f"reason={recall.reason}"
    )
    print(f"[memory] recent_messages={len(recall.recent_messages)}")
    if used_recent_context and not recall.triggered:
        print("[memory] note: this turn still included recent conversation even though long-term recall was not triggered")
    if recall.items:
        print("[memory] top memories:")
        for index, item in enumerate(recall.items, start=1):
            snippet = (item.summary or item.content).replace("\n", " ").strip()[:120]
            print(
                f"  {index}. id={item.memory_id} kind={item.kind} "
                f"scope={item.scope} score={item.score:.2f} text={snippet}"
            )
    print(f"[memory] compiled_context_chars={len(recall.compiled_text)}")


def main() -> None:
    memory = build_memory_agent()
    chat_model = build_chat_model()
    session_id = "qwen-demo-session"
    user_id = "qwen-demo-user"

    print_banner()
    print(f"[boot] chat_model={chat_model.model}")
    print(f"[boot] memory_model={os.getenv('QWEN_MEMORY_MODEL', 'qwen2.5-3b-instruct')}")
    print(f"[boot] semantic_search_enabled={memory.config.semantic_search_enabled}")
    print(f"[boot] semantic_embedding_model={memory.config.semantic_embedding_model}")
    print(f"[boot] memory_dir={memory.config.root_path}")

    try:
        while True:
            user_message = input("\nYou> ").strip()
            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                break

            turn = memory.run_turn(
                chat_model=chat_model,
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
                system_prompt=SYSTEM_PROMPT,
            )

            print_memory_log(turn)
            print(f"\nAssistant> {turn.assistant_message}")
            mode = "queued in background" if memory.config.background_write else "written synchronously"
            print(f"[memory] turn memory write {mode}")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
