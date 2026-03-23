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
    base_url = os.getenv("OPENAI_COMPAT_BASE_URL")
    memory_model = os.getenv("OPENAI_COMPAT_MEMORY_MODEL")
    api_key = os.getenv("OPENAI_COMPAT_API_KEY")
    if not base_url or not memory_model:
        raise RuntimeError("Missing OPENAI_COMPAT_BASE_URL or OPENAI_COMPAT_MEMORY_MODEL.")

    embed_base_url = os.getenv("OPENAI_COMPAT_EMBED_BASE_URL", base_url)
    embed_model = os.getenv("OPENAI_COMPAT_EMBED_MODEL")
    semantic_enabled = bool(embed_model)
    embedding_client = None
    if semantic_enabled:
        embedding_client = OpenAICompatibleEmbeddingClient(
            model=embed_model,
            base_url=embed_base_url,
            api_key=api_key,
        )

    return MemoryAgent(
        MemoryAgentConfig(
            root_dir="./openai-compat-memory-data",
            background_write=True,
            semantic_search_enabled=semantic_enabled,
            semantic_embedding_model=embed_model if semantic_enabled else None,
        ),
        client=OpenAICompatibleJSONClient(
            model=memory_model,
            base_url=base_url,
            api_key=api_key,
        ),
        embedding_client=embedding_client,
    )


def build_chat_model() -> OpenAICompatibleChatModel:
    base_url = os.getenv("OPENAI_COMPAT_BASE_URL")
    chat_model = os.getenv("OPENAI_COMPAT_CHAT_MODEL")
    api_key = os.getenv("OPENAI_COMPAT_API_KEY")
    if not base_url or not chat_model:
        raise RuntimeError("Missing OPENAI_COMPAT_BASE_URL or OPENAI_COMPAT_CHAT_MODEL.")
    return OpenAICompatibleChatModel(
        model=chat_model,
        base_url=base_url,
        api_key=api_key,
    )


def print_banner() -> None:
    print("=" * 72)
    print("memorylite + OpenAI-compatible chat example")
    print("Type 'exit' to quit.")
    print("Required env: OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_MEMORY_MODEL, OPENAI_COMPAT_CHAT_MODEL")
    print("Optional env: OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_EMBED_MODEL, OPENAI_COMPAT_EMBED_BASE_URL")
    print("=" * 72)


def main() -> None:
    memory = build_memory_agent()
    chat_model = build_chat_model()
    session_id = "openai-compat-demo-session"
    user_id = "openai-compat-demo-user"

    print_banner()
    print(f"[boot] chat_model={chat_model.model}")
    print(f"[boot] memory_model={os.getenv('OPENAI_COMPAT_MEMORY_MODEL')}")
    print(f"[boot] semantic_search_enabled={memory.config.semantic_search_enabled}")
    print(f"[boot] semantic_embedding_model={memory.config.semantic_embedding_model}")

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
            print(f"\nAssistant> {turn.assistant_message}")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
