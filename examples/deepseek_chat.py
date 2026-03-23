from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import textwrap
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memorylite import MemoryAgent, MemoryAgentConfig
from memorylite.llm import OpenAICompatibleEmbeddingClient, OpenAICompatibleJSONClient


SYSTEM_PROMPT = """\
You are a helpful assistant.
Use the provided memory context when it is relevant.
If the memory context contains stable user preferences or task state, follow them.
Keep answers practical and grounded in the current user request.
"""


class DeepSeekChatLLM:
    """Minimal DeepSeek chat client using the official OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system_prompt: str, memory_context: str, user_message: str) -> str:
        user_prompt = textwrap.dedent(
            f"""\
            Memory context:
            {memory_context}

            User message:
            {user_message}
            """
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()


def build_memory_agent() -> MemoryAgent:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    memory_model = os.getenv("DEEPSEEK_MEMORY_MODEL", "deepseek-chat")
    embedding_base_url = os.getenv("MEMORY_EMBED_BASE_URL")
    embedding_model = os.getenv("MEMORY_EMBED_MODEL")
    embedding_api_key = os.getenv("MEMORY_EMBED_API_KEY", api_key)

    embedding_client = None
    semantic_enabled = bool(embedding_base_url and embedding_model)
    if semantic_enabled:
        embedding_client = OpenAICompatibleEmbeddingClient(
            model=embedding_model,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
        )

    return MemoryAgent(
        MemoryAgentConfig(
            root_dir="./deepseek-memory-data",
            background_write=True,
            semantic_search_enabled=semantic_enabled,
            semantic_embedding_model=embedding_model if semantic_enabled else None,
        ),
        client=OpenAICompatibleJSONClient(
            model=memory_model,
            base_url=base_url,
            api_key=api_key,
        ),
        embedding_client=embedding_client,
    )


def build_chat_llm() -> DeepSeekChatLLM:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")

    return DeepSeekChatLLM(
        api_key=api_key,
        model=os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def print_banner() -> None:
    print("=" * 72)
    print("memorylite + DeepSeek chat example")
    print("Type 'exit' to quit.")
    print("Required env: DEEPSEEK_API_KEY")
    print("Optional env: DEEPSEEK_CHAT_MODEL, DEEPSEEK_MEMORY_MODEL, DEEPSEEK_BASE_URL")
    print("Optional semantic env: MEMORY_EMBED_BASE_URL, MEMORY_EMBED_MODEL, MEMORY_EMBED_API_KEY")
    print("=" * 72)


def print_memory_log(turn_result) -> None:
    recall = turn_result.recall
    print("\n[memory] recall finished")
    print(f"[memory] triggered={recall.triggered} reason={recall.reason}")
    print(f"[memory] recalled_items={len(recall.items)} recent_messages={len(recall.recent_messages)}")
    if recall.states:
        state_keys = {scope: list(payload.keys()) for scope, payload in recall.states.items() if payload}
        print(f"[memory] state_keys={state_keys}")
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
    llm = build_chat_llm()
    session_id = "deepseek-demo-session"
    user_id = "deepseek-demo-user"

    print_banner()
    print(f"[boot] chat_model={llm.model}")
    print(f"[boot] memory_model={os.getenv('DEEPSEEK_MEMORY_MODEL', 'deepseek-chat')}")
    print(f"[boot] background_write={memory.config.background_write}")
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

            print(f"\n[chat] sending request to DeepSeek model={llm.model} base_url={llm.base_url}")
            turn_result = memory.run_turn(
                chat_model=llm,
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
                system_prompt=SYSTEM_PROMPT,
            )

            print_memory_log(turn_result)
            print(f"\nAssistant> {turn_result.assistant_message}")
            mode = "queued in background" if memory.config.background_write else "written synchronously"
            print(f"[memory] turn memory write {mode}")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
