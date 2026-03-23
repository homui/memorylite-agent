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
from memorylite.llm import DemoMemoryModelClient, OllamaJSONClient


SYSTEM_PROMPT = """\
You are a helpful assistant.
Use the provided memory context when it is relevant.
If the memory context contains stable user preferences or task state, follow them.
Keep answers practical and grounded in the current user request.
"""


class DemoLLM:
    """A tiny local stand-in so the example runs without external services."""

    def generate(self, system_prompt: str, memory_context: str, user_message: str) -> str:
        lower = user_message.lower()
        if "remember" in lower or "history" in lower or "还记得" in user_message:
            return (
                "I checked the memory context and found relevant prior details.\n\n"
                f"{memory_context}"
            )
        return (
            "I received your message and would normally send the compiled prompt to your main LLM.\n"
            "Swap DemoLLM with a real backend to get production answers.\n\n"
            f"User message: {user_message}"
        )


class OllamaChatLLM:
    """Minimal Ollama backend for the main chat model."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434") -> None:
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
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["message"]["content"].strip()


def build_chat_llm():
    model_name = os.getenv("OLLAMA_CHAT_MODEL")
    if model_name:
        return OllamaChatLLM(model=model_name)
    return DemoLLM()


def build_memory_client():
    model_name = os.getenv("OLLAMA_MEMORY_MODEL")
    if model_name:
        return OllamaJSONClient(model=model_name)
    return DemoMemoryModelClient()


def call_your_llm(llm, user_message: str, memory_context: str) -> str:
    return llm.generate(
        system_prompt=SYSTEM_PROMPT,
        memory_context=memory_context,
        user_message=user_message,
    )


def print_banner() -> None:
    print("=" * 72)
    print("memorylite agent chat loop example")
    print("Type 'exit' to quit.")
    print("Set OLLAMA_MEMORY_MODEL=qwen2.5:3b-instruct for the memory agent.")
    print("Set OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct for the main chat model.")
    print("=" * 72)


def main() -> None:
    memory = MemoryAgent(
        MemoryAgentConfig(
            root_dir="./chat-memory-data",
            background_write=True,
        ),
        client=build_memory_client(),
    )
    llm = build_chat_llm()
    session_id = "chat-demo-session"
    user_id = "demo-user"

    print_banner()

    try:
        while True:
            user_message = input("\nYou> ").strip()
            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                break

            recall = memory.recall(
                query=user_message,
                session_id=session_id,
                user_id=user_id,
            )

            assistant_message = call_your_llm(
                llm=llm,
                user_message=user_message,
                memory_context=recall.compiled_text,
            )

            print(f"\nAssistant> {assistant_message}")

            memory.remember(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )
    finally:
        memory.close()


if __name__ == "__main__":
    main()
