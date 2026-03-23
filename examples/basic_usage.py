from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memorylite import MemoryAgent, MemoryAgentConfig
from memorylite.llm import DemoMemoryModelClient


def main() -> None:
    agent = MemoryAgent(
        MemoryAgentConfig(
            root_dir="./.memorylite-demo",
            background_write=False,
        ),
        client=DemoMemoryModelClient(),
    )

    agent.remember(
        session_id="demo",
        user_id="alice",
        user_message="Please remember that I prefer concise answers.",
        assistant_message="Understood. I will keep answers concise.",
    )

    agent.remember(
        session_id="demo",
        user_id="alice",
        user_message="TODO: Build the project in Python with SQLite and local files.",
        assistant_message="Recorded: Python, SQLite, and local files.",
    )

    recall = agent.recall(
        query="Do you remember what stack I want to use?",
        session_id="demo",
        user_id="alice",
    )

    print(recall.compiled_text)
    agent.close()


if __name__ == "__main__":
    main()
