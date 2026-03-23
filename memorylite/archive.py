from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArchiveStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def append_message(self, session_id: str, payload: dict[str, Any]) -> tuple[str, int]:
        path = self.root_dir / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            offset = fh.tell()
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return str(path), offset
