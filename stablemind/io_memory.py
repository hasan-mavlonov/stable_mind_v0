import json
from pathlib import Path
from typing import Any, Dict, List


class MemoryStore:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

    def _mem_file(self, session_id: str) -> Path:
        return self.root / "memory" / "episodic_memory.jsonl"

    def _belief_file(self, session_id: str) -> Path:
        return self.root / "memory" / "belief_observations.jsonl"

    def append_episode(self, session_id: str, turn: int, user_text: str, detected_events: List[str], entities: List[str], notes: str) -> None:
        record = {
            "turn": turn,
            "user_text": user_text,
            "detected_events": detected_events,
            "entities": entities,
            "notes": notes,
        }
        self._append_jsonl(self._mem_file(session_id), record)

    def append_belief_observation(self, session_id: str, turn: int, obs: Dict[str, Any]) -> None:
        record = {"turn": turn, **obs}
        self._append_jsonl(self._belief_file(session_id), record)

    def append_agent_reply(self, session_id: str, turn: int, agent_text: str) -> None:
        self._append_jsonl(self._mem_file(session_id), {"turn": turn, "agent_text": agent_text})

    def read_last_episodes(self, session_id: str, n: int = 7) -> List[Dict[str, Any]]:
        path = self._mem_file(session_id)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        out = [json.loads(x) for x in lines[-n:] if x.strip()]
        return out

    def read_beliefs_window(self, session_id: str, start_turn: int, end_turn: int) -> List[Dict[str, Any]]:
        path = self._belief_file(session_id)
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            t = obj.get("turn", -1)
            if start_turn <= t <= end_turn:
                out.append(obj)
        return out

    def _append_jsonl(self, path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")