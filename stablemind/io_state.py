import json
from pathlib import Path
from typing import Any, Dict


class StateStore:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

    # ---- load persona blocks
    def load_persona(self) -> Dict[str, Any]:
        immutable = self._read_json(self.root / "persona" / "immutable_traits.json")
        stable = self._read_json(self.root / "persona" / "stable_traits.json")
        dynamic = self._read_json(self.root / "persona" / "dynamic_traits.json")
        anchors = self._read_json(self.root / "persona" / "anchors.json")
        return {"immutable": immutable, "stable": stable, "dynamic": dynamic, "anchors": anchors}

    def save_persona(self, persona: Dict[str, Any]) -> None:
        # only stable/dynamic should ever be saved in v0
        self._write_json(self.root / "persona" / "stable_traits.json", persona["stable"])
        self._write_json(self.root / "persona" / "dynamic_traits.json", persona["dynamic"])

    # ---- load rules
    def load_rules(self) -> Dict[str, Any]:
        rules_dir = self.root / "rules"

        event_emotion = self._read_json(rules_dir / "event_emotion_deltas.json")
        emotion_trait_nudges = self._read_json(rules_dir / "emotion_trait_nudges.json")
        update_policy = self._read_json(rules_dir / "update_policy.json")
        taxonomy = self._read_json(rules_dir / "event_taxonomy.json")

        # Some engines expect rules["emotion_trait"]["emotion_decay"] etc.
        # But the nudges file also contains these parameters. We expose them in both places
        # to keep compatibility and avoid KeyErrors.
        emotion_trait = {
            "emotion_decay": emotion_trait_nudges.get("emotion_decay", 0.75),
            "trait_return_to_baseline": emotion_trait_nudges.get("trait_return_to_baseline", 0.85),
            "per_turn_trait_cap": emotion_trait_nudges.get("per_turn_trait_cap", 0.05),
        }

        return {
            # canonical keys (what your engines should use)
            "event_emotion": event_emotion,
            "emotion_trait_nudges": emotion_trait_nudges,
            "update_policy": update_policy,
            "event_taxonomy": taxonomy,
            "emotion_trait": emotion_trait,

            # backwards-compatible aliases (in case some modules still use old names)
            "taxonomy": taxonomy,
        }

    # ---- state per session (v0 single-file per session)
    def _session_dir(self, session_id: str) -> Path:
        # v0: use shared state/ files; later: create per-session dirs
        return self.root / "state"

    def load_vectors(self, session_id: str) -> Dict[str, Any]:
        return self._read_json(self._session_dir(session_id) / "vectors.json")

    def save_vectors(self, session_id: str, vectors: Dict[str, Any]) -> None:
        self._write_json(self._session_dir(session_id) / "vectors.json", vectors)

    def load_counters(self, session_id: str) -> Dict[str, Any]:
        return self._read_json(self._session_dir(session_id) / "counters.json")

    def save_counters(self, session_id: str, counters: Dict[str, Any]) -> None:
        self._write_json(self._session_dir(session_id) / "counters.json", counters)

    # ---- dynamic update placeholder
    def update_dynamic(self, dynamic: Dict[str, Any], extracted: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        # v0: only very simple updates (you’ll expand later)
        # example: append discoveries
        note = extracted.get("notes")
        if note:
            dynamic.setdefault("recent_discoveries", []).append(note)
        return dynamic

    # ---- helpers
    def _read_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)