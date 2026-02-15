import re
from pathlib import Path
import json
from typing import Any, Dict, List


class EventExtractor:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.taxonomy = json.loads((self.root / "rules" / "event_taxonomy.json").read_text(encoding="utf-8"))

    def extract(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text = user_message.lower()
        events: List[str] = []
        entities: List[str] = []
        belief_observations: List[Dict[str, Any]] = []
        notes = ""

        # --- 1) basic event heuristics (same as yours) ---
        if any(k in text for k in ["congrats", "i did it", "won", "achieved"]):
            events.append("major_achievement")
        if any(k in text for k in ["rejected", "ignored", "left me", "they hate"]):
            events.append("social_rejection")
        if any(k in text for k in ["betray", "lied to me", "backstab"]):
            events.append("betrayal")
        if any(k in text for k in ["argue", "fight", "conflict"]):
            events.append("conflict")
        if any(k in text for k in ["burnout", "exhausted", "can't do this"]):
            events.append("burnout_episode")

        # --- 2) resolve "focus entity" (explicit mention > carry-over) ---
        focused_entity = None

        # explicit mention
        if "bon" in text and "cafe" in text:
            focused_entity = "Bon cafe"

        # pronoun / deixis fallback
        # if user says "there/it/that place" and we have a last focus, use it
        if focused_entity is None:
            if any(p in text for p in ["there", "it", "that place", "this place", "here"]):
                focused_entity = context.get("last_entity_focus")

        # if still None, we simply don't attach entity-based observations
        if focused_entity:
            entities.append(focused_entity)

        # --- 3) belief observations (quietness) using focused entity ---
        noisy_cues = ["loud", "noisy", "crowded", "music", "chaos", "blasting"]
        quiet_cues = ["quiet", "calm", "peaceful", "silent"]

        if focused_entity == "Bon cafe":
            if any(k in text for k in noisy_cues):
                belief_observations.append({
                    "entity": "Bon cafe",
                    "dimension": "quietness",
                    "value": 0.8,  # noisy
                    "evidence_text": user_message
                })
                notes = "Bon cafe was noisy/loud."
            elif any(k in text for k in quiet_cues):
                belief_observations.append({
                    "entity": "Bon cafe",
                    "dimension": "quietness",
                    "value": -0.8,  # quiet
                    "evidence_text": user_message
                })
                notes = "Bon cafe was quiet/calm."

        # --- 4) feedback events (same as yours) ---
        if any(k in text for k in ["good job", "love this", "amazing"]):
            events.append("positive_feedback")
        if any(k in text for k in ["this sucks", "bad", "cringe", "hate it"]):
            events.append("negative_feedback")

        # de-dupe
        events = list(dict.fromkeys(events))
        entities = list(dict.fromkeys(entities))

        return {
            "events": events,
            "entities": entities,
            "belief_observations": belief_observations,
            "notes": notes,
            # helpful for debugging
            "focused_entity": focused_entity
        }