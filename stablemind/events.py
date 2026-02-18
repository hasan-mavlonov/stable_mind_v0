import re
import json
from pathlib import Path
from typing import Any, Dict, List


class EventExtractor:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

        taxonomy_path = self.root / "rules" / "event_taxonomy.json"
        if taxonomy_path.exists():
            self.taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        else:
            self.taxonomy = {"event_types": {}}

    def extract(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text = user_message.lower()

        events: List[str] = []
        entities: List[str] = []
        belief_observations: List[Dict[str, Any]] = []
        notes = ""

        # --- 1) event heuristics ---
        if any(k in text for k in ["congrats", "i did it", "won", "achieved", "got accepted", "promotion", "award"]):
            events.append("major_achievement")
        if any(k in text for k in ["rejected", "ignored", "left me", "they hate", "go away", "stop texting"]):
            events.append("social_rejection")
        if any(k in text for k in ["betray", "betrayed", "lied to me", "you lied", "backstab", "broke my trust"]):
            events.append("betrayal")
        if any(k in text for k in ["argue", "argument", "fight", "conflict", "confrontation"]):
            events.append("conflict")
        if any(k in text for k in ["burnout", "burned out", "burnt out", "exhausted", "can't do this"]):
            events.append("burnout_episode")

        if any(k in text for k in ["good job", "love this", "amazing", "praised", "compliment", "positive feedback"]):
            events.append("feedback_positive")
        if any(k in text for k in ["this sucks", "bad feedback", "cringe", "hate it", "criticized", "negative feedback"]):
            events.append("feedback_negative")

        # taxonomy fallback
        event_types = self.taxonomy.get("event_types", {})
        for ev_type, spec in event_types.items():
            kws = [kw.lower() for kw in spec.get("keywords", [])]
            if kws and any(kw in text for kw in kws):
                events.append(ev_type)

        # --- 2) resolve focus entity ---
        focused_entity = None

        # capture "<something> cafe" with lightweight token parsing
        tokens = re.findall(r"[a-z0-9'’-]+", text)
        for i, tok in enumerate(tokens):
            if tok == "cafe" and i > 0:
                focused_entity = f"{tokens[i - 1]} cafe".title()

        # deixis fallback ("there", "it", "that place")
        if focused_entity is None:
            if any(p in text for p in ["there", "it", "that place", "this place", "here"]):
                focused_entity = context.get("last_entity_focus")

        if focused_entity:
            entities.append(focused_entity)

        # --- 3) belief observations (raw measurements) ---
        noisy_cues = ["loud", "noisy", "music", "chaos", "chaotic", "blasting"]
        quiet_cues = ["quiet", "calm", "peaceful", "silent"]

        if focused_entity:
            if any(k in text for k in noisy_cues):
                belief_observations.append({
                    "entity": focused_entity,
                    "dimension": "noise_level",
                    "value": 0.8,
                    "evidence_text": user_message
                })
                notes = f"{focused_entity} was noisy/loud."

            elif any(k in text for k in quiet_cues):
                belief_observations.append({
                    "entity": focused_entity,
                    "dimension": "noise_level",
                    "value": 0.2,
                    "evidence_text": user_message
                })
                notes = f"{focused_entity} was quiet/calm."

        # de-dupe
        events = list(dict.fromkeys(events))
        entities = list(dict.fromkeys(entities))

        return {
            "events": events,
            "entities": entities,
            "belief_observations": belief_observations,
            "notes": notes,
            "focused_entity": focused_entity
        }