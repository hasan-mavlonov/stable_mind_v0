from pathlib import Path
from typing import Any, Dict


class PromptBuilder:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.template = (self.root / "prompts" / "prompt_template.txt").read_text(encoding="utf-8")

    def build(self, persona: Dict[str, Any], vectors: Dict[str, Any], rules: Dict[str, Any], memory, session_id: str, user_message: str, turn: int) -> str:
        immutable = persona["immutable"]
        stable = persona["stable"]
        dynamic = persona["dynamic"]
        anchors = persona["anchors"]

        # minimal renderers (upgrade later)
        immutable_facts = self._render_immutable(immutable)
        stable_identity = self._render_stable_identity(stable, anchors)
        core_values = self._render_values(stable)
        stable_voice = stable.get("personality", {}).get("tone_of_voice", "")
        stable_beliefs_nl = self._render_beliefs(stable)

        mood_summary = self._render_mood(vectors["emotion_vector"])
        trait_current = self._render_traits(vectors["trait_vector"]["current"])

        recent_eps = memory.read_last_episodes(session_id=session_id, n=7)
        relevant_episodic_memories = self._render_memories(recent_eps)
        recent_belief_observations = "(see belief_observations.jsonl)"

        prompt = self.template.format(
            display_name=immutable.get("display_name", "Rin"),
            username=immutable.get("username", "@rin"),
            immutable_facts=immutable_facts,
            stable_identity=stable_identity,
            core_values=core_values,
            stable_voice=stable_voice,
            stable_beliefs_nl=stable_beliefs_nl,
            dynamic_state=self._render_dynamic(dynamic),
            mood_summary=mood_summary,
            trait_vector_current=trait_current,
            relevant_episodic_memories=relevant_episodic_memories,
            recent_belief_observations=recent_belief_observations,
            user_message=user_message,
        )
        return prompt

    def _render_immutable(self, immutable: Dict[str, Any]) -> str:
        out = []
        out.append(f"- ID: {immutable.get('id')}")
        out.append(f"- Name: {immutable.get('display_name')}")
        out.append(f"- Username: {immutable.get('username')}")
        out.append(f"- Nationality: {immutable.get('nationality')}")
        origin = immutable.get("origin", {})
        out.append(f"- Origin: born in {origin.get('birth_city')}, story: {origin.get('story','')}")
        markers = immutable.get("immutable_appearance_markers", {})
        if markers:
            out.append(f"- Origin eye color: {markers.get('origin_eye_color')}")
            out.append(f"- Origin hair color: {markers.get('origin_hair_color')}")
        return "\n".join(out)

    def _render_stable_identity(self, stable: Dict[str, Any], anchors: Dict[str, Any]) -> str:
        p = stable.get("personality", {})
        core_traits = ", ".join(p.get("core_traits", []))
        return f"MBTI: {p.get('mbti','')}. Core traits: {core_traits}. Identity anchors: {anchors.get('identity_anchors', [])}"

    def _render_values(self, stable: Dict[str, Any]) -> str:
        vals = stable.get("personality", {}).get("values", [])
        return "\n".join([f"- {v}" for v in vals]) if vals else "- (none)"

    def _render_beliefs(self, stable: Dict[str, Any]) -> str:
        # v0 minimal
        beliefs = stable.get("stable_beliefs", {})
        return str(beliefs) if beliefs else "(none)"

    def _render_dynamic(self, dynamic: Dict[str, Any]) -> str:
        return str(dynamic)

    def _render_mood(self, emotion: Dict[str, float]) -> str:
        top = sorted(emotion.items(), key=lambda x: x[1], reverse=True)[:2]
        return "Top emotions: " + ", ".join([f"{k}={v:.2f}" for k, v in top])

    def _render_traits(self, traits: Dict[str, float]) -> str:
        # render as human-readable (not numbers-heavy)
        def label(x: float) -> str:
            if x >= 0.6: return "high"
            if x >= 0.2: return "moderate"
            if x > -0.2: return "slightly"
            if x > -0.6: return "low"
            return "very low"

        lines = []
        for k, v in traits.items():
            lines.append(f"{k}: {label(v)}")
        return "\n".join(lines)

    def _render_memories(self, eps) -> str:
        lines = []
        for e in eps:
            if "user_text" in e:
                lines.append(f"- user: {e.get('user_text')} (events={e.get('detected_events', [])})")
            if "agent_text" in e:
                lines.append(f"- agent: {e.get('agent_text')}")
        return "\n".join(lines) if lines else "(none)"