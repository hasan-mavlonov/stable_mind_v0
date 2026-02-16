from dataclasses import dataclass
from typing import Any, Dict, Optional

from .io_state import StateStore
from .io_memory import MemoryStore
from .events import EventExtractor
from .engines import EmotionEngine, TraitEngine, RuminationEngine
from .prompting import PromptBuilder
from .llm import LLMClient
from .llm import LLMClient, LLMConfig


@dataclass
class StepResult:
    text: str
    turn: int
    debug: Dict[str, Any]


class StableMindAgent:
    """
    StableMind v0 Agent.
    Host project passes only user_message (+optional context). StableMind builds prompts internally.
    """

    def __init__(self, root_dir: str, llm_client: Optional[LLMClient] = None):
        self.root_dir = root_dir
        self.state = StateStore(root_dir)
        self.memory = MemoryStore(root_dir)

        self.event_extractor = EventExtractor(root_dir)
        self.emotion_engine = EmotionEngine(root_dir)
        self.trait_engine = TraitEngine(root_dir)
        self.rumination_engine = RuminationEngine(root_dir)

        self.prompt_builder = PromptBuilder(root_dir)

        self.llm = llm_client or LLMClient.openai(
            LLMConfig(model="gpt-4.1-mini", temperature=0.7, max_output_tokens=400)
        )

    def step(self, user_message: str, session_id: str = "default", context: Optional[Dict[str, Any]] = None) -> StepResult:

        # --- Load state/persona/rules
        persona = self.state.load_persona()
        rules = self.state.load_rules()
        vectors = self.state.load_vectors(session_id)
        counters = self.state.load_counters(session_id)
        context = context or {}
        context["last_entity_focus"] = persona["dynamic"].get("last_entity_focus")
        # --- Turn increment
        counters["current_turn"] += 1
        counters["turns_since_last_rumination"] += 1
        turn = counters["current_turn"]

        # --- Event extraction (v0 rule-based)
        extracted = self.event_extractor.extract(user_message, context=context)
        detected_events = extracted["events"]
        belief_obs = extracted["belief_observations"]
        if extracted.get("entities"):
            persona["dynamic"]["last_entity_focus"] = extracted["entities"][-1]
        # --- Emotion update
        vectors["emotion_vector"] = self.emotion_engine.update(
            emotion=vectors["emotion_vector"],
            events=detected_events,
            rules=rules,
        )

        # --- Trait nudges (current only)
        vectors["trait_vector"]["current"] = self.trait_engine.apply_emotion_nudges(
            baseline=vectors["trait_vector"]["baseline"],
            current=vectors["trait_vector"]["current"],
            emotion=vectors["emotion_vector"],
            rules=rules,
        )

        # --- Dynamic updates (fast)
        dynamic = persona["dynamic"]
        dynamic = self.state.update_dynamic(dynamic, extracted, rules)
        persona["dynamic"] = dynamic

        # --- Write memory (append-only)
        self.memory.append_episode(
            session_id=session_id,
            turn=turn,
            user_text=user_message,
            detected_events=detected_events,
            entities=extracted.get("entities", []),
            notes=extracted.get("notes", ""),
        )
        for ob in belief_obs:
            self.memory.append_belief_observation(session_id=session_id, turn=turn, obs=ob)

        rumination_ran = False
        rumination_debug = {}

        # --- Rumination every 20 turns (strict)
        if counters["turns_since_last_rumination"] >= counters["rumination_window_size"]:
            rumination_ran = True
            rumination_debug = self.rumination_engine.run(
                session_id=session_id,
                persona=persona,
                vectors=vectors,
                rules=rules,
                memory=self.memory,
                turn=turn,
            )
            counters["turns_since_last_rumination"] = 0
            counters["last_rumination_turn"] = turn

        # --- Build prompt
        prompt = self.prompt_builder.build(
            persona=persona,
            vectors=vectors,
            rules=rules,
            memory=self.memory,
            session_id=session_id,
            user_message=user_message,
            turn=turn,
        )

        # --- Generate response
        print("\n===== PROMPT (TURN", turn, ") =====\n", prompt[:2000], "\n...TRUNCATED...\n")
        reply = self.llm.generate(prompt)
        print("EVENTS:", detected_events)
        print("EMOTION:", vectors["emotion_vector"])
        print("TRAIT_CURRENT:", vectors["trait_vector"]["current"])

        # --- Persist state
        self.state.save_vectors(session_id, vectors)
        self.state.save_counters(session_id, counters)
        self.state.save_persona(persona)  # stable/dynamic may change through rumination

        # --- Store agent reply
        self.memory.append_agent_reply(session_id=session_id, turn=turn, agent_text=reply)

        debug = {
            "events": detected_events,
            "belief_observations": belief_obs,
            "emotion_vector": vectors["emotion_vector"],
            "trait_current": vectors["trait_vector"]["current"],
            "rumination_ran": rumination_ran,
            "rumination": rumination_debug,
        }
        return StepResult(text=reply, turn=turn, debug=debug)