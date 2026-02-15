from typing import Any, Dict, List
import math


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class EmotionEngine:
    def __init__(self, root_dir: str):
        pass

    def update(self, emotion: Dict[str, float], events: List[str], rules: Dict[str, Any]) -> Dict[str, float]:
        deltas = rules["event_emotion"]
        decay = rules["emotion_trait"].get("emotion_decay", 0.75)

        # decay first
        emotion = {k: clamp(v * decay, 0.0, 1.0) for k, v in emotion.items()}

        # apply deltas
        for ev in events:
            if ev not in deltas:
                continue
            for emo, d in deltas[ev].items():
                emotion[emo] = clamp(emotion.get(emo, 0.0) + d, 0.0, 1.0)

        return emotion


class TraitEngine:
    def __init__(self, root_dir: str):
        pass

    def apply_emotion_nudges(self, baseline: Dict[str, float], current: Dict[str, float], emotion: Dict[str, float], rules: Dict[str, Any]) -> Dict[str, float]:
        cfg = rules["emotion_trait"]
        coeffs = cfg["coefficients_centered"]
        cap = cfg.get("per_turn_trait_cap", 0.05)
        return_to_base = cfg.get("trait_return_to_baseline", 0.85)

        # centered emotions
        centered = {k: (emotion.get(k, 0.0) - 0.5) for k in emotion.keys()}

        updated = dict(current)
        for trait, emo_map in coeffs.items():
            delta = 0.0
            for emo, w in emo_map.items():
                delta += w * centered.get(emo, 0.0)
            delta = clamp(delta, -cap, cap)
            updated[trait] = clamp(updated.get(trait, baseline.get(trait, 0.0)) + delta, -1.0, 1.0)

        # return to baseline
        for trait in updated.keys():
            updated[trait] = baseline[trait] + (updated[trait] - baseline[trait]) * return_to_base

        return updated


class RuminationEngine:
    def __init__(self, root_dir: str):
        from pathlib import Path
        self.root = Path(root_dir)

    def run(self, session_id: str, persona, vectors, rules, memory, turn: int):
        policy = rules["update_policy"]["stable_baseline_update"]
        window = rules["update_policy"]["rumination_window_turns"]

        start_turn = turn - window + 1
        end_turn = turn

        belief_obs = memory.read_beliefs_window(session_id, start_turn, end_turn)

        stable = persona["stable"]
        beliefs = stable.get("stable_beliefs", {}).get("places", {})

        debug = {
            "updated_beliefs": [],
            "turn": turn
        }

        for belief_key, belief in beliefs.items():
            entity = belief["entity"]
            dimension = belief["dimension"]
            old_mean = belief["mean"]
            old_conf = belief["confidence"]

            # collect relevant observations
            relevant = [
                b for b in belief_obs
                if b.get("entity") == entity and b.get("dimension") == dimension
            ]

            if not relevant:
                continue

            values = [b["value"] for b in relevant]
            obs_mean = sum(values) / len(values)

            # determine support vs contradict
            support = 0
            contradict = 0

            for v in values:
                # same direction
                if (old_mean <= 0 and v <= 0) or (old_mean > 0 and v > 0):
                    support += 1
                else:
                    contradict += 1

            total = len(values)
            contradict_rate = contradict / total if total > 0 else 0

            # Eligibility rule
            if (
                contradict >= policy["min_contradict_count"]
                and contradict > support
                and contradict_rate >= policy["min_contradict_rate"]
            ):
                alpha = policy["smoothing_alpha"]

                new_mean = old_mean * (1 - alpha) + obs_mean * alpha
                belief["mean"] = max(-1.0, min(1.0, new_mean))

                # confidence update
                belief["confidence"] = max(
                    0.0,
                    min(1.0, old_conf - 0.05)
                )

                belief["last_updated_turn"] = turn

                debug["updated_beliefs"].append({
                    "belief": belief_key,
                    "old_mean": old_mean,
                    "new_mean": belief["mean"],
                    "contradict_count": contradict,
                    "support_count": support
                })

            else:
                # reinforce confidence slightly if stable
                belief["confidence"] = min(1.0, old_conf + 0.02)

        # ---- Drift metric (stable trait vector L2) ----
        baseline = vectors["trait_vector"]["baseline"]
        if "initial_baseline" not in vectors["trait_vector"]:
            vectors["trait_vector"]["initial_baseline"] = dict(baseline)

        initial = vectors["trait_vector"]["initial_baseline"]

        drift_l2 = 0.0
        for k in baseline:
            drift_l2 += (baseline[k] - initial[k]) ** 2
        drift_l2 = drift_l2 ** 0.5

        # log drift
        self._append_drift_log({
            "turn": turn,
            "stable_trait_drift_L2": drift_l2,
            "updated_beliefs": debug["updated_beliefs"]
        })

        return debug

    def _append_drift_log(self, obj):
        path = self.root / "logs" / "drift_metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")