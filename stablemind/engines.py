from typing import Any, Dict, List


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# =====================================================
# Emotion Engine
# =====================================================

class EmotionEngine:
    def __init__(self, root_dir: str):
        pass

    def update(self, emotion: Dict[str, float], events: List[str], rules: Dict[str, Any]) -> Dict[str, float]:
        deltas = rules["event_emotion"]
        decay = rules["emotion_trait"].get("emotion_decay", 0.75)

        # decay toward neutral (0.5)
        neutral = 0.5
        emotion = {
            k: clamp(neutral + (v - neutral) * decay)
            for k, v in emotion.items()
        }

        # apply event deltas
        for ev in events:
            if ev not in deltas:
                continue
            for emo, d in deltas[ev].items():
                emotion[emo] = clamp(emotion.get(emo, 0.5) + d)

        return emotion


# =====================================================
# Trait Engine
# =====================================================

class TraitEngine:
    def __init__(self, root_dir: str):
        pass

    def apply_emotion_nudges(self, baseline, current, emotion, rules):
        coeffs = rules["emotion_trait_nudges"]["coefficients_centered"]
        per_turn_cap = rules["emotion_trait_nudges"]["per_turn_trait_cap"]
        return_to_base = rules["emotion_trait_nudges"]["trait_return_to_baseline"]

        updated = dict(current)

        # Apply emotion-driven deltas
        for trait, weights in coeffs.items():
            if trait not in baseline or trait not in updated:
                continue

            delta = 0.0
            for emo, w in weights.items():
                delta += w * (emotion.get(emo, 0.5) - 0.5)

            delta = clamp(delta, -per_turn_cap, per_turn_cap)

            updated[trait] = clamp(updated[trait] + delta)

        # Return slowly toward baseline
        for trait in updated:
            if trait not in baseline:
                continue

            updated[trait] = clamp(
                baseline[trait] + (updated[trait] - baseline[trait]) * return_to_base
            )

        return updated


# =====================================================
# Rumination Engine
# =====================================================

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
        places = stable.setdefault("stable_beliefs", {}).setdefault("places", {})

        debug = {
            "updated_beliefs": [],
            "turn": turn
        }

        # -------------------------------------------------
        # 1) CREATE NEW BELIEFS (if enough observations)
        # -------------------------------------------------

        groups = {}
        for b in belief_obs:
            entity = b.get("entity")
            dimension = b.get("dimension")
            if not entity or not dimension:
                continue
            groups.setdefault((entity, dimension), []).append(b)

        MIN_OBS_TO_CREATE = 2

        for (entity, dimension), obs_list in groups.items():
            belief_key = f"{entity.lower().replace(' ', '_')}_{dimension}"

            if belief_key in places:
                continue

            if len(obs_list) < MIN_OBS_TO_CREATE:
                continue

            values = [o["value"] for o in obs_list if "value" in o]
            if not values:
                continue

            mean = sum(values) / len(values)

            places[belief_key] = {
                "entity": entity,
                "dimension": dimension,
                "mean": clamp(mean),
                "confidence": 0.6,
                "last_updated_turn": turn,
                "evidence": {
                    "support_count_window": len(values),
                    "contradict_count_window": 0,
                    "window_size": len(values)
                }
            }

            debug["updated_beliefs"].append({
                "belief": belief_key,
                "created": True,
                "new_mean": mean,
                "obs_count": len(values)
            })

        # -------------------------------------------------
        # 2) UPDATE EXISTING BELIEFS
        # -------------------------------------------------

        for belief_key, belief in places.items():
            entity = belief["entity"]
            dimension = belief["dimension"]

            relevant = [
                b for b in belief_obs
                if b.get("entity") == entity and b.get("dimension") == dimension
            ]

            if not relevant:
                continue

            values = [b["value"] for b in relevant]
            obs_mean = sum(values) / len(values)

            old_mean = belief["mean"]
            old_conf = belief["confidence"]

            # contradiction logic for [0,1] scale
            contradict = sum(1 for v in values if abs(v - old_mean) > 0.3)
            support = len(values) - contradict

            contradict_rate = contradict / len(values)

            if (
                contradict >= policy["min_contradict_count"]
                and contradict_rate >= policy["min_contradict_rate"]
            ):
                alpha = policy["smoothing_alpha"]
                new_mean = old_mean * (1 - alpha) + obs_mean * alpha

                belief["mean"] = clamp(new_mean)
                belief["confidence"] = clamp(old_conf - 0.05)
                belief["last_updated_turn"] = turn

                debug["updated_beliefs"].append({
                    "belief": belief_key,
                    "old_mean": old_mean,
                    "new_mean": belief["mean"],
                    "contradict_count": contradict,
                    "support_count": support
                })
            else:
                belief["confidence"] = clamp(old_conf + 0.02)

        # -------------------------------------------------
        # 3) DRIFT METRIC
        # -------------------------------------------------

        baseline = vectors["trait_vector"]["baseline"]
        initial = vectors["trait_vector"].setdefault(
            "initial_baseline",
            dict(baseline)
        )

        drift_l2 = sum(
            (baseline[k] - initial[k]) ** 2
            for k in baseline
        ) ** 0.5

        self._append_drift_log({
            "turn": turn,
            "stable_trait_drift_L2": drift_l2,
            "updated_beliefs": debug["updated_beliefs"]
        })

        return debug

    def _append_drift_log(self, obj):
        import json
        path = self.root / "logs" / "drift_metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")