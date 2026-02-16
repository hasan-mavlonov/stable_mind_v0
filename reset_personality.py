#!/usr/bin/env python3
"""
Reset StableMind personality/state back to step 0.

- Keeps persona/immutable_traits.json unchanged.
- Resets state/counters.json, state/vectors.json
- Resets persona/dynamic_traits.json (fresh)
- Optionally clears persona/stable_traits.json stable_beliefs + evidence
- Optionally wipes memory/ + logs/ jsonl files

Usage:
  python reset_personality.py --root . --wipe-memory --clear-beliefs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


EMOTIONS_8 = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + ".bak")
    # overwrite old .bak to keep it simple
    bak.write_bytes(path.read_bytes())


def wipe_jsonl(dir_path: Path) -> List[Path]:
    deleted: List[Path] = []
    if not dir_path.exists():
        return deleted
    for p in dir_path.glob("*.jsonl"):
        p.unlink(missing_ok=True)
        deleted.append(p)
    return deleted


def clamp_trait_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[k] = clamp(float(v), 0.0, 1.0)
        except Exception:
            # if it's not numeric, keep it as-is (but this should NOT happen for traits)
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root (contains persona/, state/, rules/, memory/)")
    ap.add_argument("--wipe-memory", action="store_true", help="Delete memory/*.jsonl and logs/*.jsonl")
    ap.add_argument("--clear-beliefs", action="store_true", help="Clear persona/stable_traits.json stable_beliefs back to empty")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    # Paths
    persona_dir = root / "persona"
    state_dir = root / "state"
    memory_dir = root / "memory"
    logs_dir = root / "logs"

    immutable_path = persona_dir / "immutable_traits.json"
    stable_path = persona_dir / "stable_traits.json"
    dynamic_path = persona_dir / "dynamic_traits.json"
    anchors_path = persona_dir / "anchors.json"

    counters_path = state_dir / "counters.json"
    vectors_path = state_dir / "vectors.json"

    # --- Read existing files (immutable must exist)
    immutable = read_json(immutable_path)
    stable = read_json(stable_path)
    anchors = read_json(anchors_path)

    # Counters: preserve rumination_window_size if present, else default 20
    counters_old = read_json(counters_path)
    rum_window = int(counters_old.get("rumination_window_size", 20))

    # Vectors: keep baseline keys, but clamp to [0,1] because negatives are invalid for trait scores
    vectors_old = read_json(vectors_path)
    trait_vec = vectors_old.get("trait_vector", {})
    baseline_old = trait_vec.get("baseline", {})

    baseline = clamp_trait_dict(baseline_old)

    # If baseline was empty, fail loudly: you need baseline traits defined.
    if not baseline:
        raise RuntimeError(
            "trait_vector.baseline is empty or invalid. Define baseline trait scores in state/vectors.json first."
        )

    # Emotion neutral: 0.5 for all 8 emotions
    emotion_neutral = {e: 0.5 for e in EMOTIONS_8}

    # --- Build reset files
    counters_new = {
        "current_turn": 0,
        "last_rumination_turn": 0,
        "rumination_window_size": rum_window,
        "turns_since_last_rumination": 0,
    }

    vectors_new = {
        "turn": 0,
        "trait_vector": {
            "baseline": baseline,
            "current": dict(baseline),
            "initial_baseline": dict(baseline),
        },
        "emotion_vector": emotion_neutral,
    }

    dynamic_new = {
        "favorite_spot": None,
        "recent_hobbies": [],
        "current_focus": [],
        "recent_dislikes": [],
        "recent_discoveries": [],
        "last_visited_places": [],
        "last_entity_focus": None,
    }

    if args.clear_beliefs:
        # keep stable identity, wipe beliefs
        stable = dict(stable)
        stable["stable_beliefs"] = {"places": {}}

    # --- Backups (so you can revert easily if you regret it)
    for p in [stable_path, dynamic_path, counters_path, vectors_path]:
        backup_file(p)

    # --- Write resets
    # immutable stays untouched by design
    write_json(counters_path, counters_new)
    write_json(vectors_path, vectors_new)
    write_json(dynamic_path, dynamic_new)
    write_json(stable_path, stable)

    deleted = []
    if args.wipe_memory:
        deleted += wipe_jsonl(memory_dir)
        deleted += wipe_jsonl(logs_dir)

    print("✅ Reset complete.")
    print(f"- Kept immutable: {immutable_path}")
    print(f"- Wrote counters: {counters_path}")
    print(f"- Wrote vectors:  {vectors_path}")
    print(f"- Wrote dynamic:  {dynamic_path}")
    print(f"- Updated stable: {stable_path} (clear_beliefs={args.clear_beliefs})")
    if args.wipe_memory:
        print(f"- Wiped jsonl files: {len(deleted)}")
        for p in deleted:
            print(f"  - {p}")


if __name__ == "__main__":
    main()