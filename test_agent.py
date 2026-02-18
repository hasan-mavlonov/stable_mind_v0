from pathlib import Path
import shutil

from stablemind.agent import StableMindAgent
from stablemind.llm import LLMClient


DATA_DIRS = ["memory", "persona", "prompts", "rules", "state"]


def _build_temp_root(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    root.mkdir()

    for rel in DATA_DIRS:
        src = Path(rel)
        dst = root / rel
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    return root


def test_step_runs_without_openai_dependencies(tmp_path):
    root = _build_temp_root(tmp_path)

    agent = StableMindAgent(root_dir=str(root), llm_client=None)
    result = agent.step("Hey, I got accepted to my dream school!", session_id="s1")

    assert result.turn == 4
    assert result.text
    assert "major_achievement" in result.debug["events"]


def test_step_tracks_entity_focus_and_notes(tmp_path):
    root = _build_temp_root(tmp_path)

    agent = StableMindAgent(root_dir=str(root), llm_client=LLMClient.dummy())
    result = agent.step(
        "I visited France Cafe again today. It was very quiet this time.",
        session_id="s2",
    )

    assert result.debug["events"] == []

    dynamic = agent.state.load_persona()["dynamic"]
    assert dynamic["last_entity_focus"] == "France Cafe"
    assert any("quiet/calm" in note for note in dynamic["recent_discoveries"])
