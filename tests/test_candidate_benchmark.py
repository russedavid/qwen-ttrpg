import json
import pytest

from qwen_ttrpg.candidate_benchmark import run


def test_three_candidates_have_identical_sampling_and_blind_metadata_is_separate(
    tmp_path,
):
    calls = []

    def fake(url, model, messages, adapter, count, max_tokens, **kwargs):
        calls.append((messages, adapter, count, max_tokens, kwargs))
        return {
            "text": '{"narration":"The bell rings."}',
            "complete": True,
            "seconds": 1,
            "settings": {"adapter": adapter},
        }

    cases = [
        {
            "id": str(i),
            "task": "storyteller",
            "prompt": [{"role": "user", "content": "A scene."}],
            "provenance": {"split": "test"},
        }
        for i in range(6)
    ]
    result = run(
        cases,
        {"adapters": [{"id": 0}, {"id": 1}], "tasks": {}},
        {"base": None, "old": 0, "new": 1},
        "http://localhost/v1",
        "model",
        tmp_path / "compare",
        generator=fake,
    )
    assert len(calls) == 18 and all(c[2:4] == (2, 700) for c in calls)
    assert {c[4]["seed"] for c in calls} == {42}
    blind = json.loads((tmp_path / "compare/blind.json").read_text())
    assert all(set(c["answers"]) == {"A", "B", "C"} for c in blind)
    text = (tmp_path / "compare/blind.json").read_text()
    assert (
        "adapter" not in text
        and "seconds" not in text
        and "generation_order" not in text
    )
    assert "<h3>C</h3>" in (tmp_path / "compare/review.html").read_text()
    assert all(v["complete"] == 6 for v in result["summary"].values())
    cases[0]["provenance"]["split"] = "train"
    with pytest.raises(ValueError, match="training split"):
        run(
            cases,
            {"adapters": [{"id": 0}], "tasks": {}},
            {"base": None, "new": 0},
            "http://localhost/v1",
            "model",
            tmp_path / "invalid",
            generator=fake,
        )
