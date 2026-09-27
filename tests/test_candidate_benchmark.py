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
            "text": "The bell rings.",
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
        {"adapters": [{"id": 0, "story_contract": "story-prose-v1"},
                      {"id": 1, "story_contract": "story-prose-v1"}], "tasks": {}},
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


@pytest.mark.parametrize("marker", [None, "legacy-json", "story-prose-v0"])
def test_prose_candidate_contract_is_checked_before_network_or_output(tmp_path, monkeypatch, marker):
    from qwen_ttrpg import annotation

    calls = []
    monkeypatch.setattr(annotation, "verify_server", lambda *args: calls.append(args))
    adapter = {"id": 0}
    if marker is not None:
        adapter["story_contract"] = marker
    output = tmp_path / "comparison"
    cases = [{"id": "story", "task": "storyteller", "prompt": [],
              "provenance": {"split": "synthetic"}}]
    with pytest.raises(ValueError, match="current prose contract"):
        run(cases, {"adapters": [adapter], "tasks": {}}, {"base": None, "legacy": 0},
            "http://localhost/v1", "fixture", output)
    assert not output.exists() and calls == []


def test_nonstory_candidates_do_not_require_a_prose_contract(tmp_path):
    cases = [{"id": "extraction", "task": "classifier", "prompt": [],
              "schema": {"type": "object"}, "provenance": {"split": "synthetic"}}]

    def fake(*args, **kwargs):
        return {"text": '{"events":[]}', "complete": True, "seconds": 1}

    report = run(cases, {"adapters": [{"id": 0}], "tasks": {}},
                 {"base": None, "extractor": 0}, "http://localhost/v1", "fixture",
                 tmp_path / "extraction", generator=fake)
    assert report["status"] == "complete"
