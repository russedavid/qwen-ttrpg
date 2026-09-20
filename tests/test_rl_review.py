from copy import deepcopy
import json

import pytest

from qwen_ttrpg.rl_review import render


def report(label):
    return {"status": "complete", "candidate": label, "dataset": {"hash": "original"},
            "settings": {"seeds": [42]}, "summary": {}, "cases": [{
                "id": "original", "generation_seed": 42, "family": "authored",
                "prompt": [{"role": "system", "content": "source only"},
                           {"role": "user", "content": json.dumps({"task": "<script>not executable</script>"})}],
                "source_context": {}, "expected": {"value": 3}, "final": {"value": 3},
                "assessment": {"success": True, "tool_calls": 1, "seconds": 0.5},
                "trace": [{"raw": "<img src=x onerror=alert(1)>"}]}]}


def test_review_requires_aligned_candidate_runs_and_escapes_model_text(tmp_path):
    paths = []
    for label in ["base", "sft", "grpo"]:
        p = tmp_path / f"{label}.json"
        p.write_text(json.dumps(report(label)))
        paths.append(p)
    render(paths, tmp_path / "review")
    content = (tmp_path / "review/review.html").read_text()
    assert "&lt;script&gt;" in content and "<script>" not in content
    assert "&lt;img" in content and "<img " not in content
    assert all(f"<h3>{label}</h3>" in content for label in ["base", "sft", "grpo"])
    changed = report("grpo")
    changed["cases"][0]["generation_seed"] = 77
    paths[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="case/seed"):
        render(paths, tmp_path / "bad")


def test_review_refuses_different_snapshots_or_generation_settings(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps(report("base")))
    for key, value in [("dataset", {"hash": "different"}), ("settings", {"seeds": [1]})]:
        altered = report("adapter")
        altered[key] = value
        right.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="same data snapshot"):
            render([left, right], tmp_path / "comparison")
