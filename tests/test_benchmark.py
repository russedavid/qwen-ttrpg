import json
import pytest
import httpx

from qwen_ttrpg.benchmark import run, local_endpoint, score
from qwen_ttrpg import generation


def cases():
    return [{"id": str(i), "task": "storyteller", "prompt": [{"role": "user", "content": "Describe a quiet harbor."}],
             "provenance": {"split": "synthetic"}, "checks": {"nonempty": ["narration"]}} for i in range(8)]


def test_blinding_does_not_leak_adapter_ids_or_timing(tmp_path):
    def fake(*args, **kwargs):
        return {"text": '{"narration":"<A lantern glows.>"}', "seconds": 1.5,
                "time_to_first_text_seconds": .1, "complete": True,
                "settings": {"adapter": args[3]}}
    routing = {"adapters": [{"id": 0}], "tasks": {"storyteller": 0}}
    report = run(cases(), routing, "http://127.0.0.1:8091/v1", "local", tmp_path / "run", generator=fake)
    assert report["summary"]["adapter"]["complete"] == 8
    page = (tmp_path / "run/review.html").read_text()
    assert "&lt;A lantern glows.&gt;" in page and '"adapter"' not in page
    keys = json.loads((tmp_path / "run/review-key.json").read_text())
    assert {k["A"] for k in keys.values()} == {"base", "adapter"}
    assert any(list(k.values()) != r["generation_order"] for k, r in zip(keys.values(), report["cases"]))
    with pytest.raises(ValueError, match="new benchmark"):
        run(cases(), routing, "http://127.0.0.1/v1", "local", tmp_path / "run", generator=fake)


@pytest.mark.parametrize("url", ["https://example.org/v1", "http://192.0.2.1/v1", "file:///tmp/a", "http://u:p@localhost/v1"])
def test_private_cases_cannot_be_sent_to_a_remote_endpoint(url):
    with pytest.raises(ValueError):
        local_endpoint(url)


def test_training_split_and_missing_candidate_rejected(tmp_path):
    items = cases()
    items[0]["provenance"]["split"] = "train"
    with pytest.raises(ValueError, match="held-out"):
        run(items, {"adapters": [], "tasks": {}}, "http://localhost/v1", "m", tmp_path / "out")


def test_contract_checks_are_explicit():
    assert all(score('{"events": []}', {"equals": {"events": []}}).values())
    assert not all(score('{"events": [1]}', {"equals": {"events": []}}).values())
    assert not all(score('invalid', {"equals": {"events": []}}).values())


def test_truncated_outputs_remain_failures_in_a_completed_study(tmp_path):
    def incomplete(*args, **kwargs):
        return {"text": '{"narration":"A lantern glows."}', "seconds": 1,
                "complete": False, "finish_reason": "length"}
    routing = {"adapters": [{"id": 0}], "tasks": {"storyteller": 0}}
    report = run(cases(), routing, "http://localhost/v1", "fixture", tmp_path / "study", generator=incomplete)
    assert report["status"] == "complete" and len(report["cases"]) == 8
    for summary in report["summary"].values():
        assert summary["complete"] == 0 and summary["control_passes"] == 0
    assert (tmp_path / "study/review.html").read_text().count("Incomplete response.") == 16


def test_stream_request_disables_other_adapters_and_cache(monkeypatch):
    observed = []
    def handler(request):
        observed.append(json.loads(request.content))
        body = 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
        body += 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n\n'
        body += 'data: [DONE]\n\n'
        return httpx.Response(200, text=body)
    original = httpx.Client
    monkeypatch.setattr(generation.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    answer = generation.generate("http://localhost/v1", "local", [], 1, 3, 128)
    assert answer["complete"] and answer["text"] == "ok"
    assert answer["usage"]["completion_tokens"] == 1
    assert [v["scale"] for v in observed[0]["lora"]] == [0, 1, 0]
    assert observed[0]["cache_prompt"] is False
