import json

import httpx
import pytest

from qwen_ttrpg.contracts import completion_text, output_schema, validate_target
from qwen_ttrpg.story_contract import VERSION, contract
from qwen_ttrpg.story_workshop import collect, export_rows
from qwen_ttrpg.generation import generate


def test_story_targets_are_raw_text_and_other_tasks_stay_structured():
    text = '“No,” says the guard.\nWhich door are you trying?'
    assert validate_target("storyteller", {}, text) == text
    assert completion_text("storyteller", text) == text
    assert output_schema("storyteller") is None
    assert output_schema("classifier")["type"] == "object"
    assert output_schema("rules")["type"] == "object"
    assert output_schema("player")["type"] == "object"
    with pytest.raises(ValueError):
        validate_target("storyteller", {}, {"narration": text})
    with pytest.raises(ValueError):
        validate_target("storyteller", {}, json.dumps({"narration": text}))
    assert completion_text("storyteller", '"No."') == '"No."'


def test_message_helpers_preserve_an_explicit_user_brief():
    from qwen_ttrpg.tasks import messages
    from qwen_ttrpg.data import task_messages
    brief = 'The guard holds the key. Player: "May I pass?"'
    assert messages("storyteller", brief)[1]["content"] == brief
    assert task_messages("storyteller", brief)[1]["content"] == brief
    assert json.loads(messages("storyteller", {"brief": brief})[1]["content"]) == {"brief": brief}


@pytest.mark.parametrize("text", [
    '```\n{"narration":"No."}\n```', '```JSON\n{}\n```',
    '  ```Json\r\n["No."]\r\n```  ', '~~~JSON\n{}\n~~~',
    '````text\nThe guard nods.\n````', '```{"narration":"No."}```',
])
def test_whole_fences_are_rejected_without_unwrapping(text):
    from qwen_ttrpg.story_contract import validate_text
    with pytest.raises(ValueError, match="fenced"):
        validate_text(text)
    assert validate_text('"No."') == '"No."'
    assert validate_text('“The sign says JSON,” says the guard.')


def test_workshop_prose_contract_exports_unquoted_completion():
    from test_story_workshop import source, kept
    doc = collect({"sources": [source()]}, contract())
    assert isinstance(doc["cases"][0]["original"], str)
    rows = export_rows(doc, kept(doc), allow_model_reviews=True)["train"]
    assert rows[0]["completion"][0]["content"] == doc["cases"][0]["original"]
    assert "schema" not in rows[0]
    assert rows[0]["story_contract"] == VERSION


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_streaming_prose_preserves_chunks_and_completion_state(monkeypatch, finish):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        chunks = [{"choices": [{"delta": {"content": '"No."'}, "finish_reason": None}]},
                  {"choices": [{"delta": {}, "finish_reason": finish}],
                   "usage": {"completion_tokens": 3}}]
        return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n")

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    answer = generate("http://localhost/v1", "fixture", [], None, 0, 30, task="storyteller")
    assert "response_format" not in requests[0]
    assert requests[0]["stream"]
    assert answer["text"] == '"No."' and answer["complete"] == (finish == "stop")
    assert answer["usage"]["completion_tokens"] == 3
    assert answer["settings"]["output_kind"] == "plain_text"


def test_story_stream_does_not_accept_a_json_grammar():
    with pytest.raises(ValueError, match="without a JSON schema"):
        generate("http://localhost/v1", "fixture", [], None, 0, 30,
                 task="storyteller", schema={"type": "object"})


def test_benchmark_accepts_an_explicit_keeper_brief_and_rejects_json(tmp_path):
    from qwen_ttrpg.benchmark import run
    case = {"id": "brief", "task": "storyteller", "contract_version": 2,
            "prompt": [{"role": "system", "content": contract()["system"]},
                       {"role": "user", "content": 'A guard blocks the bridge. Player: "Who sent you?"'}],
            "target": '"The mayor sent me," says the guard.',
            "provenance": {"split": "synthetic"}}
    routing = {"adapters": [{"id": 0, "story_contract": VERSION}], "tasks": {"storyteller": 0}}

    def fake(*args, **kwargs):
        assert kwargs["schema"] is None
        text = '"The mayor," says the guard.' if args[3] is None else '{"narration":"The mayor."}'
        return {"text": text, "seconds": 1, "complete": True}

    report = run([case], routing, "http://localhost/v1", "fixture", tmp_path / "run", generator=fake)
    assert report["summary"]["base"]["control_passes"] == 1
    assert report["summary"]["adapter"]["control_passes"] == 0
    assert report["cases"][0]["answers"]["adapter"]["text"].startswith('{"narration"')


@pytest.mark.parametrize("runner", ["paired", "candidates"])
@pytest.mark.parametrize("invalid", ['{"narration":"No."}',
                                     '```JSON\n{"narration": unfinished\n```'])
def test_real_streamed_invalid_output_is_preserved_and_counted(tmp_path, monkeypatch, runner, invalid):
    from qwen_ttrpg import benchmark, candidate_benchmark

    routing = {"adapters": [{"id": 0, "path": "/synthetic/prose.gguf", "story_contract": VERSION}],
               "tasks": {"storyteller": 0}}
    cases = [{"id": name, "task": "storyteller", "contract_version": 2,
              "prompt": [{"role": "user", "content": name}],
              "target": '"No."', "provenance": {"split": "synthetic"}}
             for name in ["first", "next"]]

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json=routing["adapters"])
        payload = json.loads(request.content)
        is_adapter = payload["lora"][0]["scale"] == 1
        text = invalid if is_adapter and payload["messages"][-1]["content"] == "first" else '"No."'
        chunks = [{"choices": [{"delta": {"content": text[:4]}, "finish_reason": None}]},
                  {"choices": [{"delta": {"content": text[4:]}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 9, "completion_tokens": 7},
                   "timings": {"predicted_ms": 3}}]
        return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n")

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    output = tmp_path / runner
    if runner == "paired":
        report = benchmark.run(cases, routing, "http://localhost/v1", "fixture", output)
    else:
        report = candidate_benchmark.run(cases, routing, {"base": None, "adapter": 0},
                                         "http://localhost/v1", "fixture", output)

    assert report["status"] == "complete" and len(report["cases"]) == 2
    assert report["summary"]["adapter"]["complete"] == 2  # transport completion, not validity
    assert report["summary"]["adapter"]["control_passes"] == 1
    saved = json.loads((output / "results.json").read_text())
    failed = saved["cases"][0]["answers"]["adapter"]
    assert failed["text"] == invalid and failed["validation_error"]
    assert failed["checks"]["plain_text_contract"] is False
    assert failed["usage"]["completion_tokens"] == 7
    assert failed["server_timings"] == {"predicted_ms": 3}
    assert failed["seconds"] >= 0 and failed["time_to_first_text_seconds"] is not None
    assert saved["cases"][1]["answers"]["adapter"]["checks"]["plain_text_contract"]
