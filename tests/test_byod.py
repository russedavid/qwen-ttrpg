import copy
import json
from pathlib import Path

import pytest
import yaml

from qwen_ttrpg import data, experiment
from qwen_ttrpg.annotation import annotate
from qwen_ttrpg.contracts import validate_target, output_schema
from qwen_ttrpg.demo import create
from qwen_ttrpg.util import digest, file_digest, packed


class TinyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, return_dict, enable_thinking, add_generation_prompt=False):
        if add_generation_prompt:
            return list(packed(messages).encode()) + [1]
        return list(packed(messages[:-1]).encode()) + [1] + list(messages[-1]["content"].encode()) + [0]

    def decode(self, tokens):
        return bytes(t for t in tokens if t > 1).decode()


@pytest.fixture
def config(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{}')
    path = create(tmp_path / "example", model, runtime=tmp_path / "runtime", base_gguf=tmp_path / "base.gguf")
    doc = yaml.safe_load(path.read_text())
    for task in doc["tasks"]:
        doc["tasks"][task]["sequence_len"] = 16384
    path.write_text(yaml.safe_dump(doc))
    return data.load_config(path)


def test_original_example_builds_three_source_checked_datasets(config):
    queue = data.load_review(config)
    assert {c["task"] for c in queue["candidates"]} == set(data.TASKS)
    assert all(c["review"]["origin"] == "synthetic" for c in queue["candidates"])
    report = data.prepare_all(config, TinyTokenizer())
    assert report["tasks"]["classifier"]["examples"] == {"train": 1, "validation": 1, "test": 1}
    assert report["tasks"]["storyteller"]["examples"]["train"] == 2
    assert data.prepare_all(config, TinyTokenizer()) == report
    plan = experiment.plan(config)
    assert len(plan["training_commands"]) == 3
    assert all("--resume" in c for c in plan["training_commands"].values())


def test_changed_sources_and_context_cannot_reuse_reviews(config):
    source = data.source_path(config, config["conversations"][0]["path"])
    source.write_text(source.read_text().replace("wooden", "metal"))
    with pytest.raises(ValueError, match="changed"):
        data.load_review(config)


def test_review_input_edits_and_narrator_rewrites_are_rejected(config):
    path = data.review_path(config)
    original = json.loads(path.read_text())
    edited = copy.deepcopy(original)
    edited["candidates"][0]["input"]["target_turns"][0]["text"] = "Rewritten evidence"
    data.write_json(path, edited)
    with pytest.raises(ValueError, match="inputs changed"):
        data.load_review(config)
    edited = copy.deepcopy(original)
    next(c for c in edited["candidates"] if c["task"] == "storyteller")["target"]["narration"] = "Invented response"
    data.write_json(path, edited)
    with pytest.raises(ValueError, match="source wording"):
        data.load_review(config)


def test_target_change_invalidates_review(config):
    doc = data.load_review(config)
    next(c for c in doc["candidates"] if c["task"] == "classifier")["target"]["events"][0]["value"] = "edited"
    data.write_json(data.review_path(config), doc)
    with pytest.raises(ValueError, match="Target changed"):
        data.prepare_all(config, TinyTokenizer())


def test_source_group_leakage_is_rejected_before_training(config):
    config["conversations"][1]["group"] = config["conversations"][0]["group"]
    with pytest.raises(ValueError, match="group crosses"):
        data.build_candidates(config)


def test_alignment_metadata_survives_without_displacing_dialogue(config):
    source = data.source_path(config, config["conversations"][0]["path"])
    original = json.loads(source.read_text())
    original["turns"][0]["metadata"] = {"words": [{"word": "alignment-only-marker", "start": 1.25}]}
    source.write_text(json.dumps(original))
    cases = data.build_candidates(config)["candidates"]
    affected = [c for c in cases if c["provenance"]["split"] == "train" and c["task"] != "rules"]
    assert affected
    for case in affected:
        assert "alignment-only-marker" not in packed(case["prompt"])
        assert "alignment-only-marker" in packed(case["provenance"]["source_turn_metadata"])


def test_synthetic_reviews_need_opt_in(config):
    config["allow_synthetic"] = False
    with pytest.raises(ValueError, match="allow_synthetic"):
        data.prepare_all(config, TinyTokenizer())


def test_tokenizer_change_invalidates_prepared_data(config):
    data.prepare_all(config, TinyTokenizer())
    (Path(config["model"]) / "config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="differs"):
        data.prepare_all(config, TinyTokenizer())


def test_classification_requires_supported_events_and_real_prior_links(config):
    case = next(c for c in data.load_review(config)["candidates"] if c["task"] == "classifier")
    target = copy.deepcopy(case["target"])
    target["events"][0]["evidence"][0]["quote"] = "Words absent from input"
    with pytest.raises(ValueError, match="quotation"):
        validate_target("classifier", case["input"], target)
    target = copy.deepcopy(case["target"])
    target["events"][0]["supersedes"] = "nonexistent"
    with pytest.raises(ValueError, match="prior events"):
        validate_target("classifier", case["input"], target)
    target = copy.deepcopy(case["target"])
    target["events"][0].update(kind="resource", stage="hypothetical", value=None, delta=-1)
    with pytest.raises(ValueError, match="established"):
        validate_target("classifier", case["input"], target)


def test_rules_quotes_and_explicit_tool_inputs_are_enforced(config):
    case = next(c for c in data.load_review(config)["candidates"] if c["task"] == "rules")
    bad = copy.deepcopy(case["target"])
    bad["citations"][0]["quote"] = "Invented rule"
    with pytest.raises(ValueError, match="quotation"):
        validate_target("rules", case["input"], bad)
    body = copy.deepcopy(case["input"])
    body.update(tools=[{"name": "compare", "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False}}], tool_inputs={"compare": {"value": 3}})
    target = copy.deepcopy(case["target"])
    target["calculation"] = {"tool": "compare", "arguments": {"value": 3}}
    assert validate_target("rules", body, target)["calculation"]["arguments"]["value"] == 3
    target["calculation"]["arguments"]["value"] = 4
    with pytest.raises(ValueError, match="explicit"):
        validate_target("rules", body, target)


def test_local_label_generation_is_never_implicitly_reviewed(config):
    doc = data.load_review(config)
    case = next(c for c in doc["candidates"] if c["task"] == "classifier")
    expected = copy.deepcopy(case["target"])
    case.update(target=None, review={"verdict": "pending"})
    data.write_json(data.review_path(config), doc)
    calls = []

    def generate(*args, **kwargs):
        calls.append(args)
        return {"text": packed(expected), "complete": True}

    result = annotate(config, task="classifier", routing={"adapters": [{"id": 0}]},
                      url="http://localhost/v1", model="fixture", limit=1, generator=generate)
    assert result["proposed"] == 1 and calls[0][3] is None
    updated = next(c for c in data.load_review(config)["candidates"] if c["id"] == case["id"])
    assert updated["review"]["verdict"] == "pending" and updated["label_origin"] == "model-proposed"


def test_source_data_cannot_be_generated_inside_git(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="outside Git"):
        create(tmp_path / "outputs", tmp_path / "model")


def test_schema_serialization_keeps_explicit_nullable_fields():
    schema = output_schema("classifier")
    event = schema["$defs"]["Event"]
    assert list(event["properties"]) == sorted(event["properties"])
    assert set(event["required"]) == set(event["properties"])


def test_config_preserves_virtual_environment_interpreter_path(tmp_path):
    import sys
    interpreter = tmp_path / "training-env/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    path = create(tmp_path / "fixture", tmp_path / "model", python=interpreter)
    config = data.load_config(path)
    assert config["python"] == str(interpreter)
    assert config["python"] != str(interpreter.resolve())


def test_quoted_false_cannot_enable_model_review(config):
    path = Path(config["_path"])
    doc = yaml.safe_load(path.read_text())
    doc["allow_model_reviews"] = "false"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="YAML boolean"):
        data.load_config(path)


def fake_executor(calls, fail=None):
    def execute(argv, log):
        calls.append(argv)
        module = argv[argv.index("-m") + 1]
        if module == fail:
            raise RuntimeError("Injected interruption")
        output = Path(argv[argv.index("--output") + 1])
        if module.endswith(".train"):
            adapter = output / "portable-adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture-weight-placeholder")
            (adapter / "adapter_config.json").write_text('{}')
            data.write_json(output / "run.json", {"status": "trained", "portable_export": {"export_sha256": file_digest(adapter / "adapter_model.safetensors")}})
        elif module.endswith(".evaluate_adapter"):
            data.write_json(output, {"status": "complete", "reload_validation": {"tensors_loaded_and_compared": 1}})
        elif module.endswith(".convert_adapter"):
            output.write_bytes(b"fixture-format-placeholder")
            data.write_json(output.with_suffix(".conversion.json"), {"bytes": output.stat().st_size,
                "gguf_sha256": file_digest(output), "factor_permutation_max_errors": [0.0, 0.0]})
        else:
            raise AssertionError(module)
    return execute


def test_pipeline_resumes_completed_stages_and_rejects_changed_artifacts(config):
    data.prepare_all(config, TinyTokenizer())
    first = []
    with pytest.raises(RuntimeError, match="Injected"):
        experiment.run(config, through="convert", executor=fake_executor(first, "qwen_ttrpg.evaluate_adapter"))
    assert len(first) == 2
    second = []
    state = experiment.run(config, through="convert", executor=fake_executor(second))
    assert len(second) == 8  # successful classifier training was not repeated
    assert all(s["status"] == "complete" for s in state["stages"].values())
    assert len(state["stages"]["classifier:reload"]["attempts"]) == 2
    again = []
    experiment.run(config, through="convert", executor=fake_executor(again))
    assert again == []
    path = next(iter(state["stages"]["classifier:convert"]["artifacts"]))
    Path(path).write_text('changed')
    with pytest.raises(ValueError, match="artifact"):
        experiment.run(config, through="convert", executor=fake_executor([]))


def test_preparation_failure_leaves_no_partial_published_snapshot(config):
    for task in config["tasks"]:
        config["tasks"][task]["sequence_len"] = 4
    with pytest.raises(ValueError, match="Nonempty"):
        data.prepare_all(config, TinyTokenizer())
    assert not (Path(config["workspace"]) / "datasets").exists()
    for task in config["tasks"]:
        config["tasks"][task]["sequence_len"] = 16384
    assert data.prepare_all(config, TinyTokenizer())["tasks"]["rules"]["examples"]["train"] == 1
