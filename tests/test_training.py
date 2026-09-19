import json
from pathlib import Path
import pytest
import yaml

from qwen_ttrpg.train import verify_dataset, materialize, launch_command
from qwen_ttrpg.adapters import portable_keys
from qwen_ttrpg.util import digest


def snapshot(path):
    path.mkdir()
    row = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]}
    payload = json.dumps(row) + "\n"
    for split in ["train", "validation"]:
        (path / f"{split}.tokens.jsonl").write_text(payload)
    manifest = {"sequence_len": 4096, "examples": {"train": 1, "validation": 1},
                "tokenized_sha256": {s: digest(payload) for s in ["train", "validation"]}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_modified_dataset_is_rejected(tmp_path):
    data = tmp_path / "snapshot"
    snapshot(data)
    assert verify_dataset(data)["sequence_len"] == 4096
    (data / "train.tokens.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_dataset(data)


def test_recipe_is_installed_and_materialized_without_training(tmp_path):
    import qwen_ttrpg
    data = tmp_path / "snapshot"
    snapshot(data)
    model = tmp_path / "base"
    model.mkdir()
    recipe = Path(qwen_ttrpg.__file__).parent / "recipes/qwen38-responsive-qlora.yaml"
    config, meta = materialize(data, model, tmp_path / "run", recipe, max_steps=1)
    doc = yaml.safe_load(config.read_text())
    assert doc["sequence_len"] == 4096 and doc["max_steps"] == 1
    assert meta["status"] == "prepared"
    command = launch_command(config, doc)
    assert "qwen_ttrpg.fsdp_entry" in command
    assert command[command.index("--fsdp_offload_params") + 1] == "true"
    with pytest.raises(ValueError, match="preserve"):
        materialize(data, model, tmp_path / "run", recipe)


def test_export_unwraps_exact_segments_and_rejects_collisions():
    assert portable_keys(["layer._checkpoint_wrapped_module.lora_A.weight"]) == {
        "layer._checkpoint_wrapped_module.lora_A.weight": "layer.lora_A.weight"}
    with pytest.raises(ValueError, match="collide"):
        portable_keys(["layer.weight", "layer._checkpoint_wrapped_module.weight"])
