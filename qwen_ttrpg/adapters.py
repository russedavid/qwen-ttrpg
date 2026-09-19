"""Portable adapter exports and strict reload checks for local experiments."""

from pathlib import Path
import json
import shutil
import tempfile

from .util import digest, now, file_digest


def portable_keys(keys):
    """Remove only PyTorch activation-checkpoint wrapper path components."""
    mapped = {
        key: ".".join(p for p in key.split(".") if p != "_checkpoint_wrapped_module")
        for key in keys
    }
    if len(set(mapped.values())) != len(mapped):
        raise ValueError("Checkpoint unwrapping would collide with another tensor.")
    return mapped


def export_portable(source, destination):
    from safetensors.torch import load_file, save_file

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError(
            "Choose a new adapter destination; original exports are preserved."
        )
    files = list(source.glob("adapter_model*.safetensors"))
    if len(files) != 1:
        raise ValueError("Expected one complete adapter tensor file.")
    state = load_file(str(files[0]), device="cpu")
    mapping = portable_keys(state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".exporting-", dir=destination.parent))
    try:
        shutil.copyfile(source / "adapter_config.json", temporary / "adapter_config.json")
        save_file({mapping[k]: v for k, v in state.items()}, str(temporary / "adapter_model.safetensors"))
        report = {
            "created": now(), "source_sha256": file_digest(files[0]),
            "export_sha256": file_digest(temporary / "adapter_model.safetensors"),
            "tensors": len(state), "renamed": sum(k != v for k, v in mapping.items()),
            "transformation": "Remove activation-checkpoint wrappers; tensor values unchanged.",
            "reload_verified": False,
        }
        (temporary / "export.json").write_text(json.dumps(report, indent=2))
        if destination.exists():
            raise ValueError("Adapter destination appeared during export.")
        temporary.rename(destination)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def strict_load(base, directory):
    """Check names, shapes, load result and actual values, never just file presence."""
    import torch
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import (
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file

    directory = Path(directory)
    config = PeftConfig.from_pretrained(str(directory), local_files_only=True)
    config.inference_mode = True
    model = get_peft_model(base, config)
    state = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    expected = get_peft_model_state_dict(model)
    if set(expected) != set(state):
        raise ValueError(
            f"Adapter key mismatch: missing={sorted(set(expected) - set(state))[:3]}, unexpected={sorted(set(state) - set(expected))[:3]}"
        )
    if any(expected[k].shape != state[k].shape for k in state):
        raise ValueError("Adapter tensor shapes differ from the base model.")
    result = set_peft_model_state_dict(model, state)
    missing = [k for k in result.missing_keys if "lora_" in k]
    if missing or result.unexpected_keys:
        raise ValueError(
            f"Incomplete adapter reload: {missing[:3]}, {result.unexpected_keys[:3]}"
        )
    actual = get_peft_model_state_dict(model)
    if any(
        not torch.equal(actual[k].detach().cpu(), state[k].to(actual[k].dtype))
        for k in state
    ):
        raise ValueError("Loaded adapter values differ from saved weights.")
    model.eval()
    return model, {
        "tensors_loaded_and_compared": len(state),
        "missing_adapter_keys": 0,
        "unexpected_keys": 0,
        "adapter_sha256": file_digest(directory / "adapter_model.safetensors"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(json.dumps(export_portable(args.source, args.destination), indent=2))
