"""Run the pinned local Axolotl recipe against a verified dataset snapshot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import re
import math

from .util import digest, now, file_digest


def verify_dataset(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["examples"].get("train") or not manifest["examples"].get(
        "validation"
    ):
        raise ValueError("Both training and held-out validation data are required.")
    required = {"train", "validation"}
    if not required <= manifest["tokenized_sha256"].keys():
        raise ValueError("Both training and validation hashes are required.")
    if type(manifest["sequence_len"]) is not int or manifest["sequence_len"] < 1:
        raise ValueError("Invalid sequence length.")
    for split, expected in manifest["tokenized_sha256"].items():
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown dataset split.")
        path = directory / f"{split}.tokens.jsonl"
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"{split} dataset changed after preparation.")
        count = 0
        for line in path.read_text().splitlines():
            count += 1
            row = json.loads(line)
            ids, mask, labels = row["input_ids"], row["attention_mask"], row["labels"]
            if set(row) != {"input_ids", "attention_mask", "labels"} or not len(
                ids
            ) == len(mask) == len(labels):
                raise ValueError("Malformed tokenized example.")
            if (
                not any(v != -100 for v in labels)
                or len(ids) > manifest["sequence_len"]
            ):
                raise ValueError("Empty target loss or an overlength example.")
            if any(label not in {-100, token} for token, label in zip(ids, labels)):
                raise ValueError("Labels differ from the target token IDs.")
            if any(type(i) is not int or i < 0 for i in ids) or any(m != 1 for m in mask):
                raise ValueError("Expected unpadded integer token IDs and an all-one mask.")
            if not labels or labels[0] != -100:
                raise ValueError("Assistant targets must follow a masked prompt.")
            boundary = next(i for i, label in enumerate(labels) if label != -100)
            if any(label == -100 for label in labels[boundary:]):
                raise ValueError("Completion loss must be a contiguous suffix.")
        if count != manifest["examples"].get(split, 0):
            raise ValueError("Dataset row count differs from its manifest.")
    for split, expected in manifest.get("sources_sha256", {}).items():
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown source split.")
        if digest((directory / f"{split}.sources.jsonl").read_bytes()) != expected:
            raise ValueError("Source examples changed after preparation.")
    return manifest


def materialize(dataset, model, output, recipe, max_steps=None, epochs=None):
    import yaml

    dataset = Path(dataset).resolve()
    model = Path(model).resolve(strict=True)
    output = Path(output).resolve()
    manifest = verify_dataset(dataset)
    if output.exists():
        raise ValueError("Choose a new run directory to preserve previous runs.")
    config = yaml.safe_load(Path(recipe).read_text())
    if epochs is not None:
        if not 0 < epochs <= 20:
            raise ValueError("Choose a positive epoch count up to 20.")
        config["num_epochs"] = epochs
    config.update(
        base_model=str(model),
        tokenizer_config=str(model),
        output_dir=str(output / "adapter"),
        dataset_prepared_path=str(output / "prepared"),
        sequence_len=manifest["sequence_len"],
    )
    config["datasets"] = [
        {"path": str(dataset / "train.tokens.jsonl"), "ds_type": "json", "type": None}
    ]
    config["test_datasets"] = [
        {
            "path": str(dataset / "validation.tokens.jsonl"),
            "ds_type": "json",
            "type": None,
        }
    ]
    if max_steps is not None:
        config.update(
            max_steps=max_steps,
            gradient_accumulation_steps=1,
            save_steps=max_steps,
            eval_steps=max_steps,
        )
        # Even a one-step smoke run needs a nonzero learning-rate update. The
        # trainer otherwise rounds a fractional warm-up up to the entire run.
        config["warmup_steps"] = min(config.get("warmup_steps") or math.ceil(config.get("warmup_ratio", 0) * max_steps), max_steps - 1)
        config["warmup_ratio"] = 0.0
    output.mkdir(parents=True)
    path = output / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    versions = {}
    for name in [
        "torch",
        "torchvision",
        "axolotl",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "cut-cross-entropy",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    metadata = {
        "created": now(),
        "status": "prepared",
        "dataset_manifest": manifest,
        "config_sha256": digest(path.read_bytes()),
        "model_directory": str(model),
        "python": platform.python_version(),
        "packages": versions,
        "max_steps": max_steps,
        "recipe_sha256": file_digest(recipe),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    return path, metadata


def verify_adapter(output):
    from safetensors import safe_open

    files = list((Path(output) / "adapter").glob("adapter_model*.safetensors"))
    if not files or not (Path(output) / "adapter/adapter_config.json").exists():
        raise ValueError("Training exited without a reloadable adapter artifact.")
    changed = False
    tensors = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                tensors += 1
                import torch

                if not torch.isfinite(tensor).all():
                    raise ValueError("The saved adapter contains nonfinite weights.")
                if "lora_B" in key and tensor.abs().max().item() > 0:
                    changed = True
    if not changed:
        raise ValueError(
            "No nonzero LoRA B weights found; optimizer updates have not been demonstrated."
        )
    return {
        "adapter_files": [str(p) for p in files],
        "tensors": tensors,
        "nonzero_lora_b": True,
    }


def launch_command(config_path, config):
    fsdp = config["fsdp_config"]
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--use_fsdp",
        "--num_processes",
        "2",
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
    ]
    for name in [
        "fsdp_version",
        "offload_params",
        "state_dict_type",
        "auto_wrap_policy",
        "transformer_layer_cls_to_wrap",
        "cpu_ram_efficient_loading",
        "activation_checkpointing",
    ]:
        if name not in fsdp:
            continue
        option = name if name.startswith("fsdp_") else "fsdp_" + name
        value = (
            str(fsdp[name]).lower() if isinstance(fsdp[name], bool) else str(fsdp[name])
        )
        command += ["--" + option, value]
    return command + ["-m", "qwen_ttrpg.fsdp_entry", str(config_path)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--recipe",
        default=str(
            Path(__file__).resolve().parent / "recipes/qwen38-responsive-qlora.yaml"
        ),
    )
    p.add_argument("--max-steps", type=int)
    p.add_argument("--epochs", type=float)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse a matching run; resume the latest saved optimizer checkpoint when available")
    args = p.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive.")
    output = Path(args.output).resolve()
    if args.resume and output.exists():
        path = output / "config.yaml"
        metadata = json.loads((output / "run.json").read_text())
        if (metadata["config_sha256"] != file_digest(path)
            or metadata["dataset_manifest"] != verify_dataset(args.dataset)
            or metadata["model_directory"] != str(Path(args.model).resolve())
            or metadata.get("recipe_sha256") != file_digest(args.recipe)):
            raise ValueError("Resume inputs differ from the original training run.")
        if args.max_steps is not None and args.max_steps != metadata.get("max_steps"):
            raise ValueError("Resume cannot change the original optimizer-step limit.")
        if metadata.get("status") == "trained":
            export = metadata["portable_export"]
            if (file_digest(output / "portable-adapter/adapter_model.safetensors") != export["export_sha256"]
                or file_digest(output / "adapter/adapter_model.safetensors") != export["source_sha256"]):
                raise ValueError("Completed training artifacts changed after export.")
            print(json.dumps({"status": "already_trained", "output": str(output)}))
            return
    else:
        path, metadata = materialize(
            args.dataset, args.model, args.output, args.recipe, args.max_steps, args.epochs
        )
    if args.prepare_only:
        print(path)
        return
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    values = [int(v.strip()) for v in used.splitlines() if v.strip()]
    if len(values) != 2 or any(v > 1500 for v in values):
        raise ValueError("This recipe needs two available GPUs. Stop model serving and audio jobs before training.")
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        AXOLOTL_DO_NOT_TRACK="1",
        WANDB_MODE="disabled",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    import yaml

    config = yaml.safe_load(path.read_text())
    if args.epochs is not None and config["num_epochs"] != args.epochs:
        raise ValueError("Resume cannot change the original epoch limit.")
    checkpoints = sorted((p for p in (output / "adapter").glob("checkpoint-*")
                          if re.fullmatch(r"checkpoint-\d+", p.name) and (p / "trainer_state.json").is_file()),
                         key=lambda p: int(p.name.split("-")[-1]))
    if args.resume and checkpoints:
        config["resume_from_checkpoint"] = str(checkpoints[-1])
    attempt = int(metadata.get("attempt", 0)) + 1
    attempt_path = output / f"attempt-{attempt}.yaml"
    attempt_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = launch_command(attempt_path, config)
    metadata.update(attempt=attempt, resumed_checkpoint=config.get("resume_from_checkpoint"))
    metadata.update(status="running", command=command, started=now())
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    started = time.monotonic()
    with (output / f"train-attempt-{attempt}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    metadata.update(
        returncode=result.returncode,
        elapsed_seconds=round(time.monotonic() - started, 2),
        finished=now(),
    )
    try:
        if result.returncode:
            raise ValueError("Training failed; inspect train.log.")
        metadata["adapter_validation"] = verify_adapter(output)
        from .adapters import export_portable

        metadata["portable_export"] = export_portable(
            output / "adapter", output / "portable-adapter"
        )
        metadata["status"] = "trained"
        metadata["reload_status"] = "pending_strict_generation_check"
    except ValueError as exc:
        metadata.update(status="failed", error=str(exc))
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    if metadata["status"] != "trained":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
