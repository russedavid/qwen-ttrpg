"""Prepare, train, and assess a small source-grounded decision policy locally."""

import argparse
from collections import defaultdict
from copy import deepcopy
import importlib
import json
from pathlib import Path
import random
import statistics
import time

from .util import digest, file_digest, now
from .rl_rollout import AgentRollouts, append_context, encode_prompt, episode_reward, run_episode


def new_private(path):
    path = Path(path).expanduser().resolve()
    if path.exists() or any((p / ".git").exists() for p in [path, *path.parents]):
        raise ValueError("Choose a new private output directory outside repositories.")
    path.mkdir(parents=True, mode=0o700)
    return path


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temporary.replace(path)


def environment():
    return importlib.import_module("story_copilot.rl_environment")


def prepare(output):
    env = environment()
    out = new_private(output)
    groups = {split: env.make_suite(count, split=split)
              for split, count in [("train", 24), ("validation", 4), ("test", 4), ("transfer", 4)]}
    hashes = {}
    for split, cases in groups.items():
        path = out / f"{split}.json"
        write(path, cases)
        hashes[path.name] = file_digest(path)
    write(out / "demonstrations.json", env.demonstrations(groups["train"]))
    hashes["demonstrations.json"] = file_digest(out / "demonstrations.json")
    report = {"version": 1, "environment": env.VERSION, "created": now(),
              "provenance": "Original authored procedural scenarios; no source recordings or transcripts.",
              "counts": {k: len(v) for k, v in groups.items()}, "files": hashes,
              "environment_source_sha256": file_digest(Path(env.__file__))}
    write(out / "manifest.json", report)
    return report


def check_data(path):
    path = Path(path).expanduser().resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_digest(path / name) != expected:
            raise ValueError("The frozen data snapshot changed.")
    if file_digest(Path(environment().__file__)) != manifest["environment_source_sha256"]:
        raise ValueError("The environment changed; create a new snapshot and baseline.")
    return path, manifest


def load_model(path, adapter=None, *, trainable=False):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from trl.chat_template_utils import get_training_chat_template
    from .adapters import strict_load

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.chat_template = get_training_chat_template(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    if adapter:
        model, report = strict_load(model, adapter, trainable=trainable)
        model._rl_reload_report = report
    return model, tokenizer


def lora_config():
    from peft import LoraConfig
    # Only language decoder projections. No vision, embeddings, or output head.
    return LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))")


def training_record(args, manifest):
    from importlib.metadata import version
    return {"status": "running", "created": now(), "arguments": vars(args), "dataset": manifest,
            "packages": {name: version(name) for name in ["torch", "transformers", "trl", "peft", "accelerate"]}}


def train(args):
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
    from peft import get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    data, manifest = check_data(args.data)
    starting_step = 0
    if args.resume_checkpoint:
        checkpoint = Path(args.resume_checkpoint).expanduser().resolve()
        previous = json.loads((checkpoint.parent.parent / "run.json").read_text())
        if previous["dataset"] != manifest or previous["arguments"]["phase"] != args.phase:
            raise ValueError("Resume requires the same data snapshot and training phase.")
        for key in ["model", "seed", "context_limit", "max_new_tokens", "learning_rate"]:
            if previous["arguments"][key] != getattr(args, key):
                raise ValueError(f"Resume setting changed: {key}.")
        if not args.adapter or Path(args.adapter).resolve() != checkpoint:
            raise ValueError("Select the same checkpoint as --adapter and --resume-checkpoint.")
        starting_step = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
        if args.steps <= starting_step:
            raise ValueError("The total step target must exceed the saved step.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one training GPU with CUDA_VISIBLE_DEVICES.")
    if torch.cuda.mem_get_info()[0] < 14 * 1024**3:
        raise ValueError("The selected GPU needs at least 14 GiB free before this pilot.")
    out = new_private(args.output)
    record = training_record(args, manifest)
    record["starting_step"] = starting_step
    write(out / "run.json", record)
    set_seed(args.seed)
    started = time.monotonic()
    model, tokenizer = load_model(args.model, args.adapter, trainable=True)
    model.config.use_cache = False
    record["chat_template_sha256"] = digest(tokenizer.chat_template)
    record["initial_reload"] = getattr(model, "_rl_reload_report", None)
    if args.adapter is None:
        model = get_peft_model(model, lora_config())
    model.print_trainable_parameters()
    initial_tensors = {name: value.detach().cpu().clone()
                       for name, value in model.named_parameters() if value.requires_grad}

    class Monitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            sample = {"step": state.global_step, "seconds": time.monotonic() - started,
                      "logs": logs, "allocated": torch.cuda.max_memory_allocated(),
                      "reserved": torch.cuda.max_memory_reserved()}
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(sample) + "\n")

    common = dict(output_dir=str(out / "checkpoints"), max_steps=args.steps,
                  per_device_train_batch_size=1, gradient_accumulation_steps=4,
                  learning_rate=args.learning_rate, bf16=True, logging_steps=1,
                  save_strategy="steps", save_steps=20, save_total_limit=2,
                  report_to=[], seed=args.seed, data_seed=args.seed,
                  gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant": False},
                  max_grad_norm=1.0, lr_scheduler_type="constant", warmup_steps=0)
    if args.phase == "sft":
        examples = json.loads((data / "demonstrations.json").read_text())
        encoded = []
        for example in examples:
            prompt = encode_prompt(tokenizer, example["prompt"])
            completion = tokenizer.encode(example["completion"], add_special_tokens=False) + [tokenizer.eos_token_id]
            append_context(prompt, completion, encode_prompt(tokenizer, example["prompt"] + [
                {"role": "assistant", "content": example["completion"]},
                {"role": "user", "content": "Evidence result: []"},
            ]))
            ids = prompt + completion
            if len(ids) > args.context_limit:
                raise ValueError("A complete demonstration exceeds the context budget.")
            encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                            "labels": [-100] * len(prompt) + completion})

        def collate(rows):
            size = max(len(row["input_ids"]) for row in rows)
            return {key: torch.tensor([row[key] + [fill] * (size - len(row[key])) for row in rows])
                    for key, fill in [("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)]}

        trainer = Trainer(model=model, args=TrainingArguments(**common),
                          train_dataset=Dataset.from_list(encoded), data_collator=collate,
                          processing_class=tokenizer, callbacks=[Monitor()])
    else:
        cases = json.loads((data / "train.json").read_text())
        rollouts = AgentRollouts(cases, environment().EvidenceEpisode, tokenizer,
                                out / "rollouts.jsonl", max_new_tokens=args.max_new_tokens,
                                context_limit=args.context_limit)
        config = GRPOConfig(**common, num_generations=4, generation_batch_size=4,
                            max_completion_length=args.context_limit, temperature=1.0,
                            top_p=1.0, top_k=0, beta=0.0, use_vllm=False,
                            loss_type="dapo", mask_truncated_completions=False,
                            chat_template_kwargs={"enable_thinking": False})
        trainer = GRPOTrainer(model=model, args=config,
                              train_dataset=Dataset.from_list([{"prompt": c["prompt"]} for c in cases]),
                              processing_class=tokenizer, rollout_func=rollouts,
                              reward_funcs=[episode_reward], callbacks=[Monitor()])
    try:
        result = trainer.train(resume_from_checkpoint=args.resume_checkpoint)
        trainer.save_model(str(out / "adapter"))
        tokenizer.save_pretrained(out / "adapter")
        state = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        finite = all(torch.isfinite(value).all().item() for value in state.values())
        changed = any(torch.count_nonzero(value).item() for name, value in state.items() if "lora_B" in name)
        updated = sum(not torch.equal(initial_tensors[name], value.detach().cpu())
                      for name, value in state.items())
        if not finite or not changed:
            raise ValueError("Adapter must have finite, nonzero updates.")
        record.update(status="complete", metrics=result.metrics, final_step=trainer.state.global_step,
                      seconds=round(time.monotonic() - started, 3),
                      peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                      adapter_sha256=file_digest(out / "adapter" / "adapter_model.safetensors"),
                      trainable_tensors=len(state), updated_tensors=updated,
                      finite=finite, nonzero_lora_b=changed)
    except BaseException as exc:
        record.update(status="failed", error=str(exc), seconds=time.monotonic() - started)
        raise
    finally:
        write(out / "run.json", record)
    return record


def evaluate(args):
    import torch
    from transformers import set_seed

    data, manifest = check_data(args.data)
    out = new_private(args.output)
    model, tokenizer = load_model(args.model, args.adapter)
    model.eval()
    cases = json.loads((data / f"{args.split}.json").read_text())
    if args.limit:
        rng = random.Random(17)
        rng.shuffle(cases)
        cases = cases[:args.limit]
    report = {"status": "running", "created": now(), "dataset": manifest,
              "candidate": args.label, "adapter": args.adapter, "cases": [],
              "reload_validation": getattr(model, "_rl_reload_report", None),
              "chat_template_sha256": digest(tokenizer.chat_template),
              "settings": {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
                           "context_limit": args.context_limit, "seeds": args.seeds}}
    for seed in args.seeds:
        for case in cases:
            set_seed(seed + int(case["id"][:6], 16))
            row = run_episode(model, tokenizer, case, environment().EvidenceEpisode,
                              temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                              context_limit=args.context_limit)
            row["generation_seed"] = seed
            row["expected"] = case["expected"]
            row["acceptable_sources"] = case["source_alternatives"]
            row["prompt"] = case["prompt"]
            row["source_context"] = case["context"]
            report["cases"].append(row)
            report["summary"] = summarize(report["cases"])
            write(out / "report.json", report)
            print(args.label, len(report["cases"]), row["family"], row["assessment"], flush=True)
    report.update(status="complete", peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
    write(out / "report.json", report)
    return report["summary"]


def summarize(rows):
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row["assessment"])
    def summary(items):
        return {"cases": len(items), "successes": sum(x["success"] for x in items),
                "correct_conclusions": sum(x["correct_fields"] for x in items),
                "grounded_citations": sum(x["grounded"] for x in items),
                "truncated": sum(x["truncated"] for x in items),
                "invalid_actions": sum(x["invalid_actions"] for x in items),
                "mean_tool_calls": statistics.mean(x["tool_calls"] for x in items),
                "unnecessary_calls": sum(x["unnecessary_calls"] for x in items),
                "median_seconds": statistics.median(x["seconds"] for x in items)}
    return {"overall": summary([r["assessment"] for r in rows]),
            "families": {k: summary(v) for k, v in by_family.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    for command in ["train", "evaluate"]:
        p = sub.add_parser(command)
        p.add_argument("--data", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--adapter")
        p.add_argument("--output", required=True)
        p.add_argument("--context-limit", type=int, default=3072)
        p.add_argument("--max-new-tokens", type=int, default=160)
        if command == "train":
            p.add_argument("--phase", choices=["sft", "grpo"], required=True)
            p.add_argument("--steps", type=int, default=20)
            p.add_argument("--learning-rate", type=float, default=0.00001)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--resume-checkpoint")
        else:
            p.add_argument("--split", choices=["validation", "test", "transfer"], default="validation")
            p.add_argument("--limit", type=int)
            p.add_argument("--seeds", type=int, nargs="+", default=[42, 314])
            p.add_argument("--label", default="base")
            p.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    if args.command == "train" and (args.steps < 1 or args.learning_rate <= 0):
        parser.error("Use positive training steps and learning rate.")
    try:
        value = prepare(args.output) if args.command == "prepare" else train(args) if args.command == "train" else evaluate(args)
    except BaseException as exc:
        record = Path(args.output) / "run.json"
        if record.is_file():
            value = json.loads(record.read_text())
            if value.get("status") == "running":
                value.update(status="failed", error=str(exc))
                write(record, value)
        raise
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
