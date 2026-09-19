"""Compare an adapter with its base under identical local generation settings."""

import argparse
import contextlib
import json
from pathlib import Path
import random
import time
import math

from .util import digest, now
from .adapters import strict_load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cases", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--loss-only", action="store_true")
    p.add_argument("--sequence-len", type=int, default=2048)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Choose a new comparison output.")
    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text().splitlines()
        if line.strip()
    ]
    if any(r["provenance"]["split"] not in {"test", "validation"} for r in rows):
        raise ValueError("Use held-out examples for adapter comparison.")
    rng = random.Random(42)
    rng.shuffle(rows)
    rows = rows[: args.cases]
    if not rows:
        raise ValueError("No evaluation examples.")
    import torch
    from transformers import (
        AutoTokenizer,
        Qwen3_5ForConditionalGeneration,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Loading common base and adapter", flush=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "21GiB", 1: "21GiB"},
        attn_implementation="sdpa",
    )
    model, reload_report = strict_load(base, args.adapter)
    print("STRICT_RELOAD", json.dumps(reload_report), flush=True)
    model.eval()
    results = []
    for row in rows:
        if args.loss_only:
            from format_conversation_dataset.tokenize import encode_example

            encoded, audit = encode_example(row, tokenizer, args.sequence_len)
            if encoded is None:
                raise ValueError(
                    "Held-out example does not fit the loss evaluation window."
                )
            tensors = {
                k: torch.tensor([v], device=model.get_input_embeddings().weight.device)
                for k, v in encoded.items()
            }
            answers = {}
            for variant in ["base", "adapter"]:
                context = (
                    model.disable_adapter()
                    if variant == "base"
                    else contextlib.nullcontext()
                )
                with context, torch.inference_mode():
                    value = float(model(**tensors, use_cache=False).loss)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite held-out loss.")
                answers[variant] = {
                    "loss": value,
                    "target_tokens": audit["completion_tokens"],
                }
            results.append(
                {"id": row["id"], "answers": answers, "provenance": row["provenance"]}
            )
            aggregate = {}
            for variant in ["base", "adapter"]:
                tokens = sum(r["answers"][variant]["target_tokens"] for r in results)
                mean = (
                    sum(
                        r["answers"][variant]["loss"]
                        * r["answers"][variant]["target_tokens"]
                        for r in results
                    )
                    / tokens
                )
                aggregate[variant] = {
                    "loss": mean,
                    "perplexity": math.exp(mean),
                    "target_tokens": tokens,
                }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "created": now(),
                        "status": "complete"
                        if len(results) == len(rows)
                        else "running",
                        "expected_cases": len(rows),
                        "dataset_sha256": digest(Path(args.dataset).read_bytes()),
                        "reload_validation": reload_report,
                        "model": args.model,
                        "adapter": args.adapter,
                        "metric": "token-weighted completion cross entropy; narrative quality requires separate review",
                        "aggregate": aggregate,
                        "cases": results,
                    },
                    indent=2,
                )
            )
            print("LOSS", len(results), aggregate, flush=True)
            continue
        ids = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
        ordering = ["base", "adapter"]
        rng.shuffle(ordering)
        answers = {}
        for variant in ordering:
            context = (
                model.disable_adapter()
                if variant == "base"
                else contextlib.nullcontext()
            )
            torch.cuda.synchronize()
            started = time.monotonic()
            with context, torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            tokens = generated[0, len(ids) :].tolist()
            answers[variant] = {
                "text": tokenizer.decode(tokens, skip_special_tokens=True),
                "seconds": round(seconds, 3),
                "tokens": len(tokens),
            }
        results.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "reference": row["completion"][0]["content"],
                "provenance": row["provenance"],
                "generation_order": ordering,
                "answers": answers,
                "review_status": "unreviewed",
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "created": now(),
                    "dataset_sha256": digest(Path(args.dataset).read_bytes()),
                    "model": args.model,
                    "adapter": args.adapter,
                    "reload_validation": reload_report,
                    "settings": {
                        "do_sample": False,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "note": "Paired generations, not a quality score. First-call timings include warm-up. Recorded responses are references, not unique gold answers.",
                    "cases": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("COMPARED", row["id"], len(results), flush=True)


if __name__ == "__main__":
    main()
