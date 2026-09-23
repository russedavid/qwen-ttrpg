"""Compare several served candidates on identical inputs, with a separate blind review."""

import argparse
import json
from pathlib import Path
import random
import statistics

from .benchmark import local_endpoint, render_review, score
from .generation import generate
from .util import digest, now, packed


def run(
    cases,
    routing,
    variants,
    url,
    model,
    output,
    *,
    seed=42,
    max_tokens=700,
    profile="production",
    generator=generate,
):
    local_endpoint(url)
    output = Path(output).expanduser().resolve()
    if output.exists() or any((p / ".git").exists() for p in [output, *output.parents]):
        raise ValueError(
            "Choose a new private comparison directory outside repositories."
        )
    if not 2 <= len(variants) <= 8 or any(
        not isinstance(k, str) or not k.strip() for k in variants
    ):
        raise ValueError("Supply two to eight named candidates.")
    ids = [a["id"] for a in routing["adapters"]]
    if ids != list(range(len(ids))) or any(
        v is not None and (type(v) is not int or v not in ids)
        for v in variants.values()
    ):
        raise ValueError(
            "Candidates must refer to the complete ordered server inventory, or null for the base."
        )
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Supply nonempty cases with unique IDs.")
    if any(
        c.get("provenance", {}).get("split") not in {"validation", "test", "synthetic"}
        for c in cases
    ):
        raise ValueError(
            "Use held-out or original evaluation cases, never the training split."
        )
    if generator is generate:
        from .annotation import verify_server

        verify_server(url, routing)
    output.mkdir(parents=True, mode=0o700)
    report = {
        "created": now(),
        "status": "running",
        "model": model,
        "seed": seed,
        "case_sha256": digest(packed(cases)),
        "routing": routing,
        "variants": variants,
        "note": "Multiple seeds repeat scenarios; they do not create new independent source groups. Schema checks do not grade story quality.",
        "cases": [],
    }
    keys, blind = {}, []
    for case in cases:
        order = list(variants)
        random.Random(f"generation:{seed}:{case['id']}").shuffle(order)
        answers = {}
        for label in order:
            try:
                answer = generator(
                    url,
                    model,
                    case["prompt"],
                    variants[label],
                    len(ids),
                    max_tokens,
                    task=case["task"],
                    sampling_profile=profile,
                    seed=seed,
                    schema=case.get("schema"),
                )
                answer["checks"] = score(answer["text"], case.get("checks", {}))
            except Exception as exc:
                answer = {
                    "text": "",
                    "complete": False,
                    "error": str(exc),
                    "checks": {"generation": False},
                }
            answers[label] = answer
        labels = list(variants)
        random.Random(f"blind:{seed}:{case['id']}").shuffle(labels)
        key = dict(zip([chr(65 + i) for i in range(len(labels))], labels))
        keys[case["id"]] = key
        row = {
            "id": case["id"],
            "task": case["task"],
            "prompt": case["prompt"],
            "provenance": case["provenance"],
            "answers": answers,
            "generation_order": order,
            "expect": case.get(
                "expect", "Review against the supplied source and current task."
            ),
            "quality_review": "pending",
        }
        report["cases"].append(row)
        blind.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "expect": row["expect"],
                "answers": {
                    letter: {
                        "text": answers[name]["text"],
                        "complete": answers[name]["complete"],
                    }
                    for letter, name in key.items()
                },
            }
        )
        (output / "results.json").write_text(packed(report))
        (output / "review-key.json").write_text(packed(keys))
        (output / "blind.json").write_text(packed(blind))
        render_review(report["cases"], keys, output / "review.html")
        print(f"{len(report['cases'])}/{len(cases)} paired cases complete", flush=True)
    report["status"] = "complete"
    report["summary"] = {}
    for label in variants:
        values = [r["answers"][label] for r in report["cases"]]
        times = [v["seconds"] for v in values if "seconds" in v]
        report["summary"][label] = {
            "attempts": len(values),
            "complete": sum(v["complete"] for v in values),
            "median_seconds": statistics.median(times) if times else None,
        }
    (output / "results.json").write_text(packed(report))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases")
    p.add_argument("--routing", required=True)
    p.add_argument(
        "--variants",
        required=True,
        help="JSON object mapping labels to adapter IDs; null selects the base",
    )
    p.add_argument("--url", default="http://127.0.0.1:8091/v1")
    p.add_argument("--model", default="qwen3.8-27b-q4")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=700)
    args = vars(p.parse_args())
    args["cases"] = json.loads(Path(args["cases"]).read_text())
    args["routing"] = json.loads(Path(args["routing"]).read_text())
    args["variants"] = json.loads(args["variants"])
    run(**args)
