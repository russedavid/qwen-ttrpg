"""Paired base/adapter evaluations with blinded review and visible latency."""

import argparse
import html
import json
from pathlib import Path
import random
import statistics
from urllib.parse import urlparse
import ipaddress

from .generation import generate
from .routing import selection
from .util import now, packed, digest


def local_endpoint(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Use a local HTTP endpoint without embedded credentials.")
    if parsed.hostname != "localhost":
        try:
            if not ipaddress.ip_address(parsed.hostname).is_loopback:
                raise ValueError("Use a loopback address; tunnel remote servers locally.")
        except (ValueError, TypeError) as exc:
            raise ValueError("Use a loopback address; tunnel remote servers locally.") from exc
    return url.rstrip("/")


def score(text, checks):
    """Mechanical checks are limited claims; prose quality remains a review task."""
    results = {"nonempty": bool(text.strip())}
    if not checks:
        return results
    try:
        value = json.loads(text)
        for key, expected in checks.get("equals", {}).items():
            current = value
            for part in key.split("."):
                current = current[int(part)] if isinstance(current, list) else current[part]
            results["equals:" + key] = current == expected
        for key in checks.get("nonempty", []):
            results["nonempty:" + key] = bool(value.get(key))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        results["parse_and_contract"] = False
    return results


def run(cases, routing, url, model, output, *, max_tokens=384, seed=42, profile="production", generator=generate):
    local_endpoint(url)
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new benchmark directory.")
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Provide nonempty cases with unique IDs.")
    for case in cases:
        if case.get("provenance", {}).get("split") not in {"validation", "test", "synthetic"}:
            raise ValueError("Benchmark only held-out or original synthetic cases.")
        _, adapter = selection(case["task"], routing)
        if adapter is None:
            raise ValueError("Each benchmark task must select a candidate adapter.")
    if generator is generate:
        from .annotation import verify_server
        verify_server(url, routing)
    output.mkdir(parents=True)
    results, keys = [], {}
    for case in cases:
        _, adapter = selection(case["task"], routing)
        order = ["base", "adapter"]
        random.Random(f"generation:{seed}:{case['id']}").shuffle(order)
        answers = {}
        for variant in order:
            answer = generator(url, model, case["prompt"], None if variant == "base" else adapter,
                               len(routing["adapters"]), max_tokens, task=case["task"],
                               sampling_profile=profile, seed=seed, schema=case.get("schema"))
            answer["checks"] = score(answer["text"], case.get("checks", {}))
            if case.get("contract_version") == 1:
                from .contracts import validate_target
                try:
                    actual = validate_target(case["task"], json.loads(case["prompt"][-1]["content"]), json.loads(answer["text"]))
                    answer["checks"]["schema_and_source_checks"] = True
                    target = case["target"]
                    if case["task"] == "classifier":
                        normalize = lambda events: sorted(packed({k: v for k, v in e.items() if k != "evidence"}) for e in events)
                        answer["checks"]["matches_reviewed_event_fields"] = normalize(actual["events"]) == normalize(target["events"])
                    elif case["task"] == "rules":
                        answer["checks"]["matches_reviewed_calculation"] = actual["calculation"] == target["calculation"]
                        answer["checks"]["matches_missing_information_decision"] = bool(actual["missing_information"]) == bool(target["missing_information"])
                except (ValueError, TypeError, KeyError) as exc:
                    answer["checks"]["schema_and_source_checks"] = False
                    answer["validation_error"] = str(exc)
            answers[variant] = answer
        # Display order must not disclose randomized execution order or adapter IDs.
        labels = ["base", "adapter"]
        random.Random(f"review:{seed}:{case['id']}").shuffle(labels)
        keys[case["id"]] = dict(zip(["A", "B"], labels))
        results.append({"id": case["id"], "task": case["task"], "prompt": case["prompt"],
                        "provenance": case["provenance"], "answers": answers,
                        "reference": case.get("target"),
                        "generation_order": order, "quality_review": "pending"})
        report = {"created": now(), "status": "complete" if len(results) == len(cases) else "running",
                  "case_sha256": digest(packed(cases)), "routing": routing, "model": model,
                  "note": "Controls and latency do not establish narrative quality. First request may include warm-up.",
                  "cases": results}
        (output / "results.json").write_text(packed(report), encoding="utf-8")
        (output / "review-key.json").write_text(packed(keys), encoding="utf-8")
        render_review(results, keys, output / "review.html")
    report["summary"] = {
        variant: {"cases": len(results),
                  "complete": sum(r["answers"][variant]["complete"] for r in results),
                  "control_passes": sum(r["answers"][variant]["complete"] and
                                        all(r["answers"][variant]["checks"].values()) for r in results),
                  "median_seconds": statistics.median(r["answers"][variant]["seconds"] for r in results)}
        for variant in ["base", "adapter"]}
    (output / "results.json").write_text(packed(report), encoding="utf-8")
    return report


def render_review(rows, keys, path):
    esc = lambda value: html.escape(value if isinstance(value, str) else json.dumps(value, indent=2))
    parts = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
             '<title>Response review</title><style>body{font:17px system-ui;max-width:1100px;margin:auto;padding:24px;background:#f7f1e7;color:#342c24}pre{white-space:pre-wrap;overflow-wrap:anywhere}.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:24px}article{border-top:1px solid #bba;padding:24px 0}nav{display:flex;flex-wrap:wrap;gap:12px}</style>',
             '<h1>Response review</h1><p>Review relevance, continuity, participant agency, unsupported facts, and usefulness. A/B identities and timing are kept in separate files.</p><nav>']
    parts += [f'<a href="#case-{i}">{i+1}</a>' for i in range(len(rows))]
    parts.append('</nav>')
    for i, row in enumerate(rows):
        parts += [f'<article id="case-{i}"><h2>Case {i+1}</h2><pre>{esc(row["prompt"])}</pre><div class="answers">']
        for label in sorted(keys[row["id"]]):
            answer = row["answers"][keys[row["id"]][label]]
            warning = '<p><strong>Incomplete response.</strong> This generation did not finish.</p>' if not answer.get("complete", True) else ""
            parts.append(f'<section><h3>{label}</h3>{warning}<pre>{esc(answer["text"])}</pre></section>')
        parts.append('</div></article>')
    Path(path).write_text("".join(parts), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases", help="Local JSON array of evaluation cases")
    p.add_argument("--routing", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8091/v1")
    p.add_argument("--model", default="qwen3.8-27b-q4")
    p.add_argument("--output", required=True)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", choices=["production", "greedy"], default="production")
    args = p.parse_args()
    report = run(json.loads(Path(args.cases).read_text()), json.loads(Path(args.routing).read_text()),
                 args.url, args.model, args.output, max_tokens=args.max_tokens, seed=args.seed, profile=args.profile)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
