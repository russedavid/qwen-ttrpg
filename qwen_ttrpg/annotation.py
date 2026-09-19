"""Local base-model label proposals; generation never masquerades as human review."""

import json
from pathlib import Path
from uuid import uuid4

import httpx

from .benchmark import local_endpoint
from .contracts import validate_target
from .data import load_review, review_path, render_review, write_json
from .generation import generate
from .util import now


def verify_server(url, routing):
    local_endpoint(url)
    ids = [a["id"] for a in routing["adapters"]]
    if ids != list(range(len(ids))):
        raise ValueError("Adapter IDs must match the ordered server inventory.")
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(url.rstrip("/").removesuffix("/v1") + "/lora-adapters")
        response.raise_for_status()
        inventory = response.json()
    if len(inventory) != len(ids):
        raise ValueError("Server adapter inventory differs from routing configuration.")
    for actual, expected in zip(inventory, routing["adapters"]):
        if actual["id"] != expected["id"] or actual["path"] != expected["path"]:
            raise ValueError("The running server loaded a different adapter or adapter order.")
    return inventory


def annotate(config, *, task, routing, url, model, limit=None, generator=generate):
    if task not in {"classifier", "rules"}:
        raise ValueError("Narrator targets come from source responses, not generated replacements.")
    local_endpoint(url)
    if generator is generate:
        verify_server(url, routing)
    document = load_review(config)
    selected = [c for c in document["candidates"] if c["task"] == task and c["target"] is None and c["review"]["verdict"] == "pending"]
    if limit is not None:
        if limit < 1:
            raise ValueError("Annotation limit must be positive.")
        selected = selected[:limit]
    complete = 0
    for case in selected:
        record = {"created": now(), "case_id": case["id"], "model": model, "origin": "model-proposed", "adapter": None}
        try:
            answer = generator(url, model, case["prompt"], None, len(routing["adapters"]), 2048,
                               task=task, sampling_profile="greedy", schema=case["schema"])
            record["generation"] = answer
            if not answer["complete"]:
                raise ValueError("Label generation was truncated or incomplete.")
            case["target"] = validate_target(task, case["input"], json.loads(answer["text"]))
            case["label_origin"] = "model-proposed"
            complete += 1
            record["status"] = "proposal_needs_review"
        except Exception as exc:
            record.update(status="failed", error=str(exc))
        path = Path(config["workspace"]) / "annotations" / (case["id"] + "-" + uuid4().hex[:8] + ".json")
        write_json(path, record, exclusive=True)
        case["annotation"] = str(path)
        # Original pending verdict remains; shape and quote validation is not a
        # semantic review and does not turn generated labels into ground truth.
        write_json(review_path(config), document)
        render_review(document, review_path(config).with_suffix(".html"))
    return {"attempted": len(selected), "proposed": complete, "needs_review": complete}
